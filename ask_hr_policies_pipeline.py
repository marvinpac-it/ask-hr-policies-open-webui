"""
title: HR Policies RAG
author: Olivier Boudry
date: 2024-08-16
version: 1.0
license: MIT
description: A pipeline for a RAG application to retrieve and generate answers based on user queries on HR documents.
requirements: langchain, langchain_openai, langchain_chroma, langchain_core, pydantic, openai
"""

import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

class Pipeline:

    class Valves(BaseModel):
        CHAT_MODEL: str
        TEMPERATURE: float
    
    def __init__(self):
        # Setup variables for retrieval and model
        self.id = "ask_hr_policies"
        self.name = "Ask HR Policies"
        self.valves = self.Valves(**{"CHAT_MODEL": "gpt-4o-mini", "TEMPERATURE": 0.0})
        self.documents = None
        self.index = None
        self.history_aware_retriever = None
        self.rag_chain = None
        self.llm = None

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        # Optional setup code when the server starts up.
        # Initialize the LLM and embedding
        embedding = OpenAIEmbeddings()
        vectordb = Chroma(embedding_function=embedding, persist_directory="/app/backend/data/chroma")
        retriever = vectordb.as_retriever()

        self.llm = ChatOpenAI(model_name=self.valves.CHAT_MODEL, temperature=self.valves.TEMPERATURE)

        # Create a history-aware retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # Create the RAG chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Keep the answer concise."
            "\\n\\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)

    async def on_shutdown(self):
        # Optional cleanup code when the server shuts down.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        try:
            for chunk in self.rag_chain.stream({"input": user_message, "chat_history": messages}):
                if "answer" in chunk:
                    yield chunk["answer"]

        except Exception as e:
            print(f"Error in pipe: {str(e)}")
            return (f"Error in pipe: {str(e)}")

