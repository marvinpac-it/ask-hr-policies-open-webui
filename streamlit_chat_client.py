import streamlit as st

import os
from dotenv import load_dotenv, find_dotenv

import openai

from langchain.chains import (create_history_aware_retriever, create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import (ChatOpenAI, OpenAIEmbeddings)
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage

# Chargement de la clé d'API OpenAI
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# Création d'un retriever de documents
embedding = OpenAIEmbeddings()
vectordb = Chroma(embedding_function=embedding, persist_directory="./chromadb")
retriever = vectordb.as_retriever()

# Création du modèle
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Création d'un retriever gérant l'historique
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
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

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Création de la chaine RAG
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Interface
st.title("Ask about Marvinpac policies")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Your question here!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Dissplay user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = map(
            lambda chunk: chunk["answer"],
            filter(
                lambda chunk: "answer" in chunk,
                rag_chain.stream({"input": prompt, "chat_history": st.session_state.messages})
            )
        )
        response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})