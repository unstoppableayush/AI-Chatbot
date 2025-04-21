# Step 1: Setup UI for chatbot

# phase 1 imports
import streamlit as st

# phase 2 imports
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.title("RAG Chatbot")
st.write("This is a simple chatbot that uses RAG (Retrieval-Augmented Generation) to answer questions based on a given document.")


#setup a session state variable to hold all the old messages

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
   st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Pass your prompt here!")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})

    # Create a prompt template
    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are very smart at everything, you always give the best, 
            the most accurate and most precise answers. Answer the following Question: {user_prompt}.
            Start the answer directly. No small talk please
        """)


    # Call the LLM with the prompt
    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
    )

    # print("groq_sys_prompt", groq_sys_prompt)
    # print("groq_chat", groq_chat)
    chain = groq_sys_prompt | groq_chat | StrOutputParser()
    response = chain.invoke({"user_prompt": prompt})

    # response = "I am your Assistant!"
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({'role':'assistant','content':response})