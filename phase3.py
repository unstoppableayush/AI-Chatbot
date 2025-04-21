# Step 1: Setup UI for chatbot

# phase 1 imports
import streamlit as st

# phase 2 imports
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# phase 3 imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA




st.title("RAG Chatbot")
st.write("This is a simple chatbot that uses RAG (Retrieval-Augmented Generation) to answer questions based on a given document.")


#setup a session state variable to hold all the old messages

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_vectorstore():
    pdfname = "./OperatingSystems.pdf"
    loaders = [PyPDFLoader(pdfname)]
    # create chunks, aka vectors (ChromaDb)
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
    ).from_loaders(loaders)
    return index.vectorstore

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

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the document")
        
        chain=RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs=({"k":3})),
            return_source_documents = True
        )
        result = chain({"query":prompt})
        response = result["result"]

        # print("groq_sys_prompt", groq_sys_prompt)
        # print("groq_chat", groq_chat)
        # chain = groq_sys_prompt | groq_chat | StrOutputParser()
        # response = chain.invoke({"user_prompt": prompt})

        # response = "I am your Assistant!"
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({'role':'assistant','content':response})
    except Exception as e:
        st.error(f"Error: [{e}]")