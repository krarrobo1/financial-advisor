import streamlit as st

from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor

import os
from PyPDF2 import PdfReader
import requests
from time import sleep

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

#AVATARS
av_us = './user.png' #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = './yoda.png'


# FUNCTION TO LOG ALL CHAT MESSAGES INTO chathistory.txt
def write_history(text):
    with open('chathistory.txt', 'a') as f:
        f.write(text)
        f.write('\n')
        return True
    

def get_history():
    with open('chathistory.txt', 'r') as f:
        history = f.read()
        return history


with st.sidebar:
        #load_main = False
        #st.write("NEW")
        openaikey = st.text_input("OPENAI_API_KEY", type="password")
        os.environ["OPENAI_API_KEY"] = openaikey
        #LOAD TOKEN
        if st.button("Add Token"):
             tokenresponse = requests.get(f'http://money1:8000/load_token?token={openaikey}')
        #st.write(openaikey)
        st.divider()
        st.subheader("Your finance documents")
        uploaded_files = st.file_uploader(
            "Upload your PDFs here and click on 'Add Data'", accept_multiple_files=True
        )
        if st.button("Add Data") and openaikey:
            
            st.write(tokenresponse.content.decode("utf-8"))

            with st.spinner("Adding Data..."):
                
                documents = []
                url = 'http://money1:8000/load_pdfs'
                data_list = []
                counter = 1
                for f in uploaded_files:
                    data = ("files", f)
                    data_list.append(data)
                st.write(len(data_list))
                response = requests.post(url, files=data_list)
                st.write(response.content)
                # Initialize the Chroma vectorstore - TODO API VERSION
                #embeddings = OpenAIEmbeddings()
                #vectorstore = Chroma(persist_directory="../db", embedding_function=embeddings)
            
        on = st.toggle('ACTIVATE GECKO')


if on:
        st.write('GECKO HERE!')
        llm = OpenAI(temperature=0.5, streaming=True)
        tools = load_tools(["ddg-search"])
        agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        if prompt := st.chat_input():
             st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
             st_callback = StreamlitCallbackHandler(st.container())
             response = agent.run(prompt, callbacks=[st_callback])
             st.write(response)
else:
        st.header("Welcome to MoneyMaker! 💵")
        col1, col2 = st.columns(2)
        with col1:
            st.image("moneymaking.png", width=250)
        with col2:
            st.write('<p>Welcome to <strong>MoneyMaker</strong> -- your answer to all your financial questions. In order to help you bear in mind the following instructions:</p>', 
            unsafe_allow_html=True)
            st.write('<ol><li>Enter your Open AI key</li><li>Load your financial documents in PDF</li></ol>', 
            unsafe_allow_html=True)
            st.write('<p>Oh and if you are brave give <strong>GECKO</strong> a try :)</p>', 
            unsafe_allow_html=True)
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
                
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message(message["role"],avatar=av_us):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"],avatar=av_ass):
                    st.markdown(message["content"])
                    
        # ACCEPT USER INPUT
        if myprompt := st.chat_input("Greetings stranger, I am MoneyMaker a genius in finance, how can I help you?"):
            # ADD CHAT HISTORY
            st.session_state.messages.append({"role": "user", "content": myprompt})
            
            # USER MESSAGES
            with st.chat_message("user", avatar=av_us):
                st.markdown(myprompt)
                usertext = f"user: {myprompt}"
                write_history(usertext)

            # CHATBOT MESSAGES
            with st.chat_message("assistant", avatar=av_ass):
                message_placeholder = st.empty()
                full_response = ""
                apiresponse = requests.get(f'http://money1:8000/model?question={myprompt}')
                risposta = apiresponse.content.decode("utf-8")
                res  =  risposta[1:-1]
                response = res.split(" ")
                #TYPING EFFECT
                for r in response:
                    full_response = full_response + r + " "
                    message_placeholder.markdown(full_response + "▌")
                    sleep(0.1)
                message_placeholder.markdown(full_response)
                asstext = f"assistant: {full_response}"
                write_history(asstext)
                st.session_state.messages.append({"role": "assistant", "content": full_response})