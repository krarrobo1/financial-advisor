#UNCOMMENT FOR STREAMLIT CLOUD
#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# API import Section
from fastapi import FastAPI, File, UploadFile

# Langchain imports
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# Helper import
from typing import List
from PyPDF2 import PdfReader
import copy
import os

#from dotenv import load_dotenv
#load_dotenv()  # take environment variables from .env.


app = FastAPI(
    title="API Server",
    description="API Server for MoneyMaker",
    version="1.0",
)

def get_llm():
    llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo", streaming=True)
    return llm


def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def load_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=text_chunks, 
        embedding=embeddings,
        persist_directory="./db")
    vectorstore.add_texts(texts=text_chunks)
    return True


def get_vectorstore(search=False):
    if search == True:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="./db", 
            embedding_function=embeddings,
        ).as_retriever(search_type = 'similarity', 
                    search_kwargs = {"k": 3})
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
        persist_directory="./db", 
        embedding_function=embeddings)
    return vectorstore


def get_prompt():
    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question. If you dont know the answer, just say that you don't know, don't try to make up an answer:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    return prompt


def get_compressor():
    db = get_vectorstore(True)
    compressor = LLMChainExtractor.from_llm(llm=get_llm())
    compressor_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=db)
    return compressor_retriever


#INCORPORATE REST GET HISTORY
def get_model():
    compressor_retriever = get_compressor()
    prompt = get_prompt()
    model = RetrievalQA.from_chain_type(
    llm = get_llm(),
    chain_type = 'stuff',
    retriever=compressor_retriever,
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferWindowMemory(
            memory_key="history",
            input_key="question",
            k=10,
            return_messages=True,
            ),
        }
    )
    return model


@app.get('/load_token')
async def load_token(token:str):
    global qa
    os.environ["OPENAI_API_KEY"] = token
    print('TOKEN',token)
    if os.getenv('OPENAI_API_KEY'):
        qa = get_model()
        print('QA SET')
        return {"message":True}
    else:
        print('QA NOT SET')
        return {"message":False}


#LOAD SEVERAL PDFs ENDPOINT
@app.post('/load_pdfs')
async def load_pdfs(files: List[UploadFile]):
    archives = [f.file for f in files]
    #print(archives)
    text = get_pdf_texts(archives)
    #get chunks
    chunks = get_text_chunks(text)
    #get vector
    vector = load_vectorstore(chunks)
    return {"message":"PDF conversion succesful"}



@app.get('/model')
async def model(question : str):
    result = qa({"query": question})
    result = copy.deepcopy(result['result'])
    return result