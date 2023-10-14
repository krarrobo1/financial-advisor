from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from glob import glob
from tqdm import tqdm

def load_embeddings(model_name='multi-qa-MiniLM-L6-cos-v1'):
    return SentenceTransformerEmbeddings(model_name=model_name)

def load_documents_NLTK(directory):
    text_splitter = NLTKTextSplitter()
    documents = []
    for item_path in tqdm(glob(directory + "*.pdf")):
        loader = PyPDFLoader(item_path)
        documents.extend(loader.load_and_split(text_splitter=text_splitter))

    return documents

def load_documents(directory : str, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    for item_path in tqdm(glob(directory + "*.pdf")):
        try:
            loader = PyPDFLoader(item_path)
            documents.extend(loader.load_and_split(text_splitter=text_splitter))
        except:
            continue
    return documents

def load_documents_directory(directory):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    documents = []
    loader = DirectoryLoader(directory, use_multithreading=True, silent_errors=True, show_progress=True)
    documents.extend(loader.load_and_split(text_splitter=text_splitter))

    return documents

def load_documents_2(directory):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 520, chunk_overlap = 40)
    documents = []
    for item_path in tqdm(glob(directory + "*.pdf")):
        loader = PyMuPDFLoader(item_path)
        # loader = DirectoryLoader(item_path, use_multithreading=True, silent_errors=True)
        documents.extend(loader.load_and_split(text_splitter=text_splitter))

    return documents

def load_db(embedding_function, persistence_directory='demodb/'):
    db = Chroma(persist_directory=persistence_directory, embedding_function=embedding_function)
    return db

def split_docs_in_batches(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

def save_db(documents,embedding_function, persistence_directory='chroma_db-database'):
    docs_batches = split_docs_in_batches(documents, 100)

    for doc_batch in tqdm(docs_batches):
        db = Chroma.from_documents(
            documents=doc_batch,
            embedding=embedding_function,
            persist_directory=persistence_directory
        )

        db.persist()