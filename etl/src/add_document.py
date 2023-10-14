from src.utils.load_utils import load_documents, load_db, get_vectorizer

db = load_db(embedding_function=get_vectorizer())
db.add_documents(load_documents("new_document/"))
