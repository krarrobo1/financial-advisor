from src.utils.load_utils import (
    load_documents,
    save_db,
    get_vectorizer,
    load_db,
)

embedding_function = get_vectorizer()
documents = load_documents("documents/")

save_db(documents, embedding_function)

db = load_db(embedding_function)

print(db.similarity_search("1-800-Flowers.com"))
