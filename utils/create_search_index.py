from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.docstore.document import Document

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_search_index(docs, model_name):
    documents = [Document(page_content= row["text"], metadata= {"page_number": row["page_number"]}) for row in docs.to_dict('records')]
    embedding_model = SentenceTransformerEmbeddings(model_name= f"sentence-transformers/{model_name}")
    search_index = FAISS.from_documents(documents, embedding_model)
    pkl = search_index.serialize_to_bytes()
    return pkl

def create_csv_search_index(docs, model_name):
    embedding_model = SentenceTransformerEmbeddings(model_name= f"sentence-transformers/{model_name}")
    search_index = FAISS.from_documents(docs, embedding_model)
    pkl = search_index.serialize_to_bytes()
    return pkl