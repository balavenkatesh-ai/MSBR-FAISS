from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader ,CSVLoader,TextLoader
from utils import clean_text, est_words_tokens
import pandas as pd

def df_from_doc(filepath, filetype):
    if filetype == "pdf":
        loader = PyPDFLoader(filepath)
        docs = pd.DataFrame(loader.load_and_split(text_splitter = RecursiveCharacterTextSplitter(chunk_size=1028, chunk_overlap=128))
                            , columns = ['text', 'page_number'])
        docs["text"] = docs["text"].apply(lambda x: x[1]).apply(clean_text.clean_text); 
        docs["page_number"] = docs["page_number"].apply(lambda x: 1 + x[1]["page"])
        docs = est_words_tokens.est_words_tokens(docs)
    elif filetype == "txt":
        loader = TextLoader(filepath)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1028, chunk_overlap=128)
        docs = pd.DataFrame(text_splitter.split_documents(documents), columns = ['text', 'page_number'])
        docs["text"] = docs["text"].apply(lambda x: x[1]).apply(clean_text.clean_text); docs["page_number"] = 1
        docs = est_words_tokens.est_words_tokens(docs)
    elif filetype == "csv":
        loader = CSVLoader(filepath, encoding="utf8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        docs = pd.DataFrame({"text": docs, "page_number": range(1, len(docs) + 1)})
    return docs