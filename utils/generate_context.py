import pandas as pd
from utils import faiss_search
from pprint import pprint

def generate_context(pkl, question, model_name, num_results):
    pprint(f'generate context question----{question}')
    results = pd.DataFrame(faiss_search.faiss_search(pkl, question, model_name, num_results= num_results), 
                           columns = ['results', 'faiss_score']) # Set num_results to 3 or 5
    results["context"] = results["results"].apply(lambda x: x.page_content)
    results["page_number"] = results["results"].apply(lambda x: x.metadata["page_number"])
    results = results[["context", "page_number", "faiss_score"]]
    pprint(f'generate context results----{results}')
    context = ' '.join(results["context"].values)
    return context