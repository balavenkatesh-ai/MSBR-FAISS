import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
#from utils import load_llm
from ctransformers import AutoModelForCausalLM

DB_FAISS_PATH = 'vectorstore/db_faiss'

model_path_13b = "/root/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/3140827b4dfcb6b562cd87ee3d7f07109b014dd0/llama-2-13b-chat.ggmlv3.q5_1.bin"

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    # llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id = model_path_13b, 
                                                # model_type="llama",gpu_layers=50, context_length = 14096,max_new_tokens = 14096)
    #llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GGML", gpu_layers=50, context_length = 14096,max_new_tokens = 14096)
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/WizardLM-7B-uncensored-GGML",
                                               model_file="WizardLM-7B-uncensored.ggmlv3.q6_K.bin",
                                                model_type="llama",context_length=2048,
                                                )
    # llm = CTransformers(
    #     model = model_path_13b,
    #     model_type="llama",
    #     max_new_tokens = 14096,
    #     context_length = 14096,
    #     temperature = 0.3
    # )
    return llm

st.title("MSBR AutmotaionX")
st.subheader("LLM - Chat with Threat Library")

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file :
   #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()
    #st.json(data)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    #db.save_local(DB_FAISS_PATH)
    
    #llm = load_llm.load_llm(model_type='LLaMA-13B', model_path=model_path_13b)

    llm = load_llm()
    #llm_dict = {"llm": llm}
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")



    

