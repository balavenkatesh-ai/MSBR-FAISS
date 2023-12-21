import streamlit as st
from utils import df_from_doc, faiss_search, create_search_index, wrap_print, generate_context, load_llm, run_llm
import pathlib
from tempfile import NamedTemporaryFile

st.title("MSBR AutmotaionX")
st.subheader("LLM - Threat Library Mapping Tool")

st.divider()

uploaded_file = st.file_uploader("Upload a Single Document:", accept_multiple_files=False, type=['pdf', 'txt','csv'])

if uploaded_file is not None:
    filetype = pathlib.Path(uploaded_file.name).suffix
    with NamedTemporaryFile(dir='.', suffix=filetype) as f:
        f.write(uploaded_file.getbuffer())
        docs = df_from_doc.df_from_doc(f.name, str(filetype).replace(".", ""))

    #model_name = st.selectbox("Select Sentence-Transformers Model for Embeddings:", ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1"])
    model_name = 'all-MiniLM-L6-v2'
    pkl = create_search_index.create_search_index(docs, model_name)

    question = st.text_input("Ask a Question:")

    if st.button("Get Threat Mapping Data") and question.strip():
        context_df = generate_context.extract_mitre_description(pkl, question, model_name, num_results=3)
        st.write("Extracted Data:")
        st.dataframe(context_df)
        
        # ---------------Get vector results ----------------#
        
        # context = generate_context.generate_context(pkl, question, model_name, num_results=1)
        # st.write("Estimated Context Length:", round(4/3*len(context.split())), "tokens", "\n")
        # st.write("context:", context)
        # wrap_print.wrap_print(context)
        
        # ---------------Call LLM model ----------------#

        # model_path_13b = "/root/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGML/snapshots/3140827b4dfcb6b562cd87ee3d7f07109b014dd0/llama-2-13b-chat.ggmlv3.q5_1.bin"
        # model_path_7b = ""

        # model_type = st.selectbox("Select LLM Type:", ["LLaMA-7B", "LLaMA-13B"])
        # llm = load_llm.load_llm(model_type=model_type, model_path=model_path_13b)
        # context_dependency = st.selectbox("Select Context Dependence Level (set to low if the model is failing to generate context dependent answers):", ["low", "medium", "high"])

        # st.write("Running, may take up to 60 seconds...")
        # with st.spinner(f"Running {model_type}..."):
        #     output = run_llm.run_llm(llm, question, context, context_dependency)

        # if output["choices"][0]["text"].split("###")[4][-1] != ".":
        #     st.success(output["choices"][0]["text"].split("###")[4] + "...")
        # elif output["choices"][0]["text"].split("###")[4][-1] == ".":
        #     st.success(output["choices"][0]["text"].split("###")[4])
