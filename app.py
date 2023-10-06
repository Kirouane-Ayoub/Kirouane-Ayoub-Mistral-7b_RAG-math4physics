import streamlit as st 
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import  pipeline
from langchain.text_splitter import CharacterTextSplitter
import torch
from torch import cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
st.set_page_config(page_title='Mistral-7b_RAG-math4physics')
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
model_name = "bn22/Mistral-7B-Instruct-v0.1-sharded"
st.header(":hand: Welcome To Mistral-7b_RAG-math4physics Chatbot : ")
st.write("""
+  You can change temperature,top-ranked, and top_p values from the slider.\n
+  This chatbot Uses all-MiniLM-L6-v2 as an embedding model and Mistral-7B LLM.\n
+  The Dataset We use here is ayoub_kirouane/arxiv-math
""")
with st.sidebar : 
    st.image("icon.png")
    temperature = st.sidebar.slider("Select your temperature value : " ,min_value=0.1 ,
                                 max_value=1.0 ,
                                   value=0.5)
    top_p = st.sidebar.slider("Select your top_p value : " ,min_value=0.1 ,
                           max_value=1.0 , 
                           value=0.5)
    k_n = st.number_input("Enter the number of top-ranked retriever Results:" ,
                             min_value=1 , max_value=5 , value=2)
    n_qs = st.number_input("Enter the number of arxiv-math Dataset samples you Want To use  :" ,
                             min_value=500 , max_value=50000 , value=1000 , step=500)

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
with st.spinner("Downloading the Mistral 7B  and all-MiniLM-L6-v2....") : 
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2
    )

local_llm = HuggingFacePipeline(pipeline=pipe)
def load_dataset() : 
    train_dataset = load_dataset("ayoubkirouane/arxiv-math"  , split="train")
    train_dataset = train_dataset.to_pandas()
    train_dataset['text'] = train_dataset["question"] +  train_dataset["answer"]
    df_document = DataFrameLoader(train_dataset[:1000] , page_content_column="text").load()
    text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=10)
    texts = text_splitter.split_documents(df_document)
    chromadb_index = Chroma.from_documents(texts, embed_model , persist_directory="DB")
    return chromadb_index

with st.spinner("Preparing Arxiv-math Dataset (Download + Embedding)...") : 
    chromadb_index = load_dataset()


document_qa = RetrievalQA.from_chain_type(
    llm=local_llm, chain_type="stuff", 
    retriever=chromadb_index.as_retriever(search_kwargs={"k": k_n})
)

def run_qa(prompt) : 
    return document_qa.run(prompt)


if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi :hand: Im Your open-source Mistral-7b_RAG-math chatbot, You can ask any thing about arxiv-math Dataset content"}]
# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_qa(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
