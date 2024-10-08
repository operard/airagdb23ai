# import databutton as db
import os
import sys
import random
from pathlib import Path
import streamlit as st

import textwrap as tr
# from streamlit_utils import parse_txt, text_to_docs, pdf_read, create_text_chunks
from text_load_utils import parse_txt, text_to_docs, parse_pdf, load_default_pdf, create_text_chunks
from df_chat import user_message, bot_message

# from db23ai_utils import checkDatabaseAccess, getMaxID, insertRow, storeChunks, rag_search
import oracledb
import array


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import oracleai
from langchain_community.utilities.oracleai import OracleSummary
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.document_loaders.oracleai import OracleTextSplitter 
from langchain_community.document_loaders.oracleai import OracleDocLoader 


from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms.llamacpp import LlamaCpp



# ======== General 


# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Oracle DB23AI
user = os.environ['username']
pwd = os.environ['password']
dsn = os.environ['service']
modelPath = os.environ['model']

# Table name
table_name = "AIRAGINBOX"


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5" # Uses 


embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)


# ======== Functions

# Store Documents to DB23AI
def storeInDB (lc_docs, embedding):
    # Store In DB
    if len(lc_docs) > 0:
    
        # Oracle Vector Databaseに接続
        with oracledb.connect(user = user, password = pwd, dsn = dsn) as connection:
            # テーブル「ovs」にEmbedding結果を保存
            ovs = OracleVS.from_documents(
                lc_docs,
                embedding,
                client=connection,
                table_name=table_name,
                distance_strategy=DistanceStrategy.COSINE,
            )
        
            # Store Indexes
            oraclevs.create_index(connection, ovs, params={"idx_name": "ivf_idx1", "idx_type": "IVF"})
 

def executeRAG (embedding, model_path, question):

    # 
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        max_tokens=4096
        #n_ctx=512,
        #max_tokens=1024
    )
    print ('--> Search in Database')
    # Oracle Vector Databaseに接続
    with oracledb.connect(user = user, password = pwd, dsn = dsn) as connection:
        ovs = OracleVS(client=connection, embedding_function=embedding, 
                       table_name=table_name, 
                       distance_strategy=DistanceStrategy.COSINE)

        # Prompt
        template = """Lo siguiente es el uso del contexto y la pregunta final es la respuesta concisa. La respuesta es la ocasión, la respuesta es la ocasión.。
            {context}
            Preguntar: {question}
            Respuesta :"""

        #prompt_template = '''[INST] <<SYS>>
        #You are a friendly chatbot who always responds in the style of a pirate
        #<</SYS>>
        #{prompt}[/INST]
        #'''

        # 
        question_prompt = PromptTemplate(
            template=template, 
            input_variables=["question"] 
        )

        # RAG
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ovs.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": question_prompt}
        )

        print ('--> Invoke LLM')
        # 
        # question = "Resuma los puntos clave del PROYECTO LIMPIEZA DE MATORRAL desde una perspectiva de gestión."
        #question = origquestion

        # Call LLM
        response = qa.invoke(question)

        # Analyze Response
        # print("Pregunta: "+response['query'])
        # print("Respuesta: "+response['result'])
        return response['result']



# ======== Class 

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer
    

# ================ Streamlit Process ======================
def main(): 

    st.set_page_config("AI RAG in A BOX Demo Chat Bot with Db23ai 🤖", layout="centered")
    
    
    st.title("AI RAGDEMO IN A BOX with DB23ai and LlamaCpp")
    st.sidebar.title("Parameters")
    
    llm_model = st.sidebar.radio("Select one of the models--", options=["llama3.1", "llama3", "llama2", "mistral"], index=(0))
    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
       
    
    
    rag_selectbox = st.sidebar.radio("Select one of RAG Technique--", options=["No RAG", "Naive RAG", "Adaptive RAG", "Modular RAG"], index=(0))
    # Set assistant_type in session state
    if "rag" not in st.session_state:
        st.session_state["rag"] = rag_selectbox
    # Restart the assistant if assistant_type has changed
    elif st.session_state["rag"] != rag_selectbox:
        st.session_state["rag"] = rag_selectbox

    
    
    max_tokens = st.sidebar.slider('Pick max tokens', min_value=1, max_value=1024, value=120, step=1)
    temperature = st.sidebar.slider('Pick a temperature', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    prompt_token_input = st.sidebar.text_area(
            "Enter your Instruction", placeholder="Type your instruction here", height= 250, value = "You are a chat bot . You need to answer questions based on the text provided above . If the text doesn't contain the answer, reply 'I am here to assist with the Document uploaded. For general knowledge questions, you might consider using a search engine or another reference source. How can I assist you with questions related to the document?' "
        )
    
    hide_streamlit_style = """
                <style>
                [data-testid="stToolbar"] {visibility: hidden !important;}
                footer {visibility: hidden !important;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # temparature_slider = st.sidebar.slider("Pick a temperature", 0.1, 2.0)
    # st.markdown("%OP% Test")
    # temperature = st.sidebar.slider('Pick a temperature', 0.0, 1.0)
    
    # with st.sidebar:
    #     modelName = st.radio(
    #         "Choose a shipping method",
    #         ("command", "command-nightly")
    #     )
    
    # modelName = st.radio("--", options=["command", "command-nightly"])
    # st.info(
    #     "For your personal data! Powered by [cohere](https://cohere.com) + [LangChain](https://python.langchain.com/en/latest/index.html)"
    # )
    
    # opt = st.radio("--", options=["Try the demo!", "Upload-own-file"])
    opt = "Upload-own-file"
    # Unnecessary complication! -> Refractor required
    pages = None
    lc_docs = []
    if opt == "Upload-own-file":
        # Upload the files
        # uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
        uploaded_files = st.file_uploader(
            "**Upload a Pdf or txt file :**",
            type=["pdf", "txt"], accept_multiple_files=True
        )
        if uploaded_files:
            pages = []
            for uploaded_file in uploaded_files:
                print("--> Analyze file to store in DB23ai : "+uploaded_file.name)
                if uploaded_file:
                    if uploaded_file.name.endswith(".txt"):
                        doc = parse_txt(uploaded_file)
                    else:
                        doc = parse_pdf(uploaded_file)
                    pages2 = text_to_docs(doc)
                    for p in pages2:
                        pages.append(p)
                    # Add
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    docs = text_splitter.split_documents(pages2)
                    for doc in docs:
                        lc_docs.append(Document(page_content=doc.page_content.replace("\n", ""), metadata={'source': uploaded_file.name}))
                    
   

        #    pages_processed = processFile(uploaded_file)
        #    if pages is None:
        #        pages = pages_processed
        #    else:
        #        for p in pages_processed:
        #            pages.append(p)
        
        #uploaded_file = st.file_uploader(
        #        "**Upload a Pdf or txt file :**",
        #        type=["pdf", "txt"]
        #)
        #if uploaded_file:
        #    if uploaded_file.name.endswith(".txt"):
        #        doc = parse_txt(uploaded_file)
        #    else:
        #        doc = parse_pdf(uploaded_file)
        #    pages = text_to_docs(doc)

    else:
        st.markdown(
            "Demo PDF : Created By %OP% "
        )
        st.text("Quick Prompts to try (English | German | Spanish):")
        # pages = load_default_pdf()
    
    
    page_holder = st.empty()
    # Create our own prompt template
    prompt_template = """Text: {context}
    
    Question: {question}
    
    Answer the question based on the text provided. If asked for summarization then look to the whole document and summarize it """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    print("%OP% the final Prompt is --------------------->")
    print(PROMPT)
    chain_type_kwargs = {"prompt": PROMPT}
    
    
    # Bot UI dump
    # Session State Initiation
    prompt = st.session_state.get("prompt", None)
    
    if prompt is None:
        prompt = [{"role": "system", "content": prompt_template}]
    
    # If we have a message history, let's display it
    for message in prompt:
        if message["role"] == "user":
            user_message(message["content"])
        elif message["role"] == "assistant":
            bot_message(message["content"], bot_name="AI Bot")
    
    if pages:
        # if uploaded_file.name.endswith(".txt"):
    
        # else:
        #     doc = parse_pdf(uploaded_file)
        # pages = text_to_docs(doc)
        
    
        with page_holder.expander("File Content", expanded=False):
            pages
       
        
        # Read from Database 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)
    
        # DB
        # Vector Database: DB23ai
        storeInDB (lc_docs, embedding_model)
        
        
        
        messages_container = st.container()
        question = st.text_input(
            "", placeholder="Type your message here", label_visibility="collapsed"
        )
        originalQuestion = question
        # question = question + prompt_token_input
    
        if st.button("Run", type="secondary"):
            prompt.append({"role": "user", "content": originalQuestion})
            print("%OP% the main prompt is --->")
            print(PROMPT)
            with messages_container:
                user_message(originalQuestion)
                botmsg = bot_message("...", bot_name="Chat Bot")
                print(f"Model selected --> {llm_model}")
            
            # ############################
            # Search in DB23ai
            # ############################
            
            # concatenated_content = rag_search (originalQuestion, 5)

            # retriever = store.as_retriever(search_type="mmr")
            # retriever = store.as_retriever(search_kwargs={"k": 5})
            answer = executeRAG (embedding_model, modelPath, originalQuestion)
    
    
            # ############################
            # Analyze RAG
            # "No RAG", "Naive RAG", "Adaptive RAG", "Modular RAG"
            # ############################
            #modelname = st.session_state["llm_model"]
            #rag_type = st.session_state["rag"]
            
            print("%OP% ANSWER is --->")
            print(answer)
            print("%OP% ANSWER is ---> %OP% END")
    
    
            #result = answer.replace("\n", "").replace("Answer:", "")
            result = answer
         
            with st.spinner("Loading response .."):
                botmsg.update(result)
            # Add
            prompt.append({"role": "assistant", "content": result})
    
        st.session_state["prompt"] = prompt
    else:
        st.session_state["prompt"] = None
        st.warning("No file found. Upload a file to chat!")
    
    # Written at


# Function to check access to DB23ai

if __name__ == '__main__': 
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

    
