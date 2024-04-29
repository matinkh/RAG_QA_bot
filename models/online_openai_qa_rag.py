"""
This is an online QA RAG, meaning data and query processing happen at the same time.
Config:
    - Vector DB: FAISS
    - Embedding Model: Mistral-7B-Instruct-v0.1
    - LLM: OpenAI
"""
import os
import time

from huggingface_hub import login
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


CHUNK_SIZE = 250
CHUNK_OVERLAP = 25
LLM_MODEL_NAME = "gpt-3.5-turbo"
ACCEPTED_FILETYPES = {"pdf": PyPDFLoader}


class OnlineOpenAiQaRag:
    def __init__(self, folder: str):
        self._login_to_huggingface()
        self._initialize_vector_db(folder)
        self._initialize_agent_chain()


    def _login_to_huggingface(self):
        login(token=os.environ["HF_TOKEN"])

    def _initialize_vector_db(self, folder: str):
        tic = time.time()
        # Read files into "documents". Each "document" has a page content and metadata.
        documents = []
        for file in os.listdir(folder):
            file_extension = file.split(".")[-1]
            if ACCEPTED_FILETYPES.get(file_extension, None) is None:
                print(f"Skipping {file}. This file extension not accepted {file_extension}")
                continue

            doc_loader = ACCEPTED_FILETYPES[file_extension]
            file_path = os.path.join(folder, file)
            loader = doc_loader(file_path)
            documents.extend(loader.load())
        
        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunked_documents = text_splitter.split_documents(documents)

        # Setup the embedding model.
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Setup vector store
        self.db = FAISS.from_documents(chunked_documents, embedding_model)

        toc = time.time()
        print(f"Vector DB initialized for {folder} in {(toc-tic):.2f} seconds.")

    def _initialize_agent_chain(self):
        llm = ChatOpenAI(model_name=LLM_MODEL_NAME)
        self.agent_chain = load_qa_chain(llm, chain_type="stuff")

    
    def invoke(self, question: str):
        context_docs = self.db.similarity_search(question, k=10)
        answer = self.agent_chain.run(input_documents=context_docs, question=question)
        return answer