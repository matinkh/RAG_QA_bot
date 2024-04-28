import os
import time
import torch

from huggingface_hub import login
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline
)


CHUNK_SIZE = 250
CHUNK_OVERLAP = 25
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
TEMPERATURE=0.2
ACCEPTED_FILETYPES = {"pdf": PyPDFLoader}


class OnlineQaRag:
    """
    This RAG receives the context as the first step and initializes its vector store, before answering the user's queries.
    It uses an in-memory vector and needs to be initialized before every use.
    """
    def __init__(self, folder: str):
        self._login_to_huggingface()
        self._initialize_vector_db(folder)
        self._initialize_tokenizer()
        self._initialize_model()
        self._initialize_llm()
        self._initialize_prompt()

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
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Setup vector store
        self._db = FAISS.from_documents(chunked_documents, embedding_model)
        self.retriever = self._db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        
        toc = time.time()
        print(f"Vector DB initialized for {folder} in {(toc-tic):.2f}")
    
    def _initialize_model(self):
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
                #load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                          quantization_config=bnb_config)
        
    def _initialize_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer

    def _initialize_llm(self):
        text_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=TEMPERATURE,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )

        self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    def _initialize_prompt(self):
        self.prompt_template = """
        ### [INST] Instruction: Answer the question based on the context. Here is context to help:

        {context}

        ### QUESTION:
        {question} [/INST]
        """

    def invoke(self, question: str):
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.prompt_template,
        )

        # Create llm chain 
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        rag_chain = ( 
        {"context": self.retriever, "question": RunnablePassthrough()}
            | llm_chain
        )

        return rag_chain.invoke(question)