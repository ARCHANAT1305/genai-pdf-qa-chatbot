## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
Given a PDF document containing structured text (e.g., research papers, lecture notes, or reports), develop a system that allows a user to ask natural language questions and receive accurate answers based on the content of the PDF. The system should automatically process the document, split it into manageable chunks, create embeddings for semantic search, and use a language model to generate context-aware answers

### DESIGN STEPS:

 STEP1 : Load API key for OpenAI.

 STEP2 : Load PDF using PyPDFLoader.

 STEP 3:  Extract text from all pages.

 STEP 4:  Split text into manageable chunks with overlap.

 STEP 5:  Generate embeddings for each chunk.

 STEP 6:  Store embeddings in a vector database (DocArrayInMemorySearch).

 STEP 7: Build Retrieval QA chain using LLM and retriever.

 STEP 8: Input a question.

 STEP 9: Retrieve relevant chunks from the PDF.

 STEP10: Generate and return answer using the LLM.

### PROGRAM:
```
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def build_pdf_qa(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    db = DocArrayInMemorySearch.from_documents(chunks, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=db.as_retriever()
    )

qa = build_pdf_qa("docs/cs229_lectures/attention.pdf")
query = "What problem does the Transformer architecture aim to solve?"
answer = qa.run(query)
print("Q:", query)
print("A:", answer)

loader = PyPDFLoader("docs/cs229_lectures/attention.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages from the PDF.")
```

### OUTPUT:
<img width="1249" height="150" alt="image" src="https://github.com/user-attachments/assets/1e547ed2-a833-4370-817f-ac6f1ad22330" />

### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain is executed successfully.
