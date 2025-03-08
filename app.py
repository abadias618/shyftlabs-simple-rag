# server
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
import uvicorn
from dotenv import load_dotenv
# Get input from user
import shutil
import os
# Text Splitters
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embeddings
from langchain_openai import OpenAIEmbeddings
# Mock DB
from langchain_core.vectorstores import InMemoryVectorStore
app = FastAPI()
# Retrieval
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from prompt_enhance import KEYWORD_PROMPT, SEMANTIC_PROMPT
# Get back to the user
from langchain.callbacks.manager import AsyncCallbackManager
from fastapi.responses import StreamingResponse
import io
import asyncio
from TokenStream import TokenStreamHandler


UPLOAD_DIR = "./uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# load our key
load_dotenv()
# Our Mock DB
global_vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    print("File Name:", file)
    print("File.content_type:",file.content_type)
    if file.content_type not in ["text/html", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only HTML and PDF files are allowed.")
    
    # Since the our Mock DB isn't persistent we'd want to save the files to disk (jsut in case).
    destination_file_path = os.path.join(UPLOAD_DIR, file.filename)
    print("destination_file_path",destination_file_path)
    
    try:
        with open(destination_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()
    
    try:
        with open(destination_file_path, "rb") as f:
            file_content = f.read()
        print("read file from disk succesfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading file from disk.")
    print(f"{len(file_content)} file content length.")
    
    text = ""
    if file.content_type == "application/pdf":
        try:
            pdf_stream = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_stream)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error processing PDF file.")
    elif file.content_type == "text/html":
        try:
            soup = BeautifulSoup(file_content, "html.parser")
            text = soup.get_text(separator="\n")
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error processing HTML file.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in your file.")
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    docs = [Document(page_content=chunk) for chunk in chunks]
    # Add the generated documents to the global in‑memory vector store.
    docs_added = global_vector_store.add_documents(documents=docs)
    print(f"{len(docs_added)} documents added to InMemoryVectorStore.")

    return {"filename": file.filename, "message": "File uploaded and processed successfully."}
    

@app.post("/query/")
async def query_document(query: str = Form(...)):
    # Here is a naive approach to handling the keyword search (but will probs prompt engineer further down)
    num_tokens = len(query.split())
    k_retrieval = 5
    if num_tokens <= 3:
        k_retrieval = 8
    print("num_tokens",num_tokens)
    retriever = global_vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": k_retrieval}
    )

    # Configure a streaming callback handler.
    stream_handler = TokenStreamHandler()
    callback_manager = AsyncCallbackManager([stream_handler])

    chat = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True,
        callbacks=callback_manager,
    )
    
    # Enhancer 
    # TODO: if not to expensive generate brief summary of the doc
    # to provide as context for the question enhancer.
    enh = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
        streaming=False
    )
    
    enh_prompt = ""
    if num_tokens <= 3:
        enh_prompt = [("system", KEYWORD_PROMPT),("human", query)]
    else:
        enh_prompt = [("system", SEMANTIC_PROMPT),("human", query)]
    
    enhanced_prompt = enh.invoke(enh_prompt).content
    print(f"\nEnhanced prompt:\n{enhanced_prompt}")
    
    # Build the RetrievalQA chain, comes with a default prompt
    
    # "stuff" chain type method concatenates the retrieved documents.
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        verbose=True,
        chain_type_kwargs={"verbose": True}
    )
    
    answer_task = asyncio.create_task(qa_chain.ainvoke(enhanced_prompt))

    # The token generator yields tokens as they are produced.
    async def token_generator():
        while True:
            token = await stream_handler.queue.get()
            if token is None:
                break
            yield token
        await answer_task

    return StreamingResponse(token_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)