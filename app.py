import os
<<<<<<< HEAD
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
from typing import List, Optional
=======
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
>>>>>>> 83440cad0cf9f4fb7bb632c7357083b78f358e03
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

<<<<<<< HEAD
app = FastAPI()
=======
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
>>>>>>> 83440cad0cf9f4fb7bb632c7357083b78f358e03

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv'}
UPLOAD_FOLDER = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_documents(file_path: str) -> List[str]:
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        raise

<<<<<<< HEAD
# Load bad words from file
=======
>>>>>>> 83440cad0cf9f4fb7bb632c7357083b78f358e03
with open("bad.txt", "r") as f:
    bad_words = set(word.strip().lower() for word in f)

def filter_bad_words(text: str) -> str:
    words = text.split()
    filtered_words = [word if word.lower() not in bad_words else "[FILTERED]" for word in words]
    return " ".join(filtered_words)

class DatabaseConfig(BaseModel):
    url: str

class ChatQuery(BaseModel):
    query: str
    db_url: Optional[str] = None

db_engines = {}

def get_db_session(db_url: str):
    if db_url not in db_engines:
        engine = create_engine(db_url)
        db_engines[db_url] = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return db_engines[db_url]()

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(["Initial document"], embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-16k")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        texts = load_documents(file_path)
        docsearch.add_documents(texts)
        return JSONResponse(content={"message": f"File {file.filename} has been processed and added to the knowledge base."})
    except Exception as e:
        logging.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/chat")
async def chat(chat_query: ChatQuery):
    filtered_query = filter_bad_words(chat_query.query)

    if filtered_query != chat_query.query:
        return JSONResponse(content={"response": "Your message contained inappropriate language and has been filtered."})

    try:
        if chat_query.db_url:
            db_session = get_db_session(chat_query.db_url)
            try:
                result = db_session.execute(text(filtered_query))
                db_results = result.fetchall()
                formatted_results = "\n".join([str(row) for row in db_results])
                combined_query = f"{filtered_query}\n\nRelevant database information:\n{formatted_results}"
                response = qa({"query": combined_query})
            except SQLAlchemyError as e:
                logging.error(f"Database error: {str(e)}")
                raise HTTPException(status_code=500, detail="Database query error")
            finally:
                db_session.close()
        else:
            response = qa({"query": filtered_query})

        filtered_response = filter_bad_words(response['result'])
        logging.debug(f"Chat response: {filtered_response}")
        return JSONResponse(content={"response": filtered_response})
    except Exception as e:
        logging.error(f"Error in chat route: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

if __name__ == '__main__':
<<<<<<< HEAD
    import uvicorn
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
=======
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
>>>>>>> 83440cad0cf9f4fb7bb632c7357083b78f358e03
