import os
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import logging
import io

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = FastAPI()

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv'}

# Database setup
DATABASE_URL = "sqlite:///./chatbot.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(LargeBinary)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_documents(file_content: bytes, filename: str) -> List[str]:
    try:
        file_extension = filename.rsplit('.', 1)[1].lower()
        if file_extension == 'pdf':
            loader = PyPDFLoader(io.BytesIO(file_content))
        elif file_extension == 'txt':
            loader = TextLoader(io.StringIO(file_content.decode('utf-8')))
        elif file_extension == 'csv':
            loader = CSVLoader(io.StringIO(file_content.decode('utf-8')))
        else:
            raise ValueError(f"Unsupported file type: {filename}")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        logging.error(f"Error loading file {filename}: {str(e)}")
        raise

# Load bad words from file
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
async def upload_file(file: UploadFile = File(...), db: SessionLocal = Depends(get_db)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    file_content = await file.read()
    
    try:
        texts = load_documents(file_content, file.filename)
        docsearch.add_documents(texts)
        
        db_file = File(filename=file.filename, content=file_content)
        db.add(db_file)
        db.commit()
        
        return JSONResponse(content={"message": f"File {file.filename} has been processed and added to the knowledge base."})
    except Exception as e:
        db.rollback()
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)