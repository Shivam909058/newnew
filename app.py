import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import sqlite3
from typing import List, Tuple
import logging


logging.basicConfig(level=logging.DEBUG)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv'}

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

with open("bad.txt", "r") as f:
    bad_words = set(word.strip().lower() for word in f)

def filter_bad_words(text: str) -> str:
    words = text.split()
    filtered_words = [word if word.lower() not in bad_words else "[FILTERED]" for word in words]
    return " ".join(filtered_words)


def setup_database():
    conn = sqlite3.connect('company_data.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            position TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hr_policies (
            id INTEGER PRIMARY KEY,
            policy_name TEXT,
            description TEXT
        )
    ''')

    conn.commit()
    return conn


def query_database(query: str) -> List[Tuple]:
    conn = setup_database()
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results

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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                texts = load_documents(file_path)
                docsearch.add_documents(texts)
                flash(f'File {filename} has been processed and added to the knowledge base.')
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")
                flash(f'Error processing file: {str(e)}')

            return redirect(url_for('upload_file'))
    return render_template('upload.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query', '')
    logging.debug(f"Received chat query: {user_query}")

    filtered_query = filter_bad_words(user_query)

    if filtered_query != user_query:
        return jsonify({"response": "Your message contained inappropriate language and has been filtered."}), 200

    try:
        if "employee" in filtered_query.lower() or "hr policy" in filtered_query.lower():
            if "employee" in filtered_query.lower():
                db_query = "SELECT * FROM employees"
            else:
                db_query = "SELECT * FROM hr_policies"

            db_results = query_database(db_query)
            formatted_results = "\n".join([str(row) for row in db_results])
            combined_query = f"{filtered_query}\n\nRelevant database information:\n{formatted_results}"
            response = qa({"query": combined_query})
        else:
            response = qa({"query": filtered_query})

        filtered_response = filter_bad_words(response['result'])
        logging.debug(f"Chat response: {filtered_response}")
        return jsonify({"response": filtered_response}), 200
    except Exception as e:
        logging.error(f"Error in chat route: {str(e)}")
        return jsonify({"response": "I'm sorry, but I encountered an error while processing your request."}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
