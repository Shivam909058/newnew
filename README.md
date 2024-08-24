# Document Upload and Chat Application

## Overview

This application is a Flask-based web service that allows users to upload documents (TXT, PDF, CSV) and chat with a bot that can answer questions based on the uploaded content. It uses OpenAI's language models and FAISS for efficient similarity search and retrieval.

## Features

- Document upload support for TXT, PDF, and CSV files
- Chat interface to interact with a bot
- Integration with OpenAI's language models for natural language processing
- FAISS-based vector storage for efficient document retrieval
- SQLite database for storing employee and HR policy information
- Profanity filter to maintain appropriate conversation

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. Create a `bad.txt` file with a list of words to be filtered (one word per line).

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Use the file upload form to upload documents (TXT, PDF, or CSV).

4. Use the chat interface to ask questions about the uploaded documents or inquire about employee information and HR policies.

## Project Structure

- `app.py`: Main Flask application file
- `templates/upload.html`: HTML template for the web interface
- `uploads/`: Directory where uploaded files are stored
- `company_data.db`: SQLite database file for storing employee and HR policy information
- `bad.txt`: List of words to be filtered from chat messages

## Key Components

1. **Document Processing**: 
   - Uses `langchain` library to load and process different file types.
   - Splits documents into smaller chunks for efficient processing.

2. **Vector Storage**: 
   - Uses FAISS to create a vector store of document chunks for fast similarity search.

3. **Chat Interface**: 
   - Provides a simple web interface for users to interact with the chatbot.

4. **Database Integration**: 
   - Uses SQLite to store and retrieve employee and HR policy information.

5. **Language Model Integration**: 
   - Utilizes OpenAI's language models through the `langchain` library for natural language understanding and generation.

## Troubleshooting

- If you encounter any issues with document processing, check the application logs for detailed error messages.
- Ensure that your OpenAI API key is correctly set in the `.env` file.
- Verify that the `uploads` directory exists and has the correct permissions.
