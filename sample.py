import requests
import os

# Test upload endpoint
def test_upload():
    url = "http://127.0.0.1:8000/upload"
    # Get the full path to the file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'sample.txt')
    
    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        files = {'file': ('sample.txt', file, 'text/plain')}
        response = requests.post(url, files=files)
    
    print(response.json())

# Test chat endpoint
def test_chat():
    url = "http://127.0.0.1:8000/chat"
    payload = {
        "query": "What was the financial performance of GAIL (India) Limited for the fiscal year?"
    }
    response = requests.post(url, json=payload)
    print(response.json())

if __name__ == "__main__":
    test_upload()
    test_chat()