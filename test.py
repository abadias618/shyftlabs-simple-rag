import os
import requests

def test_upload(upload_url, file_path):
    """
    Upload a file to the /upload/ endpoint.
    The file can be either PDF or HTML.
    """
    # Set the MIME type based on the file extension.
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        mime_type = "application/pdf"
    elif ext in [".htm", ".html"]:
        mime_type = "text/html"
    else:
        raise ValueError("Unsupported file type. Use PDF or HTML.")
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, mime_type)}
        print(f"Uploading {file_path} to {upload_url} ...")
        response = requests.post(upload_url, files=files)

    print("Upload response status code:", response.status_code)
    try:
        print("Upload response JSON:", response.json())
    except Exception as e:
        print("Error decoding JSON response:", e)
        
def test_query(query_url, query_text):
    """
    Query the stored documents via the /query/ endpoint.
    This function prints the streamed output token-by-token.
    """
    data = {"query": query_text}
    print(f"\nSending query to {query_url} ...\nWith Query: {query_text}")
    response = requests.post(query_url, data=data, stream=True)
    print("Query response status code:", response.status_code)
    print("Streamed response:")
    for chunk in response.iter_content(chunk_size=128, decode_unicode=True):
        if chunk:
            print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    # Base URL for the FastAPI local server.
    BASE_URL = "http://127.0.0.1:8000"
    UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
    QUERY_ENDPOINT = f"{BASE_URL}/query"
    TEST_FILE_PATH = "./deepseek_v3_paper.pdf"
    QUERY_TEXT = "What is done in Post-training?"

    # 1. Upload the file. TEST 1
    test_upload(UPLOAD_ENDPOINT, TEST_FILE_PATH)
    print("upload seems to be working")
    # 2. Query the stored documents (the streaming output will be printed). TEST 2
    test_query(QUERY_ENDPOINT, QUERY_TEXT)
    print("query seems to be working")