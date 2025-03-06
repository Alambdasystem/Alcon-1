from flask import Flask, request, jsonify, render_template, session
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from azure.storage.blob import BlobServiceClient
import fitz  # PyMuPDF
import os
import sqlite3
import logging
import uuid
from datetime import datetime
import nltk
from openai import AzureOpenAI
import re
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import tempfile

# Configure logging with INFO level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download required NLTK resources.
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Initialize the Flask app and secret key
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a secure secret key

# ---------------------
# Global Configuration Variables
# ---------------------
OPENAI_API_KEY = "2yEKfg6KvcBXgL0YV1fkHYweHoo18ZV2QumJP3HtRiT4eItEQmsrJQQJ99AKACHrzpqXJ3w3AAABACOG3dpB"
DEPLOYMENT_NAME = "gpt-4"
OPENAI_API_ENDPOINT = ("https://alconaiservice.openai.azure.com/"
                         "openai/deployments/gpt-4/chat/completions?api-version=2024-02-15-preview")

azure_blob_connection_string = ("DefaultEndpointsProtocol=https;"
                                  "AccountName=formulationscondensed;"
                                  "AccountKey=wJh4mvSgbEOoOFfcDsLVX7GSIzw59gkJF0Do77rOmOv60EFCODcNvNTaHVusWcVrvnLtkK3x4wX5+AStWjV9Ew=="
                                  "EndpointSuffix=core.windows.net")
azure_blob_container_name = "input"
blob_service_client = BlobServiceClient.from_connection_string(azure_blob_connection_string)
container_client = blob_service_client.get_container_client(azure_blob_container_name)

# Form Recognizer configuration (replace with your actual key if available)
form_recognizer_endpoint = "https://documentanalysisclient.cognitiveservices.azure.com/"
form_recognizer_key = "YOUR_FORM_RECOGNIZER_KEY"  # Replace with your actual Form Recognizer key

# Local directory to store downloaded PDFs
PDF_DOWNLOAD_PATH = "downloaded_pdfs"
os.makedirs(PDF_DOWNLOAD_PATH, exist_ok=True)

# ---------------------
# Database Initialization Functions
# ---------------------
def init_db():
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                feedback INTEGER
            )
        ''')
        conn.commit()

def init_user_db():
    with sqlite3.connect("user_data.db") as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

def init_pdf_cache_db():
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS pdf_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_name TEXT NOT NULL,
                content TEXT NOT NULL
            )
        ''')
        conn.commit()

# ---------------------
# PDF Processing Logic
# ---------------------
def download_blob(blob_name):
    """
    Downloads a PDF from Azure Blob Storage and saves it locally.
    """
    try:
        local_path = os.path.join(PDF_DOWNLOAD_PATH, blob_name)
        blob_client = container_client.get_blob_client(blob_name)
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        logging.info(f"Downloaded and saved PDF to: {local_path}")
        return local_path
    except Exception as e:
        logging.error(f"Error downloading blob {blob_name}: {e}")
        return None

def extract_text_from_pdf_with_recognition(pdf_bytes):
    """
    Saves PDF bytes to a temporary file, sends the file to Azure Form Recognizer 
    (using the prebuilt-layout model) for text extraction, and removes the temporary file.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_filename = tmp_file.name
        logging.info(f"Temporary PDF saved to: {tmp_filename}")
        
        document_analysis_client = DocumentAnalysisClient(
            form_recognizer_endpoint, AzureKeyCredential(form_recognizer_key)
        )
        
        with open(tmp_filename, "rb") as f:
            poller = document_analysis_client.begin_analyze_document("prebuilt-layout", f)
            result = poller.result()
        
        extracted_text = ""
        for page in result.pages:
            for line in page.lines:
                extracted_text += line.content + "\n"
        
        os.remove(tmp_filename)
        logging.info("Temporary PDF file removed.")
        return extracted_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF using Form Recognizer: {e}")
        return ""

def clean_extracted_text(text):
    return re.sub(r'\n+', '\n', text).strip()

def process_single_pdf(pdf_name):
    """
    Downloads a PDF from Azure Blob Storage, extracts text using the updated logic,
    cleans the extracted text, and returns a tuple (pdf_name, cleaned_text).
    """
    try:
        logging.info(f"Processing PDF: {pdf_name}")
        pdf_path = download_blob(pdf_name)
        if not pdf_path:
            logging.error(f"Failed to download PDF: {pdf_name}")
            return None
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        extracted_text = extract_text_from_pdf_with_recognition(pdf_bytes)
        cleaned_text = clean_extracted_text(extracted_text)
        logging.info(f"Extracted text from {pdf_name}: {cleaned_text[:100]}...")
        return (pdf_name, cleaned_text)
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_name}: {e}")
        return None

def batch_insert_pdfs(records):
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.executemany('''
            INSERT INTO pdf_texts (pdf_name, content)
            VALUES (?, ?)
        ''', records)
        conn.commit()
        logging.info(f"Inserted {len(records)} PDFs into the database")

def list_blobs():
    try:
        return [blob.name for blob in container_client.list_blobs()]
    except Exception as e:
        logging.error(f"Error listing blobs: {e}")
        return []

def preprocess_pdfs_to_db(limit=100):
    pdf_list = list_blobs()[:limit]
    logging.info(f"Processing {len(pdf_list)} PDFs")
    batch = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
        for result in executor.map(process_single_pdf, pdf_list):
            if result:
                batch.append(result)
                if len(batch) >= 100:
                    batch_insert_pdfs(batch)
                    batch = []
    if batch:
        batch_insert_pdfs(batch)

# ---------------------
# Chat, Summaries, and Utility Functions
# ---------------------
def save_message_to_db(username, role, content, conversation_id):
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (conversation_id, username, role, content, timestamp, feedback)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (conversation_id, username, role, content, datetime.now(), False))
        conn.commit()

def load_chat_history():
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, conversation_id, username, role, content, timestamp
            FROM chat_history
            ORDER BY timestamp
        ''')
        return [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5]}
            for row in c.fetchall()
        ]

def query_openai(user_message, relevant_paragraphs, conversation_id):
    try:
        client = AzureOpenAI(api_key=OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=OPENAI_API_ENDPOINT)
        with sqlite3.connect("chat_history.db") as conn:
            c = conn.cursor()
            c.execute('''
                SELECT role, content
                FROM chat_history
                WHERE conversation_id = ?
                ORDER BY timestamp
            ''', (conversation_id,))
            history = [{"role": role, "content": content} for role, content in c.fetchall()]
        system_msg = {
            "role": "system",
            "content": "You are a helpful AI assistant specializing in lens formulations, chemistry, optics, and AI-driven research analysis. Focus on the most relevant details."
        }
        messages = [system_msg] + history[-5:]
        relevant_text = "\n\n".join([
            f"{p['paragraph']} (Source: {p['source']}, PDF: {p['pdf_name']})" for p in relevant_paragraphs
        ])
        messages.append({
            "role": "user",
            "content": user_message + "\n\nRelevant paragraphs:\n" + relevant_text
        })
        logging.info(f"Sending {len(messages)} messages to OpenAI for query.")
        response = client.chat.completions.create(model="gpt-4", messages=messages)
        logging.info("Received response from OpenAI.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error querying OpenAI: {e}")
        return "Sorry, I encountered an error while processing your request."

def is_relevant(paragraph, query):
    query_words = query.lower().split()
    return any(word in paragraph.lower() for word in query_words)

def parse_pdf_metadata(pdf_name):
    base_name = pdf_name.rsplit('.', 1)[0]
    parts = base_name.split('_')
    if len(parts) >= 4:
        return {"author": parts[0], "year": parts[1], "title": parts[2], "publisher": parts[3]}
    return {"author": "Unknown", "year": "n.d.", "title": base_name, "publisher": "Unknown"}

def search_pdfs_helper(user_message, max_paragraphs=5):
    relevant_paragraphs = []
    try:
        with sqlite3.connect("pdf_cache.db") as conn:
            c = conn.cursor()
            c.execute("SELECT pdf_name, content FROM pdf_texts")
            pdfs = [{"name": row[0], "content": row[1]} for row in c.fetchall()]
    except Exception as e:
        logging.error(f"Error retrieving PDFs from cache: {e}")
        return relevant_paragraphs
    for pdf in pdfs:
        metadata = parse_pdf_metadata(pdf["name"])
        apa_citation = f"{metadata['author']} ({metadata['year']}). {metadata['title']}. {metadata['publisher']}."
        paragraphs = [p.strip() for p in pdf["content"].split("\n") if p.strip()]
        for paragraph in paragraphs:
            if is_relevant(paragraph, user_message):
                relevant_paragraphs.append({
                    "paragraph": paragraph,
                    "source": apa_citation,
                    "pdf_name": pdf["name"]
                })
    return relevant_paragraphs[:max_paragraphs]

def update_context_summary(conversation_id, max_history_messages=20):
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT role, content
            FROM chat_history
            WHERE conversation_id = ? AND role != 'summary'
            ORDER BY timestamp
        ''', (conversation_id,))
        all_messages = c.fetchall()
    if len(all_messages) <= max_history_messages:
        logging.info("No need to update summary; conversation history is within limit.")
        return
    conversation_text = "\n".join([f"{role}: {content}" for role, content in all_messages])
    summary_prompt = (
        "The following is a conversation history:\n\n"
        f"{conversation_text}\n\n"
        "Please provide a concise summary that captures the key points of the conversation."
    )
    logging.info("Requesting summary from OpenAI for context compression.")
    summary_text = call_openai_summary(summary_prompt).strip()
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (conversation_id, username, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, "system", "summary", summary_text, datetime.now()))
        conn.commit()
    logging.info("Context summary updated for conversation_id: " + conversation_id)

def call_openai_summary(prompt):
    try:
        client = AzureOpenAI(api_key=OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=OPENAI_API_ENDPOINT)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a summarization engine."},
                {"role": "user", "content": prompt}
            ]
        )
        logging.info("Summary received from OpenAI.")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in summarization call: {e}")
        return "Summary unavailable."

def decide_workflow(user_message):
    try:
        decision_prompt = (
            "You are an expert in lens formulations and patents. Based on the user's question, decide which workflow to use:\n"
            "1. Respond with 'patent' if the question is specifically asking for patent information.\n"
            "2. Respond with 'pdf' if the question requires searching through technical PDF content.\n"
            "3. Respond with 'both' if the question seems to need both patent info and PDF content.\n\n"
            f"User question: \"{user_message}\"\n\n"
            "Please reply with just one word: 'patent', 'pdf', or 'both'."
        )
        client = AzureOpenAI(api_key=OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=OPENAI_API_ENDPOINT)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a workflow decision assistant."},
                {"role": "user", "content": decision_prompt}
            ]
        )
        decision = response.choices[0].message.content.strip().lower()
        logging.info(f"Workflow decision from OpenAI: {decision}")
        if decision not in ["patent", "pdf", "both"]:
            logging.info("Decision not recognized. Defaulting to 'both'.")
            decision = "both"
        return decision
    except Exception as e:
        logging.error(f"Error in decide_workflow: {e}")
        return "both"

def get_patent_info(user_message):
    try:
        patent_prompt = (
            "You are a patent research assistant specialized in lens formulations. "
            "Provide detailed patent information based on the following query:\n"
            f"{user_message}\n\n"
            "Include relevant patent numbers, filing dates, and a brief summary if available."
        )
        client = AzureOpenAI(api_key=OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=OPENAI_API_ENDPOINT)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a patent research assistant."},
                {"role": "user", "content": patent_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in get_patent_info: {e}")
        return "Patent info unavailable due to an error."

# ---------------------
# API Endpoints
# ---------------------
@app.route("/")
def index():
    logging.info("Rendering index page.")
    return render_template("index.html")

@app.route("/list_chats", methods=["GET"])
def list_chats():
    logging.info("Listing chat messages.")
    username = request.args.get("username")
    query = '''
        SELECT id, conversation_id, username, role, content, timestamp
        FROM chat_history
        ORDER BY timestamp
    '''
    params = ()
    if username:
        query = '''
            SELECT id, conversation_id, username, role, content, timestamp
            FROM chat_history
            WHERE username = ?
            ORDER BY timestamp
        '''
        params = (username,)
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute(query, params)
        chat_history = [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5]}
            for row in c.fetchall()
        ]
    return jsonify(chat_history), 200

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    logging.info("Retrieving full chat history.")
    chat_history = load_chat_history()
    return jsonify(chat_history), 200

@app.route("/loadChat", methods=["GET"])
def load_chat():
    logging.info("Loading chat history for a given chat_id.")
    chat_id = request.args.get("chat_id")
    if not chat_id:
        logging.error("No chat_id provided.")
        return jsonify({"error": "No chat ID provided"}), 400
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, conversation_id, username, role, content, timestamp
            FROM chat_history
            WHERE id = ?
            ORDER BY timestamp
        ''', (chat_id,))
        chat_history = [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5]}
            for row in c.fetchall()
        ]
    return jsonify(chat_history), 200

@app.route("/search_pdfs", methods=["GET"])
def search_pdfs():
    logging.info("Searching PDFs.")
    search_term = request.args.get("search_term")
    if not search_term:
        logging.error("No search_term provided.")
        return jsonify({"error": "No search term provided"}), 400
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT pdf_name, content
            FROM pdf_texts
            WHERE content LIKE ?
            LIMIT 1
        ''', (f"%{search_term}%",))
        row = c.fetchone()
    return jsonify(row), 200

@app.route("/chat", methods=["POST"])
def chat():
    logging.info("Received chat request.")
    data = request.json
    user_message = data.get("message", "").strip()
    username = session.get("username")
    conversation_id = data.get("conversation_id", "").strip() or str(uuid.uuid4())

    if not user_message or not username:
        logging.error("Missing username or message.")
        return jsonify({"error": "Username and message are required"}), 400

    workflow_decision = decide_workflow(user_message)
    logging.info(f"Workflow decision: {workflow_decision}")
    combined_response = ""

    if workflow_decision in ["patent", "both"]:
        patent_info = get_patent_info(user_message)
        combined_response += "Patent Information:\n" + patent_info + "\n\n"

    if workflow_decision in ["pdf", "both"]:
        relevant_paragraphs = search_pdfs_helper(user_message, max_paragraphs=5)
        save_message_to_db(username, "user", user_message, conversation_id)
        update_context_summary(conversation_id, max_history_messages=20)
        pdf_response = query_openai(user_message, relevant_paragraphs, conversation_id)
        if relevant_paragraphs:
            apa_references = "\n\nReferences (APA format):\n" + "\n".join([
                f"{p['source']} (PDF: {p['pdf_name']}) - Excerpt: \"{p['paragraph']}\""
                for p in relevant_paragraphs
            ])
            pdf_response += "\n\n" + apa_references
        combined_response += "Chat Response:\n" + pdf_response

    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (conversation_id, username, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, username, "assistant", combined_response, datetime.now()))
        conn.commit()
        message_id = c.lastrowid

    logging.info(f"Chat processed for conversation_id: {conversation_id}")
    return jsonify({"response": combined_response, "messageId": message_id, "conversation_id": conversation_id}), 200

@app.route("/new_chat", methods=["GET"])
def new_chat():
    logging.info("Starting a new chat session.")
    username = request.args.get("username")
    if not username:
        logging.error("Username missing in new_chat request.")
        return jsonify({"error": "Username required to start a new chat"}), 400
    conversation_id = str(uuid.uuid4())
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (conversation_id, username, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, username, "assistant", "Hello, how can I help you?", datetime.now()))
        conn.commit()
    logging.info(f"New conversation created with ID: {conversation_id}")
    return jsonify({"conversation_id": conversation_id, "response": "Hello, how can I help you?"}), 200

@app.route("/processed_pdfs", methods=["GET"])
def processed_pdfs():
    logging.info("Retrieving all processed PDFs.")
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.execute("SELECT pdf_name, content FROM pdf_texts")
        pdfs = [{"pdf_name": row[0], "content": row[1]} for row in c.fetchall()]
    return jsonify(pdfs), 200

@app.route("/all_chat_history", methods=["GET"])
def all_chat_history():
    logging.info("Fetching all chat history.")
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, conversation_id, username, role, content, timestamp, feedback
            FROM chat_history
            ORDER BY timestamp
        ''')
        chat_history = [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5], "feedback": row[6]}
            for row in c.fetchall()
        ]
    return jsonify(chat_history), 200

@app.route("/feedback", methods=["POST"])
def feedback():
    logging.info("Recording feedback.")
    data = request.json
    message_id = data.get("message_id")
    rating = data.get("rating")
    if message_id is None or rating is None:
        logging.error("Invalid feedback data received.")
        return jsonify({"error": "Invalid feedback data"}), 400

    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            logging.error("Rating must be between 1 and 5.")
            return jsonify({"error": "Rating must be between 1 and 5."}), 400
    except ValueError:
        logging.error("Rating must be an integer.")
        return jsonify({"error": "Rating must be an integer."}), 400

    try:
        with sqlite3.connect("chat_history.db") as conn:
            c = conn.cursor()
            c.execute("UPDATE chat_history SET feedback = ? WHERE id = ?", (rating, message_id))
            conn.commit()
        logging.info(f"Feedback recorded for message_id: {message_id} with rating: {rating}")
        return jsonify({"status": "Feedback recorded successfully"}), 200
    except Exception as e:
        logging.error(f"Error recording feedback: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/register", methods=["POST"])
def register():
    logging.info("Registering a new user.")
    data = request.json
    user_id = data.get("username")
    password = data.get("password")
    if user_id is None or password is None:
        logging.error("Invalid registration data.")
        return jsonify({"error": "Invalid registration data"}), 400
    try:
        with sqlite3.connect("user_data.db") as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (user_id, password) VALUES (?, ?)", (user_id, password))
            conn.commit()
        logging.info(f"User {user_id} registered successfully.")
        return jsonify({"status": "User registered successfully"}), 200
    except sqlite3.IntegrityError:
        logging.error(f"User {user_id} already exists.")
        return jsonify({"error": "User ID already exists"}), 400
    except Exception as e:
        logging.error(f"Error registering user: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/login", methods=["POST"])
def login():
    logging.info("Processing login request.")
    data = request.json
    user_id = data.get("username")
    password = data.get("password")
    if user_id is None or password is None:
        logging.error("Invalid login data.")
        return jsonify({"error": "Invalid login data"}), 400
    try:
        with sqlite3.connect("user_data.db") as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE user_id = ?", (user_id,))
            row = c.fetchone()
            if row is None:
                logging.error(f"User {user_id} not found.")
                return jsonify({"error": "User not found"}), 404
            stored_password = row[0]
            if password == stored_password:
                session["username"] = user_id
                logging.info(f"User {user_id} logged in successfully.")
                return jsonify({"status": "Login successful"}), 200
            else:
                logging.error("Invalid password provided.")
                return jsonify({"error": "Invalid password"}), 401
    except Exception as e:
        logging.error(f"Error logging in user: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/conversations", methods=["GET"])
def conversations():
    logging.info("Fetching conversation summaries.")
    username = request.args.get("username")
    if not username:
        logging.error("Username missing in conversations request.")
        return jsonify({"error": "Username required"}), 400
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT conversation_id, COUNT(*) as message_count, MIN(timestamp) as first_time
            FROM chat_history
            WHERE username = ?
            GROUP BY conversation_id
            ORDER BY first_time DESC
        ''', (username,))
        convos = []
        for row in c.fetchall():
            conversation_id = row[0]
            message_count = row[1]
            first_time = row[2]
            c.execute('''
                SELECT content FROM chat_history 
                WHERE conversation_id = ? AND role = 'user'
                ORDER BY timestamp ASC LIMIT 1
            ''', (conversation_id,))
            user_message_row = c.fetchone()
            first_message = user_message_row[0] if user_message_row else ""
            convos.append({
                "conversation_id": conversation_id,
                "first_message": first_message,
                "message_count": message_count,
                "first_time": first_time
            })
    return jsonify(convos), 200

@app.route("/loadConversation", methods=["GET"])
def load_conversation():
    logging.info("Loading conversation details.")
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        logging.error("No conversation_id provided.")
        return jsonify({"error": "Conversation ID required"}), 400
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, conversation_id, username, role, content, timestamp
            FROM chat_history
            WHERE conversation_id = ?
            ORDER BY timestamp
        ''', (conversation_id,))
        messages = [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5]}
            for row in c.fetchall()
        ]
    return jsonify(messages), 200

@app.route("/status", methods=["GET"])
def status():
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Conversation ID required"}), 400
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT COUNT(*)
            FROM chat_history
            WHERE conversation_id = ?
        ''', (conversation_id,))
        count = c.fetchone()[0]
    return jsonify({"message_count": count}), 200
from flask import Flask, request, jsonify, render_template, session
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from azure.storage.blob import BlobServiceClient
import fitz  # PyMuPDF
import os
import sqlite3
import logging
import uuid
from datetime import datetime
import nltk
from openai import AzureOpenAI
import re
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import tempfile

# Configure logging with INFO level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download required NLTK resources.
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Initialize the Flask app and secret key
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a secure secret key

# ---------------------
# Global Configuration Variables
# ---------------------
OPENAI_API_KEY = "2yEKfg6KvcBXgL0YV1fkHYweHoo18ZV2QumJP3HtRiT4eItEQmsrJQQJ99AKACHrzpqXJ3w3AAABACOG3dpB"
DEPLOYMENT_NAME = "gpt-4"
OPENAI_API_ENDPOINT = ("https://alconaiservice.openai.azure.com/"
                         "openai/deployments/gpt-4/chat/completions?api-version=2024-02-15-preview")

azure_blob_connection_string = ("DefaultEndpointsProtocol=https;"
                                  "AccountName=formulationscondensed;"
                                  "AccountKey=wJh4mvSgbEOoOFfcDsLVX7GSIzw59gkJF0Do77rOmOv60EFCODcNvNTaHVusWcVrvnLtkK3x4wX5+AStWjV9Ew=="
                                  "EndpointSuffix=core.windows.net")
azure_blob_container_name = "input"
blob_service_client = BlobServiceClient.from_connection_string(azure_blob_connection_string)
container_client = blob_service_client.get_container_client(azure_blob_container_name)

# Form Recognizer configuration (replace with your actual key if available)
form_recognizer_endpoint = "https://documentanalysisclient.cognitiveservices.azure.com/"
form_recognizer_key = "YOUR_FORM_RECOGNIZER_KEY"  # Replace with your actual Form Recognizer key

# Local directory to store downloaded PDFs
PDF_DOWNLOAD_PATH = "downloaded_pdfs"
os.makedirs(PDF_DOWNLOAD_PATH, exist_ok=True)

# ---------------------
# Database Initialization Functions
# ---------------------
def init_db():
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                feedback INTEGER
            )
        ''')
        conn.commit()

def init_user_db():
    with sqlite3.connect("user_data.db") as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

def init_pdf_cache_db():
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS pdf_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_name TEXT NOT NULL,
                content TEXT NOT NULL
            )
        ''')
        conn.commit()

# ---------------------
# PDF Processing Logic
# ---------------------
def download_blob(blob_name):
    """
    Downloads a PDF from Azure Blob Storage and saves it locally.
    """
    try:
        local_path = os.path.join(PDF_DOWNLOAD_PATH, blob_name)
        blob_client = container_client.get_blob_client(blob_name)
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        logging.info(f"Downloaded and saved PDF to: {local_path}")
        return local_path
    except Exception as e:
        logging.error(f"Error downloading blob {blob_name}: {e}")
        return None

def extract_text_from_pdf_with_recognition(pdf_bytes):
    """
    Saves PDF bytes to a temporary file, sends the file to Azure Form Recognizer 
    (using the prebuilt-layout model) for text extraction, and removes the temporary file.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_filename = tmp_file.name
        logging.info(f"Temporary PDF saved to: {tmp_filename}")
        
        document_analysis_client = DocumentAnalysisClient(
            form_recognizer_endpoint, AzureKeyCredential(form_recognizer_key)
        )
        
        with open(tmp_filename, "rb") as f:
            poller = document_analysis_client.begin_analyze_document("prebuilt-layout", f)
            result = poller.result()
        
        extracted_text = ""
        for page in result.pages:
            for line in page.lines:
                extracted_text += line.content + "\n"
        
        os.remove(tmp_filename)
        logging.info("Temporary PDF file removed.")
        return extracted_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF using Form Recognizer: {e}")
        return ""

def clean_extracted_text(text):
    return re.sub(r'\n+', '\n', text).strip()

def process_single_pdf(pdf_name):
    """
    Downloads a PDF from Azure Blob Storage, extracts text using the updated logic,
    cleans the extracted text, and returns a tuple (pdf_name, cleaned_text).
    """
    try:
        logging.info(f"Processing PDF: {pdf_name}")
        pdf_path = download_blob(pdf_name)
        if not pdf_path:
            logging.error(f"Failed to download PDF: {pdf_name}")
            return None
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        extracted_text = extract_text_from_pdf_with_recognition(pdf_bytes)
        cleaned_text = clean_extracted_text(extracted_text)
        logging.info(f"Extracted text from {pdf_name}: {cleaned_text[:100]}...")
        return (pdf_name, cleaned_text)
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_name}: {e}")
        return None

def batch_insert_pdfs(records):
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.executemany('''
            INSERT INTO pdf_texts (pdf_name, content)
            VALUES (?, ?)
        ''', records)
        conn.commit()
        logging.info(f"Inserted {len(records)} PDFs into the database")

def list_blobs():
    try:
        return [blob.name for blob in container_client.list_blobs()]
    except Exception as e:
        logging.error(f"Error listing blobs: {e}")
        return []

def preprocess_pdfs_to_db(limit=100):
    pdf_list = list_blobs()[:limit]
    logging.info(f"Processing {len(pdf_list)} PDFs")
    batch = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
        for result in executor.map(process_single_pdf, pdf_list):
            if result:
                batch.append(result)
                if len(batch) >= 100:
                    batch_insert_pdfs(batch)
                    batch = []
    if batch:
        batch_insert_pdfs(batch)

# ---------------------
# Chat, Summaries, and Utility Functions
# ---------------------
def save_message_to_db(username, role, content, conversation_id):
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (conversation_id, username, role, content, timestamp, feedback)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (conversation_id, username, role, content, datetime.now(), False))
        conn.commit()

def load_chat_history():
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, conversation_id, username, role, content, timestamp
            FROM chat_history
            ORDER BY timestamp
        ''')
        return [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5]}
            for row in c.fetchall()
        ]

def query_openai(user_message, relevant_paragraphs, conversation_id):
    try:
        client = AzureOpenAI(api_key=OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=OPENAI_API_ENDPOINT)
        with sqlite3.connect("chat_history.db") as conn:
            c = conn.cursor()
            c.execute('''
                SELECT role, content
                FROM chat_history
                WHERE conversation_id = ?
                ORDER BY timestamp
            ''', (conversation_id,))
            history = [{"role": role, "content": content} for role, content in c.fetchall()]
        system_msg = {
            "role": "system",
            "content": "You are a helpful AI assistant specializing in lens formulations, chemistry, optics, and AI-driven research analysis. Focus on the most relevant details."
        }
        messages = [system_msg] + history[-5:]
        relevant_text = "\n\n".join([
            f"{p['paragraph']} (Source: {p['source']}, PDF: {p['pdf_name']})" for p in relevant_paragraphs
        ])
        messages.append({
            "role": "user",
            "content": user_message + "\n\nRelevant paragraphs:\n" + relevant_text
        })
        logging.info(f"Sending {len(messages)} messages to OpenAI for query.")
        response = client.chat.completions.create(model="gpt-4", messages=messages)
        logging.info("Received response from OpenAI.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error querying OpenAI: {e}")
        return "Sorry, I encountered an error while processing your request."

def is_relevant(paragraph, query):
    query_words = query.lower().split()
    return any(word in paragraph.lower() for word in query_words)

def parse_pdf_metadata(pdf_name):
    base_name = pdf_name.rsplit('.', 1)[0]
    parts = base_name.split('_')
    if len(parts) >= 4:
        return {"author": parts[0], "year": parts[1], "title": parts[2], "publisher": parts[3]}
    return {"author": "Unknown", "year": "n.d.", "title": base_name, "publisher": "Unknown"}

def search_pdfs_helper(user_message, max_paragraphs=5):
    relevant_paragraphs = []
    try:
        with sqlite3.connect("pdf_cache.db") as conn:
            c = conn.cursor()
            c.execute("SELECT pdf_name, content FROM pdf_texts")
            pdfs = [{"name": row[0], "content": row[1]} for row in c.fetchall()]
    except Exception as e:
        logging.error(f"Error retrieving PDFs from cache: {e}")
        return relevant_paragraphs
    for pdf in pdfs:
        metadata = parse_pdf_metadata(pdf["name"])
        apa_citation = f"{metadata['author']} ({metadata['year']}). {metadata['title']}. {metadata['publisher']}."
        paragraphs = [p.strip() for p in pdf["content"].split("\n") if p.strip()]
        for paragraph in paragraphs:
            if is_relevant(paragraph, user_message):
                relevant_paragraphs.append({
                    "paragraph": paragraph,
                    "source": apa_citation,
                    "pdf_name": pdf["name"]
                })
    return relevant_paragraphs[:max_paragraphs]

def update_context_summary(conversation_id, max_history_messages=20):
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT role, content
            FROM chat_history
            WHERE conversation_id = ? AND role != 'summary'
            ORDER BY timestamp
        ''', (conversation_id,))
        all_messages = c.fetchall()
    if len(all_messages) <= max_history_messages:
        logging.info("No need to update summary; conversation history is within limit.")
        return
    conversation_text = "\n".join([f"{role}: {content}" for role, content in all_messages])
    summary_prompt = (
        "The following is a conversation history:\n\n"
        f"{conversation_text}\n\n"
        "Please provide a concise summary that captures the key points of the conversation."
    )
    logging.info("Requesting summary from OpenAI for context compression.")
    summary_text = call_openai_summary(summary_prompt).strip()
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (conversation_id, username, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, "system", "summary", summary_text, datetime.now()))
        conn.commit()
    logging.info("Context summary updated for conversation_id: " + conversation_id)

def call_openai_summary(prompt):
    try:
        client = AzureOpenAI(api_key=OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=OPENAI_API_ENDPOINT)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a summarization engine."},
                {"role": "user", "content": prompt}
            ]
        )
        logging.info("Summary received from OpenAI.")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in summarization call: {e}")
        return "Summary unavailable."

def decide_workflow(user_message):
    try:
        decision_prompt = (
            "You are an expert in lens formulations and patents. Based on the user's question, decide which workflow to use:\n"
            "1. Respond with 'patent' if the question is specifically asking for patent information.\n"
            "2. Respond with 'pdf' if the question requires searching through technical PDF content.\n"
            "3. Respond with 'both' if the question seems to need both patent info and PDF content.\n\n"
            f"User question: \"{user_message}\"\n\n"
            "Please reply with just one word: 'patent', 'pdf', or 'both'."
        )
        client = AzureOpenAI(api_key=OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=OPENAI_API_ENDPOINT)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a workflow decision assistant."},
                {"role": "user", "content": decision_prompt}
            ]
        )
        decision = response.choices[0].message.content.strip().lower()
        logging.info(f"Workflow decision from OpenAI: {decision}")
        if decision not in ["patent", "pdf", "both"]:
            logging.info("Decision not recognized. Defaulting to 'both'.")
            decision = "both"
        return decision
    except Exception as e:
        logging.error(f"Error in decide_workflow: {e}")
        return "both"

def get_patent_info(user_message):
    try:
        patent_prompt = (
            "You are a patent research assistant specialized in lens formulations. "
            "Provide detailed patent information based on the following query:\n"
            f"{user_message}\n\n"
            "Include relevant patent numbers, filing dates, and a brief summary if available."
        )
        client = AzureOpenAI(api_key=OPENAI_API_KEY, api_version="2024-02-01", azure_endpoint=OPENAI_API_ENDPOINT)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a patent research assistant."},
                {"role": "user", "content": patent_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in get_patent_info: {e}")
        return "Patent info unavailable due to an error."

# ---------------------
# API Endpoints
# ---------------------
@app.route("/")
def index():
    logging.info("Rendering index page.")
    return render_template("index.html")

@app.route("/list_chats", methods=["GET"])
def list_chats():
    logging.info("Listing chat messages.")
    username = request.args.get("username")
    query = '''
        SELECT id, conversation_id, username, role, content, timestamp
        FROM chat_history
        ORDER BY timestamp
    '''
    params = ()
    if username:
        query = '''
            SELECT id, conversation_id, username, role, content, timestamp
            FROM chat_history
            WHERE username = ?
            ORDER BY timestamp
        '''
        params = (username,)
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute(query, params)
        chat_history = [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5]}
            for row in c.fetchall()
        ]
    return jsonify(chat_history), 200

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    logging.info("Retrieving full chat history.")
    chat_history = load_chat_history()
    return jsonify(chat_history), 200

@app.route("/loadChat", methods=["GET"])
def load_chat():
    logging.info("Loading chat history for a given chat_id.")
    chat_id = request.args.get("chat_id")
    if not chat_id:
        logging.error("No chat_id provided.")
        return jsonify({"error": "No chat ID provided"}), 400
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, conversation_id, username, role, content, timestamp
            FROM chat_history
            WHERE id = ?
            ORDER BY timestamp
        ''', (chat_id,))
        chat_history = [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5]}
            for row in c.fetchall()
        ]
    return jsonify(chat_history), 200

@app.route("/search_pdfs", methods=["GET"])
def search_pdfs():
    logging.info("Searching PDFs.")
    search_term = request.args.get("search_term")
    if not search_term:
        logging.error("No search_term provided.")
        return jsonify({"error": "No search term provided"}), 400
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT pdf_name, content
            FROM pdf_texts
            WHERE content LIKE ?
            LIMIT 1
        ''', (f"%{search_term}%",))
        row = c.fetchone()
    return jsonify(row), 200

@app.route("/chat", methods=["POST"])
def chat():
    logging.info("Received chat request.")
    data = request.json
    user_message = data.get("message", "").strip()
    username = session.get("username")
    conversation_id = data.get("conversation_id", "").strip() or str(uuid.uuid4())

    if not user_message or not username:
        logging.error("Missing username or message.")
        return jsonify({"error": "Username and message are required"}), 400

    workflow_decision = decide_workflow(user_message)
    logging.info(f"Workflow decision: {workflow_decision}")
    combined_response = ""

    if workflow_decision in ["patent", "both"]:
        patent_info = get_patent_info(user_message)
        combined_response += "Patent Information:\n" + patent_info + "\n\n"

    if workflow_decision in ["pdf", "both"]:
        relevant_paragraphs = search_pdfs_helper(user_message, max_paragraphs=5)
        save_message_to_db(username, "user", user_message, conversation_id)
        update_context_summary(conversation_id, max_history_messages=20)
        pdf_response = query_openai(user_message, relevant_paragraphs, conversation_id)
        if relevant_paragraphs:
            apa_references = "\n\nReferences (APA format):\n" + "\n".join([
                f"{p['source']} (PDF: {p['pdf_name']}) - Excerpt: \"{p['paragraph']}\""
                for p in relevant_paragraphs
            ])
            pdf_response += "\n\n" + apa_references
        combined_response += "Chat Response:\n" + pdf_response

    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (conversation_id, username, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, username, "assistant", combined_response, datetime.now()))
        conn.commit()
        message_id = c.lastrowid

    logging.info(f"Chat processed for conversation_id: {conversation_id}")
    return jsonify({"response": combined_response, "messageId": message_id, "conversation_id": conversation_id}), 200

@app.route("/new_chat", methods=["GET"])
def new_chat():
    logging.info("Starting a new chat session.")
    username = request.args.get("username")
    if not username:
        logging.error("Username missing in new_chat request.")
        return jsonify({"error": "Username required to start a new chat"}), 400
    conversation_id = str(uuid.uuid4())
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (conversation_id, username, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, username, "assistant", "Hello, how can I help you?", datetime.now()))
        conn.commit()
    logging.info(f"New conversation created with ID: {conversation_id}")
    return jsonify({"conversation_id": conversation_id, "response": "Hello, how can I help you?"}), 200

@app.route("/processed_pdfs", methods=["GET"])
def processed_pdfs():
    logging.info("Retrieving all processed PDFs.")
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.execute("SELECT pdf_name, content FROM pdf_texts")
        pdfs = [{"pdf_name": row[0], "content": row[1]} for row in c.fetchall()]
    return jsonify(pdfs), 200

@app.route("/all_chat_history", methods=["GET"])
def all_chat_history():
    logging.info("Fetching all chat history.")
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, conversation_id, username, role, content, timestamp, feedback
            FROM chat_history
            ORDER BY timestamp
        ''')
        chat_history = [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5], "feedback": row[6]}
            for row in c.fetchall()
        ]
    return jsonify(chat_history), 200

@app.route("/feedback", methods=["POST"])
def feedback():
    logging.info("Recording feedback.")
    data = request.json
    message_id = data.get("message_id")
    rating = data.get("rating")
    if message_id is None or rating is None:
        logging.error("Invalid feedback data received.")
        return jsonify({"error": "Invalid feedback data"}), 400

    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            logging.error("Rating must be between 1 and 5.")
            return jsonify({"error": "Rating must be between 1 and 5."}), 400
    except ValueError:
        logging.error("Rating must be an integer.")
        return jsonify({"error": "Rating must be an integer."}), 400

    try:
        with sqlite3.connect("chat_history.db") as conn:
            c = conn.cursor()
            c.execute("UPDATE chat_history SET feedback = ? WHERE id = ?", (rating, message_id))
            conn.commit()
        logging.info(f"Feedback recorded for message_id: {message_id} with rating: {rating}")
        return jsonify({"status": "Feedback recorded successfully"}), 200
    except Exception as e:
        logging.error(f"Error recording feedback: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/register", methods=["POST"])
def register():
    logging.info("Registering a new user.")
    data = request.json
    user_id = data.get("username")
    password = data.get("password")
    if user_id is None or password is None:
        logging.error("Invalid registration data.")
        return jsonify({"error": "Invalid registration data"}), 400
    try:
        with sqlite3.connect("user_data.db") as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (user_id, password) VALUES (?, ?)", (user_id, password))
            conn.commit()
        logging.info(f"User {user_id} registered successfully.")
        return jsonify({"status": "User registered successfully"}), 200
    except sqlite3.IntegrityError:
        logging.error(f"User {user_id} already exists.")
        return jsonify({"error": "User ID already exists"}), 400
    except Exception as e:
        logging.error(f"Error registering user: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/login", methods=["POST"])
def login():
    logging.info("Processing login request.")
    data = request.json
    user_id = data.get("username")
    password = data.get("password")
    if user_id is None or password is None:
        logging.error("Invalid login data.")
        return jsonify({"error": "Invalid login data"}), 400
    try:
        with sqlite3.connect("user_data.db") as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE user_id = ?", (user_id,))
            row = c.fetchone()
            if row is None:
                logging.error(f"User {user_id} not found.")
                return jsonify({"error": "User not found"}), 404
            stored_password = row[0]
            if password == stored_password:
                session["username"] = user_id
                logging.info(f"User {user_id} logged in successfully.")
                return jsonify({"status": "Login successful"}), 200
            else:
                logging.error("Invalid password provided.")
                return jsonify({"error": "Invalid password"}), 401
    except Exception as e:
        logging.error(f"Error logging in user: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/conversations", methods=["GET"])
def conversations():
    logging.info("Fetching conversation summaries.")
    username = request.args.get("username")
    if not username:
        logging.error("Username missing in conversations request.")
        return jsonify({"error": "Username required"}), 400
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT conversation_id, COUNT(*) as message_count, MIN(timestamp) as first_time
            FROM chat_history
            WHERE username = ?
            GROUP BY conversation_id
            ORDER BY first_time DESC
        ''', (username,))
        convos = []
        for row in c.fetchall():
            conversation_id = row[0]
            message_count = row[1]
            first_time = row[2]
            c.execute('''
                SELECT content FROM chat_history 
                WHERE conversation_id = ? AND role = 'user'
                ORDER BY timestamp ASC LIMIT 1
            ''', (conversation_id,))
            user_message_row = c.fetchone()
            first_message = user_message_row[0] if user_message_row else ""
            convos.append({
                "conversation_id": conversation_id,
                "first_message": first_message,
                "message_count": message_count,
                "first_time": first_time
            })
    return jsonify(convos), 200

@app.route("/loadConversation", methods=["GET"])
def load_conversation():
    logging.info("Loading conversation details.")
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        logging.error("No conversation_id provided.")
        return jsonify({"error": "Conversation ID required"}), 400
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, conversation_id, username, role, content, timestamp
            FROM chat_history
            WHERE conversation_id = ?
            ORDER BY timestamp
        ''', (conversation_id,))
        messages = [
            {"chat_id": row[0], "conversation_id": row[1], "username": row[2],
             "role": row[3], "content": row[4], "timestamp": row[5]}
            for row in c.fetchall()
        ]
    return jsonify(messages), 200

@app.route("/status", methods=["GET"])
def status():
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Conversation ID required"}), 400
    with sqlite3.connect("chat_history.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT COUNT(*)
            FROM chat_history
            WHERE conversation_id = ?
        ''', (conversation_id,))
        count = c.fetchone()[0]
    return jsonify({"message_count": count}), 200

# ---------------------
# App Initialization
# ---------------------
if __name__ == "__main__":
    init_db()
    init_user_db()
    # Uncomment the following lines to initialize and process the PDF cache:
    # init_pdf_cache_db()
    # preprocess_pdfs_to_db(limit=100)
    app.run(host="0.0.0.0", port=8000, debug=True)
# ---------------------
# App Initialization
# ---------------------
if __name__ == "__main__":
    init_db()
    init_user_db()
    # Uncomment the following lines to initialize and process the PDF cache:
    # init_pdf_cache_db()
    # preprocess_pdfs_to_db(limit=100)
    app.run(host="0.0.0.0", port=8000, debug=True)
