from flask import Flask, request, jsonify, render_template
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from azure.storage.blob import BlobServiceClient
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
import time
import json
import sqlite3
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
from openai import AzureOpenAI
from flasgger import Swagger, swag_from
import logging
import re
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import uuid  # For generating conversation IDs

# Configure logging with INFO level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK resources.
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Initialize the Flask app and Swagger documentation.
app = Flask(__name__)
swagger = Swagger(app)

# ---------------------
# Global Configuration Variables
# ---------------------
OPENAI_API_KEY = ""
DEPLOYMENT_NAME = "gpt-4"
OPENAI_API_ENDPOINT = (
    f"https://alconaiservice.openai.azure.com/"
    f"openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
)

azure_blob_connection_string = (
    "DefaultEndpointsProtocol=https;"
    "AccountName=formulationscondensed;"
    "AccountKey="
    "EndpointSuffix=core.windows.net"
)
azure_blob_container_name = "input"
blob_service_client = BlobServiceClient.from_connection_string(azure_blob_connection_string)
container_client = blob_service_client.get_container_client(azure_blob_container_name)

# Form Recognizer configuration for document analysis (for metadata extraction)
FORM_RECOGNIZER_ENDPOINT = "https://documentanalysisclient.cognitiveservices.azure.com/"
FORM_RECOGNIZER_KEY = ""

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

def init_pdf_cache_db():
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        # Added a metadata TEXT column to store extracted metadata as JSON
        c.execute('''
            CREATE TABLE IF NOT EXISTS pdf_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_name TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT
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

# ---------------------
# Utility Functions
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

def extract_text_and_metadata_from_pdf(pdf_bytes):
    """
    Uses the Form Recognizer prebuilt-document model to extract both text and metadata.
    """
    document_analysis_client = DocumentAnalysisClient(endpoint=FORM_RECOGNIZER_ENDPOINT,
                                                        credential=AzureKeyCredential(FORM_RECOGNIZER_KEY))
    try:
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", pdf_bytes)
        result = poller.result()
        full_text = ""
        for page in result.pages:
            for line in page.lines:
                full_text += line.content + "\n"
        # Extract metadata from the first document if available
        metadata = {}
        if result.documents:
            doc = result.documents[0]
            for field_name, field in doc.fields.items():
                # Save the field's value if it exists; otherwise, empty string
                metadata[field_name] = field.value if field.value is not None else ""
        return full_text.strip(), metadata
    except Exception as e:
        logging.error(f"Error extracting text and metadata from PDF: {e}")
        return "", {}

def process_single_pdf(pdf_name):
    try:
        logging.info(f"Processing PDF: {pdf_name}")
        pdf_bytes = download_blob(pdf_name)
        if not pdf_bytes:
            logging.error(f"Failed to download PDF: {pdf_name}")
            return None
        full_text, metadata = extract_text_and_metadata_from_pdf(pdf_bytes)
        cleaned_text = clean_extracted_text(full_text)
        logging.info(f"Extracted text from {pdf_name}: {cleaned_text[:100]}...")
        # Return a tuple with pdf_name, content, and metadata (as JSON string)
        return (pdf_name, cleaned_text, json.dumps(metadata))
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_name}: {e}")
        return None

def clean_extracted_text(text):
    return re.sub(r'\n+', '\n', text).strip()

def preprocess_pdfs_to_db(limit=100):
    pdf_list = list_blobs()[:limit]
    logging.info(f"Processing {len(pdf_list)} PDFs")
    batch = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for result in executor.map(process_single_pdf, pdf_list):
            if result:
                batch.append(result)
                if len(batch) >= 100:
                    batch_insert_pdfs(batch)
                    batch = []
    if batch:
        batch_insert_pdfs(batch)

def list_blobs():
    try:
        return [blob.name for blob in container_client.list_blobs()]
    except Exception as e:
        logging.error(f"Error listing blobs: {e}")
        return []

def download_blob(blob_name):
    try:
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall()
    except Exception as e:
        logging.error(f"Error downloading blob {blob_name}: {e}")
        return None

def batch_insert_pdfs(records):
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.executemany('''
            INSERT INTO pdf_texts (pdf_name, content, metadata)
            VALUES (?, ?, ?)
        ''', records)
        conn.commit()
        logging.info(f"Inserted {len(records)} PDFs into the database")

# ---------------------
# PDF Search & Citation Functions
# ---------------------
def parse_pdf_metadata(pdf_name, metadata_json):
    """
    Instead of parsing the file name, we now use the extracted metadata if available.
    If metadata is empty, fall back to basic parsing.
    """
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except Exception:
        metadata = {}
    if metadata:
        # Example: Use fields "Author", "Title", "PublicationDate" if available.
        author = metadata.get("Author", "Unknown")
        title = metadata.get("Title", pdf_name.rsplit('.', 1)[0])
        pub_date = metadata.get("PublicationDate", "n.d.")
        publisher = metadata.get("Publisher", "Unknown")
        return f"{author} ({pub_date}). {title}. {publisher}."
    else:
        # Fallback: Parse the filename.
        base_name = pdf_name.rsplit('.', 1)[0]
        parts = base_name.split('_')
        if len(parts) >= 4:
            return f"{parts[0]} ({parts[1]}). {parts[2]}. {parts[3]}."
        return f"Unknown (n.d.). {base_name}. Unknown."

def is_relevant(paragraph, query):
    query_words = query.lower().split()
    return any(word in paragraph.lower() for word in query_words)

def search_pdfs_helper(user_message, max_paragraphs=5):
    relevant_paragraphs = []
    try:
        with sqlite3.connect("pdf_cache.db") as conn:
            c = conn.cursor()
            c.execute("SELECT pdf_name, content, metadata FROM pdf_texts")
            pdfs = [{"name": row[0], "content": row[1], "metadata": row[2]} for row in c.fetchall()]
    except Exception as e:
        logging.error(f"Error retrieving PDFs from cache: {e}")
        return relevant_paragraphs

    for pdf in pdfs:
        citation = parse_pdf_metadata(pdf["name"], pdf["metadata"])
        paragraphs = [p.strip() for p in pdf["content"].split("\n") if p.strip()]
        for paragraph in paragraphs:
            if is_relevant(paragraph, user_message):
                relevant_paragraphs.append({
                    "paragraph": paragraph,
                    "source": citation,
                    "pdf_name": pdf["name"]
                })
    return relevant_paragraphs[:max_paragraphs]

# ---------------------
# Context Summarization Functions
# ---------------------
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
    summary_response = call_openai_summary(summary_prompt)
    summary_text = summary_response.strip()

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

# ---------------------
# OpenAI Logic Adapter Functions
# ---------------------
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
@app.route('/')
@swag_from({
    'get': {
        'summary': 'Render Home Page',
        'description': 'Renders the home page for the application.',
        'responses': {'200': {'description': 'Home page rendered successfully.'}}
    }
})
def index():
    logging.info("Rendering index page.")
    return render_template('index.html')

@app.route('/list_chats', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'List Individual Chat Messages',
        'description': 'Retrieve a list of individual chat messages, optionally filtered by username.',
        'parameters': [
            {'name': 'username', 'in': 'query', 'type': 'string', 'required': False, 'description': 'Optional username to filter chat messages.'}
        ],
        'responses': {
            '200': {
                'description': 'A list of chat messages.',
                'schema': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'chat_id': {'type': 'integer'},
                            'conversation_id': {'type': 'string'},
                            'username': {'type': 'string'},
                            'role': {'type': 'string'},
                            'content': {'type': 'string'},
                            'timestamp': {'type': 'string'}
                        }
                    }
                }
            }
        }
    }
})
def list_chats():
    logging.info("Listing chat messages.")
    username = request.args.get('username')
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

@app.route('/get_chat_history', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Get All Chat History',
        'description': 'Retrieve all chat history from the database.',
        'responses': {'200': {'description': 'A list of all chat messages with details.'}}
    }
})
def get_chat_history():
    logging.info("Retrieving full chat history.")
    chat_history = load_chat_history()
    return jsonify(chat_history), 200

@app.route('/loadChat', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Load Chat by ID',
        'description': 'Load a specific chat conversation by chat ID.',
        'parameters': [
            {'name': 'chat_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'The ID of the chat to load.'}
        ],
        'responses': {
            '200': {'description': 'Chat conversation loaded successfully.'},
            '400': {'description': 'No chat ID provided.'}
        }
    }
})
def load_chat():
    logging.info("Loading chat history for a given chat_id.")
    chat_id = request.args.get('chat_id')
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

@app.route('/search_pdfs', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Search PDFs',
        'description': 'Search the cached PDFs for a given term.',
        'parameters': [
            {'name': 'search_term', 'in': 'query', 'type': 'string', 'required': True, 'description': 'The term to search for in PDF contents.'}
        ],
        'responses': {
            '200': {
                'description': 'The first matching PDF record.',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'pdf_name': {'type': 'string'},
                        'content': {'type': 'string'},
                        'metadata': {'type': 'string'}
                    }
                }
            },
            '400': {'description': 'No search term provided.'}
        }
    }
})
def search_pdfs():
    logging.info("Searching PDFs.")
    search_term = request.args.get('search_term')
    if not search_term:
        logging.error("No search_term provided.")
        return jsonify({"error": "No search term provided"}), 400
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.execute('''
            SELECT pdf_name, content, metadata
            FROM pdf_texts
            WHERE content LIKE ?
            LIMIT 1
        ''', (f"%{search_term}%",))
        row = c.fetchone()
    return jsonify(row), 200

@app.route('/chat', methods=['POST'])
@swag_from({
    'post': {
        'summary': 'Send Chat Message',
        'description': (
            "Receive a chat message from a user, decide which workflow to execute "
            "(patent info, PDF search, or both), call the appropriate functions, "
            "and save both the user and assistant messages. The assistant message includes APA-formatted details of relevant PDF paragraphs."
        ),
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message': {'type': 'string'},
                        'username': {'type': 'string'},
                        'conversation_id': {'type': 'string'}
                    },
                    'required': ['message', 'username', 'conversation_id']
                }
            }
        ],
        'responses': {
            '200': {
                'description': 'Chat processed successfully with response and message ID.',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'response': {'type': 'string'},
                        'messageId': {'type': 'integer'}
                    }
                }
            },
            '400': {'description': 'Missing required fields.'}
        }
    }
})
def chat():
    logging.info("Received chat request.")
    data = request.json
    user_message = data.get("message", "").strip()
    username = data.get("username", "").strip()
    conversation_id = data.get("conversation_id", "").strip()

    if not user_message or not username or not conversation_id:
        logging.error("Missing username, conversation_id, or message.")
        return jsonify({"error": "Username, conversation ID, and message are required"}), 400

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
    return jsonify({"response": combined_response, "messageId": message_id}), 200

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
            all_history = c.fetchall()
        
        summary_messages = [msg for msg in all_history if msg[0] == "summary"]
        other_messages = [msg for msg in all_history if msg[0] != "summary"]
        recent_messages = other_messages[-5:]
        
        messages = [{
            "role": "system",
            "content": (
                "You are a helpful AI assistant specializing in lens formulations, chemistry, optics, and AI-driven research analysis. "
                "Focus on the most relevant details."
            )
        }]
        if summary_messages:
            messages.append({"role": "system", "content": f"Context summary: {summary_messages[-1][1]}"} )
        for role, content in recent_messages:
            messages.append({"role": role, "content": content})
        
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

@app.route('/new_chat', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Start New Chat Session',
        'description': 'Generate a new conversation ID and add a welcome message.',
        'parameters': [
            {'name': 'username', 'in': 'query', 'type': 'string', 'required': True, 'description': 'Username starting the chat.'}
        ],
        'responses': {
            '200': {
                'description': 'New chat session started.',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'conversation_id': {'type': 'string'},
                        'response': {'type': 'string'}
                    }
                }
            },
            '400': {'description': 'Username not provided.'}
        }
    }
})
def new_chat():
    logging.info("Starting a new chat session.")
    username = request.args.get('username')
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

@app.route('/processed_pdfs', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Get Processed PDFs',
        'description': 'Retrieve a list of all processed PDFs and their content, including metadata.',
        'responses': {'200': {'description': 'A list of processed PDFs with their content and metadata.'}}
    }
})
def processed_pdfs():
    logging.info("Retrieving all processed PDFs.")
    with sqlite3.connect("pdf_cache.db") as conn:
        c = conn.cursor()
        c.execute('SELECT pdf_name, content, metadata FROM pdf_texts')
        pdfs = [{"pdf_name": row[0], "content": row[1], "metadata": row[2]} for row in c.fetchall()]
    return jsonify(pdfs), 200

@app.route('/all_chat_history', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Retrieve All Chat History',
        'description': 'Retrieve a complete list of all chat messages, including feedback.',
        'responses': {'200': {'description': 'All chat history.'}}
    }
})
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

@app.route('/feedback', methods=['POST'])
@swag_from({
    'post': {
        'summary': 'Submit Feedback',
        'description': 'Record a 5‑star rating feedback for a specific chat message.',
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message_id': {'type': 'integer'},
                        'rating': {'type': 'integer', 'minimum': 1, 'maximum': 5}
                    },
                    'required': ['message_id', 'rating']
                }
            }
        ],
        'responses': {
            '200': {'description': 'Feedback recorded successfully.'},
            '400': {'description': 'Invalid feedback data.'},
            '500': {'description': 'Internal server error.'}
        }
    }
})
def feedback():
    logging.info("Recording feedback.")
    data = request.json
    message_id = data.get('message_id')
    rating = data.get('rating')

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
            c.execute('UPDATE chat_history SET feedback = ? WHERE id = ?', (rating, message_id))
            conn.commit()
        logging.info(f"Feedback recorded for message_id: {message_id} with rating: {rating}")
        return jsonify({"status": "Feedback recorded successfully"}), 200
    except Exception as e:
        logging.error(f"Error recording feedback: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/register', methods=['POST'])
@swag_from({
    'post': {
        'summary': 'Register New User',
        'description': 'Register a new user with a username and password.',
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'username': {'type': 'string'},
                        'password': {'type': 'string'}
                    },
                    'required': ['username', 'password']
                }
            }
        ],
        'responses': {
            '200': {'description': 'Register users successfully.'},
            '400': {'description': 'Invalid Register users data.'},
            '500': {'description': 'Internal server error.'}
        }
    }
})
def register():
    logging.info("Registering a new user.")
    data = request.json
    user_id = data.get('username')
    password = data.get('password')

    if user_id is None or password is None:
        logging.error("Invalid registration data.")
        return jsonify({"error": "Invalid registration data"}), 400

    try:
        with sqlite3.connect("user_data.db") as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (user_id, password) VALUES (?, ?)', (user_id, password))
            conn.commit()
        logging.info(f"User {user_id} registered successfully.")
        return jsonify({"status": "User registered successfully"}), 200
    except sqlite3.IntegrityError:
        logging.error(f"User {user_id} already exists.")
        return jsonify({"error": "User ID already exists"}), 400
    except Exception as e:
        logging.error(f"Error registering user: {e}")
        return jsonify({"error": "Internal server error"}), 500



@app.route('/loadConversation', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Load Full Conversation',
        'description': 'Load all messages for a given conversation ID.',
        'parameters': [
            {'name': 'conversation_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'The conversation ID to load messages from.'}
        ],
        'responses': {
            '200': {'description': 'Conversation loaded successfully.'},
            '400': {'description': 'Conversation ID required.'}
        }
    }
})
def load_conversation():
    logging.info("Loading conversation details.")
    conversation_id = request.args.get('conversation_id')
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

@app.route('/conversations', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Get Conversation Summaries',
        'description': 'Retrieve summaries of conversations for a specific user. The summary includes the conversation ID, first user message, and message count.',
        'parameters': [
            {'name': 'username', 'in': 'query', 'type': 'string', 'required': True, 'description': 'Username whose conversations are to be retrieved.'}
        ],
        'responses': {
            '200': {
                'description': 'Conversation summaries retrieved successfully.',
                'schema': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'conversation_id': {'type': 'string'},
                            'first_message': {'type': 'string'},
                            'message_count': {'type': 'integer'},
                            'first_time': {'type': 'string'}
                        }
                    }
                }
            },
            '400': {'description': 'Username required.'}
        }
    }
})
def conversations_route():
    logging.info("Fetching conversation summaries.")
    username = request.args.get('username')
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

@app.route('/loadConversation', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Load Full Conversation',
        'description': 'Load all messages for a given conversation ID.',
        'parameters': [
            {'name': 'conversation_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'The conversation ID to load messages from.'}
        ],
        'responses': {
            '200': {'description': 'Conversation loaded successfully.'},
            '400': {'description': 'Conversation ID required.'}
        }
    }
})
def load_conversation_route():
    logging.info("Loading conversation details.")
    conversation_id = request.args.get('conversation_id')
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

@app.route('/users', methods=['GET'])
@swag_from({
    'responses': {
        '200': {
            'description': 'A list of users',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'integer'},
                        'user_id': {'type': 'string'}
                    }
                }
            }
        },
        '500': {'description': 'Internal server error'}
    }
})
def get_users():
    try:
        with sqlite3.connect("user_data.db") as conn:
            c = conn.cursor()
            c.execute("SELECT id, user_id FROM users ORDER BY id ASC")
            rows = c.fetchall()
        users = [{"id": row[0], "user_id": row[1]} for row in rows]
        logging.info(f"Retrieved {len(users)} users.")
        return jsonify(users), 200
    except Exception as e:
        logging.error(f"Error retrieving users: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/status', methods=['GET'])
def status():
    conversation_id = request.args.get('conversation_id')
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

@app.route('/feedback', methods=['POST'])
@swag_from({
    'post': {
        'summary': 'Submit Feedback',
        'description': 'Record a 5‑star rating feedback for a specific chat message.',
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message_id': {'type': 'integer'},
                        'rating': {'type': 'integer', 'minimum': 1, 'maximum': 5}
                    },
                    'required': ['message_id', 'rating']
                }
            }
        ],
        'responses': {
            '200': {'description': 'Feedback recorded successfully.'},
            '400': {'description': 'Invalid feedback data.'},
            '500': {'description': 'Internal server error.'}
        }
    }
})
def feedback_route():
    logging.info("Recording feedback.")
    data = request.json
    message_id = data.get('message_id')
    rating = data.get('rating')

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
            c.execute('UPDATE chat_history SET feedback = ? WHERE id = ?', (rating, message_id))
            conn.commit()
        logging.info(f"Feedback recorded for message_id: {message_id} with rating: {rating}")
        return jsonify({"status": "Feedback recorded successfully"}), 200
    except Exception as e:
        logging.error(f"Error recording feedback: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/register', methods=['POST'])
def register_route():
    logging.info("Registering a new user.")
    data = request.json
    user_id = data.get('username')
    password = data.get('password')

    if user_id is None or password is None:
        logging.error("Invalid registration data.")
        return jsonify({"error": "Invalid registration data"}), 400

    try:
        with sqlite3.connect("user_data.db") as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (user_id, password) VALUES (?, ?)', (user_id, password))
            conn.commit()
        logging.info(f"User {user_id} registered successfully.")
        return jsonify({"status": "User registered successfully"}), 200
    except sqlite3.IntegrityError:
        logging.error(f"User {user_id} already exists.")
        return jsonify({"error": "User ID already exists"}), 400
    except Exception as e:
        logging.error(f"Error registering user: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/login', methods=['POST'])
@swag_from({
    'post': {
        'summary': 'User Login',
        'description': 'Login an existing user by verifying username and password.',
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'username': {'type': 'string'},
                        'password': {'type': 'string'}
                    },
                    'required': ['username', 'password']
                }
            }
        ],
        'responses': {
            '200': {'description': 'Login successful.'},
            '400': {'description': 'Invalid login data.'},
            '401': {'description': 'Invalid password.'},
            '404': {'description': 'User not found.'},
            '500': {'description': 'Internal server error.'}
        }
    }
})
def login():
    logging.info("Processing login request.")
    data = request.json
    user_id = data.get('username')
    password = data.get('password')

    if user_id is None or password is None:
        logging.error("Invalid login data.")
        return jsonify({"error": "Invalid login data"}), 400

    try:
        with sqlite3.connect("user_data.db") as conn:
            c = conn.cursor()
            c.execute('SELECT password FROM users WHERE user_id = ?', (user_id,))
            row = c.fetchone()
            if row is None:
                logging.error(f"User {user_id} not found.")
                return jsonify({"error": "User not found"}), 404
            stored_password = row[0]
            if password == stored_password:
                logging.info(f"User {user_id} logged in successfully.")
                return jsonify({"status": "Login successful"}), 200
            else:
                logging.error("Invalid password provided.")
                return jsonify({"error": "Invalid password"}), 401
    except Exception as e:
        logging.error(f"Error logging in user: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/conversations', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Get Conversation Summaries',
        'description': 'Retrieve summaries of conversations for a specific user. The summary includes the conversation ID, first user message, and message count.',
        'parameters': [
            {'name': 'username', 'in': 'query', 'type': 'string', 'required': True, 'description': 'Username whose conversations are to be retrieved.'}
        ],
        'responses': {
            '200': {
                'description': 'Conversation summaries retrieved successfully.',
                'schema': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'conversation_id': {'type': 'string'},
                            'first_message': {'type': 'string'},
                            'message_count': {'type': 'integer'},
                            'first_time': {'type': 'string'}
                        }
                    }
                }
            },
            '400': {'description': 'Username required.'}
        }
    }
})
def conversations():
    logging.info("Fetching conversation summaries.")
    username = request.args.get('username')
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

@app.route('/loadConversation', methods=['GET'])
@swag_from({
    'get': {
        'summary': 'Load Full Conversation',
        'description': 'Load all messages for a given conversation ID.',
        'parameters': [
            {'name': 'conversation_id', 'in': 'query', 'type': 'string', 'required': True, 'description': 'The conversation ID to load messages from.'}
        ],
        'responses': {
            '200': {'description': 'Conversation loaded successfully.'},
            '400': {'description': 'Conversation ID required.'}
        }
    }
})
def loadConversation():
    logging.info("Loading conversation details.")
    conversation_id = request.args.get('conversation_id')
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

# ---------------------
# App Initialization
# ---------------------
if __name__ == '__main__':
    init_db()
    init_user_db()
    #init_pdf_cache_db()
    #preprocess_pdfs_to_db(limit=100)
    app.run(host='0.0.0.0', port=8000, debug=True)
