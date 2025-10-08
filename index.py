from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import process, fuzz
from deep_translator import GoogleTranslator
import threading
import requests
import os
import logging
import time
from datetime import datetime, timedelta
from textblob import TextBlob
from openai import OpenAI
import json
import hashlib
import secrets
import jwt

# ========== CONFIGURATION ==========
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

institution_name = "Ecobank Rwanda"
DIALOGUE_FILE = "custom_dialogues.txt"
METADATA_FILE = "dialogue_metadata.json"
USERS_FILE = "users.json"
SECRET_KEY = "your-secret-key-change-this-in-production"  # Change this!
TOKEN_EXPIRY_HOURS = 24

client = OpenAI(api_key="")

INTENTS = {
    "greeting": [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening"
    ],
    "goodbye": [
        "bye", "goodbye", "see you", "farewell", "talk later"
    ],
    "thanks": [
        "thanks", "thank you", "thx", "appreciate", "much thanks"
    ],
    "account_opening": [
        "open account", "create account", "start account", "how to open account",
        "open ecobank account", "open express account", "documents for account opening",
        "requirements for account", "needed to open account"
    ],
    "account_closure": [
        "close account", "delete account", "terminate account", "shut down account"
    ],
    "balance_inquiry": [
        "check balance", "what is my balance", "account balance", "current balance",
        "how much money", "balance info"
    ],
    "loan_inquiry": [
        "apply for loan", "loan application", "get a loan", "loan status", "loan repayment",
        "ecobank loan", "salary loan", "loan rates"
    ],
    "transaction_history": [
        "transaction history", "recent transactions", "account activity",
        "mini statement", "print statement"
    ],
    "money_transfer": [
        "send money", "transfer funds", "transfer money", "make payment",
        "ecobank to mtn", "send to bank", "pay someone", "momo transfer", "airtel money"
    ],
    "ecobank_xpress": [
        "xpress account", "express account", "open xpress", "xpress loan", "xpress transfer"
    ],
    "ecobank_mobile": [
        "ecobank app", "mobile app", "ecobank mobile", "ecobank ussd", "*790#", "how to use ecobank app"
    ],
    "card_services": [
        "lost card", "block card", "card replacement", "card delivery", "activate card", 
        "debit card", "credit card", "ecobank visa", "ecobank mastercard"
    ],
    "currency_exchange": [
        "exchange rate", "forex", "convert currency", "usd to rwf", "currency rate today"
    ],
    "atm_locator": [
        "nearest atm", "atm near me", "ecobank atm", "find atm", "withdraw money"
    ],
    "branch_locator": [
        "nearest branch", "ecobank branch", "branch near me", "where is ecobank"
    ],
    "bill_payment": [
        "pay bill", "pay electricity", "pay water", "pay tv", "pay internet", "pay utility"
    ],
    "complaint": [
        "not working", "problem", "issue", "complain", "error", "failed transaction"
    ],
    "limit_inquiry": [
        "transfer limit", "daily limit", "limit for momo", "limit for transfer"
    ],
    "pin_reset": [
        "reset pin", "forgot pin", "change pin", "atm pin", "card pin"
    ],
    "general_question": [
        "what", "how", "when", "where", "why", "can I", "do you"
    ]
}

# ========== DIALOGUE DATA WITH METADATA ==========
dialogue_lock = threading.Lock()
metadata_lock = threading.Lock()
users_lock = threading.Lock()
active_sessions = {}  # Store active user sessions in memory

def load_custom_dialogues(filename):
    """Load dialogues from file"""
    dialogues = {}
    if not os.path.exists(filename):
        return dialogues
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" || ")
            if len(parts) == 2:
                dialogues[parts[0].strip().lower()] = parts[1].strip()
    return dialogues

def load_metadata(filename):
    """Load metadata (timestamp and count) from JSON file"""
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        logging.error(f"Error loading {filename}, returning empty metadata")
        return {}

def save_metadata(metadata, filename):
    """Save metadata to JSON file"""
    with metadata_lock:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

# ========== USER MANAGEMENT ==========
def load_users(filename):
    """Load users from JSON file"""
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        logging.error(f"Error loading {filename}, returning empty users")
        return {}

def save_users(users, filename):
    """Save users to JSON file"""
    with users_lock:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(users, file, indent=2, ensure_ascii=False)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token(user_id, role):
    """Generate JWT token for user"""
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token):
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorator to require authentication"""
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return jsonify({"error": "No authorization header"}), 401
        
        try:
            token = auth_header.split(" ")[1]  # Bearer <token>
        except IndexError:
            return jsonify({"error": "Invalid authorization header format"}), 401
        
        payload = verify_token(token)
        
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        # Add user info to request
        request.user_id = payload["user_id"]
        request.user_role = payload["role"]
        
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function

def require_admin(f):
    """Decorator to require admin role"""
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return jsonify({"error": "No authorization header"}), 401
        
        try:
            token = auth_header.split(" ")[1]
        except IndexError:
            return jsonify({"error": "Invalid authorization header format"}), 401
        
        payload = verify_token(token)
        
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        if payload["role"] != "admin":
            return jsonify({"error": "Admin access required"}), 403
        
        request.user_id = payload["user_id"]
        request.user_role = payload["role"]
        
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function

# Load users on startup
users = load_users(USERS_FILE)

# Create default admin if no users exist
if not users:
    admin_id = secrets.token_hex(16)
    users[admin_id] = {
        "username": "admin",
        "email": "admin@ecobank.rw",
        "password": hash_password("admin123"),
        "role": "admin",
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
    save_users(users, USERS_FILE)
    logging.info("Default admin created - username: admin, password: admin123")

def save_custom_dialogue(user_input, response):
    """Save new dialogue to file"""
    with dialogue_lock:
        with open(DIALOGUE_FILE, "a", encoding="utf-8") as file:
            file.write(f"{user_input} || {response}\n")

def add_or_update_metadata(user_input, metadata, intent=None, response_time=None, language=None):
    """Add new metadata or update count for existing question"""
    user_input_lower = user_input.lower()
    
    if user_input_lower not in metadata:
        # New question - add with timestamp and count = 1
        metadata[user_input_lower] = {
            "count": 1,
            "created_at": datetime.now().isoformat(),
            "last_asked": datetime.now().isoformat(),
            "intents": {},
            "languages": {},
            "avg_response_time": response_time if response_time else 0,
            "total_response_time": response_time if response_time else 0
        }
    else:
        # Existing question - increment count and update last_asked
        metadata[user_input_lower]["count"] += 1
        metadata[user_input_lower]["last_asked"] = datetime.now().isoformat()
        
        # Update response time average
        if response_time:
            total = metadata[user_input_lower].get("total_response_time", 0)
            count = metadata[user_input_lower]["count"]
            metadata[user_input_lower]["total_response_time"] = total + response_time
            metadata[user_input_lower]["avg_response_time"] = round(
                metadata[user_input_lower]["total_response_time"] / count, 3
            )
    
    # Track intent frequency
    if intent:
        if "intents" not in metadata[user_input_lower]:
            metadata[user_input_lower]["intents"] = {}
        metadata[user_input_lower]["intents"][intent] = metadata[user_input_lower]["intents"].get(intent, 0) + 1
    
    # Track language frequency
    if language:
        if "languages" not in metadata[user_input_lower]:
            metadata[user_input_lower]["languages"] = {}
        metadata[user_input_lower]["languages"][language] = metadata[user_input_lower]["languages"].get(language, 0) + 1
    
    save_metadata(metadata, METADATA_FILE)
    return metadata[user_input_lower]

# Load dialogues and metadata on startup
dialogues = load_custom_dialogues(DIALOGUE_FILE)
dialogue_metadata = load_metadata(METADATA_FILE)

# Initialize metadata for existing dialogues that don't have it
for question in dialogues.keys():
    if question not in dialogue_metadata:
        dialogue_metadata[question] = {
            "count": 0,
            "created_at": datetime.now().isoformat(),
            "last_asked": None,
            "intents": {},
            "languages": {},
            "avg_response_time": 0,
            "total_response_time": 0
        }
save_metadata(dialogue_metadata, METADATA_FILE)

# ========== INTENT & RELEVANCE ==========
def detect_intent(user_input):
    """Fuzzy intent detection"""
    user_input = user_input.lower()
    best_intent = "unknown"
    best_score = 0

    for intent, phrases in INTENTS.items():
        result = process.extractOne(user_input, phrases, scorer=fuzz.partial_ratio)
        if result:
            _, score, _ = result
            if score > best_score:
                best_intent = intent
                best_score = score

    if best_score >= 80:  # threshold to accept match
        return best_intent
    return "unknown"

def is_question_relevant(user_input):
    """Fuzzy relevance detection for banking-related questions"""
    keywords = [
        "ecobank", "bank", "account", "balance", "statement", "transaction", "transfer",
        "deposit", "withdraw", "atm", "loan", "credit", "debit", "card", "mobile banking",
        "internet banking", "branch", "swift", "sort code", "checkbook", "overdraft", "interest",
        "charges", "fees", "pin", "otp", "currency", "exchange", "remittance", "send money",
        "receive money", "open account", "close account", "update info", "kyc", "customer care",
        "support", "limit", "blocked", "unblock", "fraud", "scam", "security", "lost card", "reset pin",
        "ecobank xpress", "ecobank pay", "ecobank app", "ussd", "*326#", "swift code", "iban", "ecobank branch",
        "services", "products", "fees", "charges", "support", "help", "contact"
    ]
    result = process.extractOne(user_input.lower(), keywords, scorer=fuzz.partial_ratio)
    return result and result[1] >= 65

# ========== CUSTOM RESPONSE ==========
def get_custom_response(user_input, dialogues, metadata, intent=None, response_time=None, language=None):
    """Get custom response and update metadata"""
    user_input = user_input.strip().lower()
    questions = list(dialogues.keys())
    result = process.extractOne(user_input, questions, scorer=fuzz.ratio)
    
    if not result:
        return None, 0, None
    
    best_match, score, _ = result
    
    if score >= 70:
        # Save new question variant if not already present
        if user_input not in dialogues:
            save_custom_dialogue(user_input, dialogues[best_match])
            dialogues[user_input] = dialogues[best_match]
        
        # Return the matched question for metadata update later
        return dialogues[best_match], score, best_match
    
    return None, score, None

# ========== OPENAI RESPONSE ==========
def get_openai_response(user_input, metadata, intent=None, response_time=None, language=None):
    """Get OpenAI response and save with metadata"""
    context = f"You are a helpful assistant specializing in answering questions about {institution_name}. Remember: if the question is not related to banking, politely decline to answer."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_input}
            ],
            max_tokens=200,
            temperature=0.2,
        )
        text_response = response.choices[0].message.content.strip()
        
        # Save to dialogue file
        save_custom_dialogue(user_input, text_response)
        dialogues[user_input] = text_response
        
        # Return matched question key for metadata update later
        return text_response, user_input.lower()
    except Exception as e:
        logging.error(f"OpenAI API exception: {e}")
        return "Sorry, I couldn't get an answer at the moment.", None

# ========== MAIN CHATBOT LOGIC ==========
def chatbot_response(user_input, dialogues, metadata, intent=None, response_time=None, language=None):
    """Main chatbot logic with metadata tracking"""
    response, score, matched_question = get_custom_response(user_input, dialogues, metadata, intent, response_time, language)
    
    if response:
        return response, score, matched_question
    
    if not is_question_relevant(user_input):
        return ("Sorry, your question doesn't seem related to Ecobank or banking services. Please ask about Ecobank products, services, or operations.", 100, None)
    
    openai_response, matched_question = get_openai_response(user_input, metadata, intent, response_time, language)
    return openai_response, None, matched_question

# ========== FLASK ROUTES ==========

# ========== AUTHENTICATION ROUTES ==========
@app.route("/register", methods=["POST"])
def register():
    """Register a new user"""
    data = request.json
    
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    role = data.get("role", "user").strip()  # Default to 'user'
    
    # Validation
    if not username or not email or not password:
        return jsonify({"error": "Username, email, and password are required"}), 400
    
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    
    if role not in ["user", "admin"]:
        return jsonify({"error": "Role must be 'user' or 'admin'"}), 400
    
    # Check if username or email already exists
    for user_data in users.values():
        if user_data["username"].lower() == username.lower():
            return jsonify({"error": "Username already exists"}), 409
        if user_data["email"].lower() == email.lower():
            return jsonify({"error": "Email already exists"}), 409
    
    # Create new user
    user_id = secrets.token_hex(16)
    users[user_id] = {
        "username": username,
        "email": email,
        "password": hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
    
    save_users(users, USERS_FILE)
    
    # Generate token
    token = generate_token(user_id, role)
    
    return jsonify({
        "success": True,
        "message": "User registered successfully",
        "user": {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role
        },
        "token": token,
        "expires_in": f"{TOKEN_EXPIRY_HOURS} hours"
    }), 201

@app.route("/login", methods=["POST"])
def login():
    """Login user"""
    data = request.json
    
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    # Find user by username or email
    user_id = None
    user_data = None
    
    for uid, udata in users.items():
        if udata["username"].lower() == username.lower() or udata["email"].lower() == username.lower():
            user_id = uid
            user_data = udata
            break
    
    if not user_data:
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Verify password
    if user_data["password"] != hash_password(password):
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Update last login
    users[user_id]["last_login"] = datetime.now().isoformat()
    save_users(users, USERS_FILE)
    
    # Generate token
    token = generate_token(user_id, user_data["role"])
    
    # Store session
    active_sessions[user_id] = {
        "token": token,
        "login_time": datetime.now().isoformat()
    }
    
    return jsonify({
        "success": True,
        "message": "Login successful",
        "user": {
            "user_id": user_id,
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"]
        },
        "token": token,
        "expires_in": f"{TOKEN_EXPIRY_HOURS} hours"
    })

@app.route("/logout", methods=["POST"])
@require_auth
def logout():
    """Logout user"""
    user_id = request.user_id
    
    # Remove from active sessions
    if user_id in active_sessions:
        del active_sessions[user_id]
    
    return jsonify({
        "success": True,
        "message": "Logged out successfully"
    })

@app.route("/me", methods=["GET"])
@require_auth
def get_current_user():
    """Get current user info"""
    user_id = request.user_id
    
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    
    user_data = users[user_id]
    
    return jsonify({
        "user_id": user_id,
        "username": user_data["username"],
        "email": user_data["email"],
        "role": user_data["role"],
        "created_at": user_data["created_at"],
        "last_login": user_data["last_login"]
    })

@app.route("/users", methods=["GET"])
@require_admin
def get_all_users():
    """Get all users (admin only)"""
    users_list = []
    
    for user_id, user_data in users.items():
        users_list.append({
            "user_id": user_id,
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"],
            "created_at": user_data["created_at"],
            "last_login": user_data["last_login"]
        })
    
    return jsonify({
        "total": len(users_list),
        "users": users_list
    })

@app.route("/users/<user_id>", methods=["DELETE"])
@require_admin
def delete_user(user_id):
    """Delete a user (admin only)"""
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    
    # Don't allow deleting yourself
    if user_id == request.user_id:
        return jsonify({"error": "Cannot delete your own account"}), 400
    
    deleted_user = users[user_id]
    del users[user_id]
    save_users(users, USERS_FILE)
    
    # Remove from active sessions
    if user_id in active_sessions:
        del active_sessions[user_id]
    
    return jsonify({
        "success": True,
        "message": "User deleted successfully",
        "deleted_user": {
            "username": deleted_user["username"],
            "email": deleted_user["email"]
        }
    })

@app.route("/change-password", methods=["POST"])
@require_auth
def change_password():
    """Change user password"""
    data = request.json
    user_id = request.user_id
    
    old_password = data.get("old_password", "").strip()
    new_password = data.get("new_password", "").strip()
    
    if not old_password or not new_password:
        return jsonify({"error": "Old and new passwords are required"}), 400
    
    if len(new_password) < 6:
        return jsonify({"error": "New password must be at least 6 characters"}), 400
    
    # Verify old password
    if users[user_id]["password"] != hash_password(old_password):
        return jsonify({"error": "Old password is incorrect"}), 401
    
    # Update password
    users[user_id]["password"] = hash_password(new_password)
    save_users(users, USERS_FILE)
    
    return jsonify({
        "success": True,
        "message": "Password changed successfully"
    })

# ========== CHATBOT ROUTES ==========
@app.route("/chat", methods=["POST"])
def chat():
    start_time = time.time()

    data = request.json
    user_input_original = data.get("user_input", "").strip()
    user_input = user_input_original
    language = data.get("language", "en").strip().lower()

    if not user_input:
        return jsonify({"error": "No user input provided"}), 400
    if language not in ["en", "fr", "rw"]:
        return jsonify({"error": "Unsupported language. Please use 'en', 'fr', or 'rw'."}), 400

    # Translate to English for processing
    try:
        user_input_en = GoogleTranslator(source="auto", target="en").translate(user_input)
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

    # Sentiment analysis
    blob = TextBlob(user_input_en)
    sentiment_score = round(blob.sentiment.polarity, 3)
    if sentiment_score > 0.2:
        sentiment_label = "positive"
    elif sentiment_score < -0.2:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # Intent detection
    intent = detect_intent(user_input_en)

    # Get chatbot response (returns matched question key instead of metadata)
    bot_response, confidence, matched_question_key = chatbot_response(
        user_input_en, 
        dialogues, 
        dialogue_metadata, 
        intent=intent, 
        response_time=None,
        language=language
    )

    # Translate response back to user language
    try:
        bot_response_translated = GoogleTranslator(source="auto", target=language).translate(bot_response)
    except Exception as e:
        return jsonify({"error": f"Response translation failed: {str(e)}"}), 500

    # Format confidence
    if confidence is not None:
        confidence_str = f"{round(confidence, 2)}%"
    else:
        confidence_str = "N/A"

    response_time = round(time.time() - start_time, 3)
    
    # Update metadata ONCE with actual response time (only if we have a matched question)
    question_metadata = None
    if matched_question_key:
        question_metadata = add_or_update_metadata(
            matched_question_key, 
            dialogue_metadata, 
            intent=intent, 
            response_time=response_time, 
            language=language
        )
        # Reload metadata to get the updated values
        question_metadata = dialogue_metadata.get(matched_question_key.lower())

    # Build response with metadata
    response_data = {
        "user_input": user_input_original,
        "language": language,
        "bot_response": bot_response_translated,
        "intent": intent,
        "confidence": confidence_str,
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "response_time": f"{response_time} seconds"
    }

    # Add metadata if available
    if question_metadata and isinstance(question_metadata, dict):
        # Get most common intent
        intents_data = question_metadata.get("intents", {})
        most_common_intent = max(intents_data.items(), key=lambda x: x[1])[0] if intents_data else "unknown"
        
        # Get most common language
        languages_data = question_metadata.get("languages", {})
        most_common_language = max(languages_data.items(), key=lambda x: x[1])[0] if languages_data else "en"
        
        response_data["metadata"] = {
            "times_asked": question_metadata["count"],
            "created_at": question_metadata["created_at"],
            "last_asked": question_metadata["last_asked"],
            "avg_response_time": f"{question_metadata.get('avg_response_time', 0)} seconds",
            "intents": intents_data,
            "most_common_intent": most_common_intent,
            "languages": languages_data,
            "most_common_language": most_common_language
        }

    return jsonify(response_data)

@app.route("/analytics", methods=["GET"])
def analytics():
    """Get analytics about dialogue usage"""
    # Sort by count (most asked questions)
    sorted_metadata = sorted(
        dialogue_metadata.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    analytics_data = {
        "total_questions": len(dialogue_metadata),
        "total_interactions": sum(meta["count"] for meta in dialogue_metadata.values()),
        "top_questions": []
    }
    
    # Get top 10 most asked questions
    for question, meta in sorted_metadata[:10]:
        if question in dialogues:
            # Get most common intent and language
            intents_data = meta.get("intents", {})
            most_common_intent = max(intents_data.items(), key=lambda x: x[1])[0] if intents_data else "unknown"
            
            languages_data = meta.get("languages", {})
            most_common_language = max(languages_data.items(), key=lambda x: x[1])[0] if languages_data else "en"
            
            analytics_data["top_questions"].append({
                "question": question,
                "answer": dialogues[question],
                "times_asked": meta["count"],
                "created_at": meta["created_at"],
                "last_asked": meta["last_asked"],
                "avg_response_time": f"{meta.get('avg_response_time', 0)} seconds",
                "intents": intents_data,
                "most_common_intent": most_common_intent,
                "languages": languages_data,
                "most_common_language": most_common_language
            })
    
    return jsonify(analytics_data)

@app.route("/question/<path:question>", methods=["GET"])
def get_question_info(question):
    """Get detailed info about a specific question"""
    question_lower = question.lower()
    
    if question_lower not in dialogue_metadata:
        return jsonify({"error": "Question not found"}), 404
    
    meta = dialogue_metadata[question_lower]
    
    # Get most common intent and language
    intents_data = meta.get("intents", {})
    most_common_intent = max(intents_data.items(), key=lambda x: x[1])[0] if intents_data else "unknown"
    
    languages_data = meta.get("languages", {})
    most_common_language = max(languages_data.items(), key=lambda x: x[1])[0] if languages_data else "en"
    
    return jsonify({
        "question": question,
        "answer": dialogues.get(question_lower, "No answer found"),
        "times_asked": meta["count"],
        "created_at": meta["created_at"],
        "last_asked": meta["last_asked"],
        "avg_response_time": f"{meta.get('avg_response_time', 0)} seconds",
        "intents": intents_data,
        "most_common_intent": most_common_intent,
        "languages": languages_data,
        "most_common_language": most_common_language
    })

@app.route("/train", methods=["POST"])
@require_admin
def train():
    """Train the chatbot with new question-answer pairs"""
    data = request.json
    
    # Validate input
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Support single training entry
    if "question" in data and "answer" in data:
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        category = data.get("category", "general")
        
        if not question or not answer:
            return jsonify({"error": "Both question and answer are required"}), 400
        
        question_lower = question.lower()
        
        # Check if question already exists
        if question_lower in dialogues:
            return jsonify({
                "error": "Question already exists",
                "existing_answer": dialogues[question_lower]
            }), 409
        
        # Save to dialogue file
        save_custom_dialogue(question_lower, answer)
        dialogues[question_lower] = answer
        
        # Initialize metadata
        dialogue_metadata[question_lower] = {
            "count": 0,
            "created_at": datetime.now().isoformat(),
            "last_asked": None,
            "intents": {},
            "languages": {},
            "avg_response_time": 0,
            "total_response_time": 0,
            "category": category
        }
        save_metadata(dialogue_metadata, METADATA_FILE)
        
        return jsonify({
            "success": True,
            "message": "Training data added successfully",
            "question": question,
            "answer": answer,
            "category": category
        }), 201
    
    # Support batch training
    elif "training_data" in data:
        training_data = data.get("training_data", [])
        
        if not isinstance(training_data, list):
            return jsonify({"error": "training_data must be a list"}), 400
        
        added = []
        skipped = []
        errors = []
        
        for idx, entry in enumerate(training_data):
            if not isinstance(entry, dict):
                errors.append({"index": idx, "error": "Entry must be a dictionary"})
                continue
            
            question = entry.get("question", "").strip()
            answer = entry.get("answer", "").strip()
            category = entry.get("category", "general")
            
            if not question or not answer:
                errors.append({"index": idx, "error": "Missing question or answer"})
                continue
            
            question_lower = question.lower()
            
            # Skip if already exists
            if question_lower in dialogues:
                skipped.append({
                    "question": question,
                    "reason": "Already exists"
                })
                continue
            
            # Save to dialogue file
            save_custom_dialogue(question_lower, answer)
            dialogues[question_lower] = answer
            
            # Initialize metadata
            dialogue_metadata[question_lower] = {
                "count": 0,
                "created_at": datetime.now().isoformat(),
                "last_asked": None,
                "intents": {},
                "languages": {},
                "avg_response_time": 0,
                "total_response_time": 0,
                "category": category
            }
            
            added.append({
                "question": question,
                "answer": answer,
                "category": category
            })
        
        save_metadata(dialogue_metadata, METADATA_FILE)
        
        return jsonify({
            "success": True,
            "message": f"Batch training completed",
            "summary": {
                "total": len(training_data),
                "added": len(added),
                "skipped": len(skipped),
                "errors": len(errors)
            },
            "added": added,
            "skipped": skipped,
            "errors": errors
        }), 201
    
    else:
        return jsonify({
            "error": "Invalid request format. Provide either 'question' and 'answer' or 'training_data' list"
        }), 400

@app.route("/train/<path:question>", methods=["PUT"])
@require_admin
def update_training(question):
    """Update an existing training entry"""
    question_lower = question.lower()
    
    if question_lower not in dialogues:
        return jsonify({"error": "Question not found"}), 404
    
    data = request.json
    new_answer = data.get("answer", "").strip()
    
    if not new_answer:
        return jsonify({"error": "Answer is required"}), 400
    
    old_answer = dialogues[question_lower]
    
    # Update in memory
    dialogues[question_lower] = new_answer
    
    # Update in file - rewrite entire file
    with dialogue_lock:
        with open(DIALOGUE_FILE, "w", encoding="utf-8") as file:
            for q, a in dialogues.items():
                file.write(f"{q} || {a}\n")
    
    return jsonify({
        "success": True,
        "message": "Training data updated successfully",
        "question": question,
        "old_answer": old_answer,
        "new_answer": new_answer
    })

@app.route("/train/<path:question>", methods=["DELETE"])
def delete_training(question):
    """Delete a training entry"""
    question_lower = question.lower()
    
    if question_lower not in dialogues:
        return jsonify({"error": "Question not found"}), 404
    
    deleted_answer = dialogues[question_lower]
    
    # Remove from dialogues
    del dialogues[question_lower]
    
    # Remove from metadata
    if question_lower in dialogue_metadata:
        del dialogue_metadata[question_lower]
        save_metadata(dialogue_metadata, METADATA_FILE)
    
    # Rewrite dialogue file
    with dialogue_lock:
        with open(DIALOGUE_FILE, "w", encoding="utf-8") as file:
            for q, a in dialogues.items():
                file.write(f"{q} || {a}\n")
    
    return jsonify({
        "success": True,
        "message": "Training data deleted successfully",
        "question": question,
        "deleted_answer": deleted_answer
    })

# ========== RUN APP ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)