from flask import Flask, request, jsonify
from flask_cors import CORS
from rapidfuzz import process, fuzz
from deep_translator import GoogleTranslator
import threading
import requests
import os
import logging
import time
from textblob import TextBlob
from openai import OpenAI

# ========== CONFIGURATION ==========
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

institution_name = "Ecobank Rwanda"
DIALOGUE_FILE = "custom_dialogues.txt"
client = OpenAI(api_key="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")

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
        "open ecobank account", "open express account"
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

# ========== DIALOGUE DATA ==========
dialogue_lock = threading.Lock()
def load_custom_dialogues(filename):
    dialogues = {}
    if not os.path.exists(filename):
        return dialogues
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" || ")
            if len(parts) == 2:
                dialogues[parts[0].strip().lower()] = parts[1].strip()
    return dialogues

def save_custom_dialogue(user_input, response):
    with dialogue_lock:
        with open(DIALOGUE_FILE, "a", encoding="utf-8") as file:
            file.write(f"{user_input} || {response}\n")

dialogues = load_custom_dialogues(DIALOGUE_FILE)

# ========== INTENT & RELEVANCE ==========
def detect_intent(user_input):
    user_input = user_input.lower()
    for intent, phrases in INTENTS.items():
        for phrase in phrases:
            if phrase in user_input:
                return intent
    return "unknown"

def is_question_relevant(user_input):
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
    return any(kw in user_input.lower() for kw in keywords)

# ========== CUSTOM RESPONSE ==========
def get_custom_response(user_input, dialogues):
    user_input = user_input.strip().lower()
    questions = list(dialogues.keys())
    result = process.extractOne(user_input, questions, scorer=fuzz.ratio)
    if not result:
        return None, 0
    best_match, score, _ = result
    if score >= 70:
        # Save new question if not already present
        if user_input not in dialogues:
            save_custom_dialogue(user_input, dialogues[best_match])
            dialogues[user_input] = dialogues[best_match]
        return dialogues[best_match], score
    return None, score

# ========== OPENAI RESPONSE ==========
def get_openai_response(user_input):
    context = f"You are a helpful assistant specializing in answering questions about {institution_name}."
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,
            temperature=0.2,
        )
        text_response = response.choices[0].message.content.strip()
        save_custom_dialogue(user_input, text_response)
        dialogues[user_input] = text_response
        return text_response
    except Exception as e:
        logging.error(f"OpenAI API exception: {e}")
        return "Sorry, I couldn't get an answer at the moment."

# ========== MAIN CHATBOT LOGIC ==========
def chatbot_response(user_input, dialogues):
    response, score = get_custom_response(user_input, dialogues)
    if response:
        return response, score  # Return both response and confidence score
    if not is_question_relevant(user_input):
        return ("Sorry, your question doesn't seem related to Ecobank or banking services. Please ask about Ecobank products, services, or operations.", 100)
    openai_response = get_openai_response(user_input)
    return (openai_response, None)  # No confidence for OpenAI

# ========== FLASK ROUTES ==========
@app.route("/chat", methods=["POST"])
def chat():
    start_time = time.time()  # Start timing

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

    # Get chatbot response and confidence
    bot_response, confidence = chatbot_response(user_input_en, dialogues)

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

    response_time = round(time.time() - start_time, 3)  # End timing

    # Final response
    return jsonify({
        "user_input": user_input_original,
        "language": language,
        "bot_response": bot_response_translated,
        "intent": intent,
        "confidence": confidence_str,
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "response_time": f"{response_time} seconds"
    })

# ========== RUN APP ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)