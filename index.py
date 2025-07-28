from flask import Flask, request, jsonify
import threading
import requests
from rapidfuzz import process, fuzz
from deep_translator import GoogleTranslator
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

institution_name = "Ecobank Rwanda"

# Load dataset into a dictionary
def load_custom_dialogues(filename):
    dialogues = {}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" || ")  # Using " || " as separator
            if len(parts) == 2:
                dialogues[parts[0].strip().lower()] = parts[1].strip()
    return dialogues

# Find answer in custom dataset using fuzzy matching
def get_custom_response(user_input, dialogues):
    user_input = user_input.strip().lower()
    
    # Use fuzzy matching to find the closest match
    questions = list(dialogues.keys())
    best_match, score, _ = process.extractOne(user_input, questions, scorer=fuzz.ratio)  # Ignore the index
    
    # If the score is between 70 and 98, add the user input to custom_dialogues.txt if not already present
    if 70 <= score < 98:
        # Check if the key already exists in the file
        with open("custom_dialogues.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
        
        existing_keys = {line.split(" || ")[0].strip() for line in lines if " || " in line}
        if user_input not in existing_keys:  # Only add if the key is not already in the file
            with open("custom_dialogues.txt", "a", encoding="utf-8") as file:
                file.write(f"{user_input} || {dialogues[best_match]}\n")
        
        # Also update the in-memory dictionary
        dialogues[user_input] = dialogues[best_match]
        return dialogues[best_match]
    
    # If the score is 90 or above, return the best match
    if score >= 99:
        return dialogues[best_match]
    
    # If no sufficiently close match is found, return None
    return None

# Get response from Gemini API if no custom response is found
def get_gemini_response(user_input):
    # Add context to guide the Gemini API
    context = f"You are a helpful assistant specializing in answering questions about {institution_name}. "
    context_end = ", summarize and use single sentence. "
    prompt = context + user_input + context_end  # Combine context with user input

    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyCkq_F2wYgEjxf4A1jEK4y_Fu138exW9t0"
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(api_url, json=data)

    if response.status_code == 200:
        # Extract the response text
        response_json = response.json()
        text_response = response_json['candidates'][0]['content']['parts'][0]['text']

        # Check if the key already exists in the file
        with open("custom_dialogues.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
        
        existing_keys = {line.split(" || ")[0].strip() for line in lines if " || " in line}
        if user_input not in existing_keys:  # Only add if the key is not already in the file
            with open("custom_dialogues.txt", "a", encoding="utf-8") as file:
                file.write(f"{user_input} || {text_response}\n")
        
        return text_response
    else:
        return f"Error: {response.status_code}, {response.text}"

# Main function to get chatbot response
def chatbot_response(user_input, dialogues):
    # Search for a response in the custom dialogues
    response = get_custom_response(user_input, dialogues)

    if response:
        return response  # Return from dataset if found
    else:
        return get_gemini_response(user_input)  # Otherwise, use Gemini API

# Load dataset
dialogues = load_custom_dialogues("custom_dialogues.txt")

# Flask API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input_original = data.get("user_input", "").strip()
    user_input = data.get("user_input", "").strip()
    language = data.get("language", "").strip().lower()  # Get the language field

    if not user_input:
        return jsonify({"error": "No user input provided"}), 400

    # Validate the language field
    if language not in ["en", "fr", "rw"]:
        return jsonify({"error": "Unsupported language. Please use 'en' for English, 'fr' for French, or 'rw' for Kinyarwanda."}), 400

    # Translate input to English if the language is Kinyarwanda
    if language == "rw":
        try:
            user_input = GoogleTranslator(source="auto", target="en").translate(user_input)
        except Exception as e:
            return jsonify({"error": f"Translation failed: {str(e)}"}), 500
        
    # Translate input to English if the language is English
    if language == "en":
        try:
            user_input = GoogleTranslator(source="auto", target="en").translate(user_input)
        except Exception as e:
            return jsonify({"error": f"Translation failed: {str(e)}"}), 500
        
    # Translate input to English if the language is Français
    if language == "fr":
        try:
            user_input = GoogleTranslator(source="auto", target="en").translate(user_input)
        except Exception as e:
            return jsonify({"error": f"Translation failed: {str(e)}"}), 500
   
    # Get chatbot response
    bot_response = chatbot_response(user_input, dialogues)

    # Translate bot response back to Kinyarwanda if the input language is "rw"
    if language == "rw":
        try:
            bot_response = GoogleTranslator(source="auto", target="rw").translate(bot_response)  # Use 'rw' for Kinyarwanda
        except Exception as e:
            return jsonify({"error": f"Translation of bot response failed: {str(e)}"}), 500

    # Translate bot response back to English if the input language is "en"
    if language == "en":
        try:
            bot_response = GoogleTranslator(source="auto", target="en").translate(bot_response)  # Use 'rw' for Kinyarwanda
        except Exception as e:
            return jsonify({"error": f"Translation of bot response failed: {str(e)}"}), 500

    # Translate bot response back to Kinyarwanda if the input language is "rw"
    if language == "fr":
        try:
            bot_response = GoogleTranslator(source="auto", target="fr").translate(bot_response)  # Use 'rw' for Kinyarwanda
        except Exception as e:
            return jsonify({"error": f"Translation of bot response failed: {str(e)}"}), 500

    return jsonify({"user_input": user_input_original, "language": language, "bot_response": bot_response})

# Run Flask app in a separate thread
def run_flask():
    app.run(host="0.0.0.0", port=5000)

# Start Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# Keep the main thread alive
try:
    while True:
        pass  # Keep the main thread running
except KeyboardInterrupt:
    print("Shutting down...")