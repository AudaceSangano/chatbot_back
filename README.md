# Bank of Kigali Chatbot

This is a chatbot project designed to provide information about Bank of Kigali's services, including account opening, customer care, and more. The chatbot uses Flask as the backend framework and supports multiple languages (English, French, and Kinyarwanda).

---

## Features

- Provides responses based on a custom dataset (`custom_dialogues.txt`).
- Supports fuzzy matching for user queries.
- Falls back to the Gemini API for responses not found in the dataset.
- Translates user input and responses between English, French, and Kinyarwanda.
- RESTful API with a `/chat` endpoint.

---

## Prerequisites

- Python 3.8 or later
- `pip` (Python package manager)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/chatbot.git
cd chatbot

python3 -m venv chatbot-env
source chatbot-env/bin/activate  # On Linux/Mac
# chatbot-env\Scripts\activate  # On Windows

pip install -r requirements.txt

pip install Flask flask-cors requests rapidfuzz deep-translator
```

---

## Example Request

```json
{
  "user_input": "How can I open an account?",
  "language": "en",
  "bot_response": "You can open an account online at https://www.bk.rw or visit any of our branches with your ID and proof of address."
}
```
