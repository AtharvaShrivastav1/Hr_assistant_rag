import os
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

test_models = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.0-flash",
    "gemini-pro"
]

for model_name in test_models:
    try:
        print(f"Testing {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, write a 1-word answer.")
        print(f"Success with {model_name}: {response.text.strip()}")
    except Exception as e:
        print(f"Failed with {model_name}: {e}")
