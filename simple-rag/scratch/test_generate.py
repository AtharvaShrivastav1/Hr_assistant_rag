import os
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

for model_name in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]:
    try:
        print(f"Testing {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, write a 1-word answer.")
        print(f"Success with {model_name}: {response.text.strip()}")
    except Exception as e:
        print(f"Failed with {model_name}: {e}")
