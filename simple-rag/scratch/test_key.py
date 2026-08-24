import os
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
print("API Key exists:", bool(api_key))
if api_key:
    print("API Key Prefix:", api_key[:10])
    
genai.configure(api_key=api_key)
try:
    print("Listing models...")
    for m in genai.list_models():
        print(m.name)
except Exception as e:
    print("Error listing models:", e)
