'''
Created on Jan 9, 2025

@author: memorylab-aj
'''

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pyannote.audio import Pipeline as audiopipeline
import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")  # Load Hugging Face token from environment variable

# Function to verify token validity
def is_token_valid(token):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
        print("Token verification response:", response.status_code, response.json())
        return response.status_code == 200
    except Exception:
        return False

# List of multilingual models to download
multilingual_models = [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-tiny",
    "Berly00/whisper-large-v3-spanish"    
]

# Download the pyannote model for speaker diarization only if token is available and valid
if token and is_token_valid(token):
    try:
        print("Token is valid. Downloading pyannote speaker diarization model...")
        audiopipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        print("Successfully downloaded pyannote model")
    except Exception as e:
        print(f"Failed to download pyannote model. Error: {e}")
else:
    if not token:
        print("HF_TOKEN environment variable not set. Skipping pyannote model download.")
    else:
        print("Invalid HF_TOKEN. Skipping pyannote model download.")

# Download each model and its processor
for model_id in multilingual_models:
    print(f"Downloading model and processor for: {model_id}")
    AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    AutoProcessor.from_pretrained(model_id)