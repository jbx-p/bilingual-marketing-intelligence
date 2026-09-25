import pandas as pd
import torch
from transformers import pipeline
from google_play_scraper import app
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()
print("Pandas OK:", pd.__version__)
print("Torch OK:", torch.__version__)
print("API key loaded:", bool(os.getenv("ANTHROPIC_API_KEY")))
print("All imports successful.")
