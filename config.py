from dotenv import load_dotenv
import os

# load variables from .env
load_dotenv()

# API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# LLM settings
MODEL_NAME = "openai/gpt-4o-mini"

# OpenRouter endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# PubMed settings
DEFAULT_MAX_PAPERS = 30
DEFAULT_MAX_RETRIES = 3

RERANK_TOP_K = 10