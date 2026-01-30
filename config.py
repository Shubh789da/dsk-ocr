"""
Configuration for DeepSeek-OCR-2 RunPod Serverless Endpoint
"""

import os
from transformers import AutoTokenizer

# Model configuration
BASE_SIZE = int(os.getenv('BASE_SIZE', '1024'))
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '768'))
CROP_MODE = os.getenv('CROP_MODE', 'True').lower() == 'true'
MIN_CROPS = int(os.getenv('MIN_CROPS', '2'))
MAX_CROPS = int(os.getenv('MAX_CROPS', '6'))
MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY', '50'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '32'))
PRINT_NUM_VIS_TOKENS = os.getenv('PRINT_NUM_VIS_TOKENS', 'False').lower() == 'true'
SKIP_REPEAT = os.getenv('SKIP_REPEAT', 'True').lower() == 'true'

MODEL_PATH = os.getenv('MODEL_PATH', 'deepseek-ai/DeepSeek-OCR-2')

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# Initialize tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
