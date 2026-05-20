# GENAI

A lightweight local text-generation demo using Hugging Face Transformers with 4-bit quantization for lower memory usage.

## What this project does
- Loads `unsloth/Llama-3.2-1B-Instruct`
- Uses 4-bit quantization via `BitsAndBytesConfig`
- Runs a sample chat-style prompt
- Prints the generated response

## Requirements
- Python 3.10+
- A machine with compatible PyTorch setup

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
```bash
python app.py
```

## Notes
- 4-bit quantization helps reduce RAM usage.
- First model load may take time due to download/caching.
