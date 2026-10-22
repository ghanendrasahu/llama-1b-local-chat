import gc

import torch
from transformers import BitsAndBytesConfig, pipeline

# This is the secret sauce for 8GB RAM
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "unsloth/Llama-3.2-1B-Instruct" 

print("\n--- LOADING IN 4-BIT MODE (NO DISK OFF-LOADING) ---")

generator = pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={"quantization_config": quant_config}, # Squeezes it!
    device_map="auto" 
)

# Use the 'messages' format to keep the AI focused
messages = [
    {"role": "user", "content": "Explain why a 1B parameter model is better for my 8GB RAM laptop than a 7B model."}
]

print("--- AI IS THINKING (Should be instant now) ---")
result = generator(messages, max_new_tokens=100, temperature=0.7)

print("\nRESULT:")
print(result[0]['generated_text'][-1]['content'])

# Cleanup
del generator
gc.collect()