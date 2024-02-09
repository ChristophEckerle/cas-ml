# INIT INFERENCE SCRIPT FOR TEST

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Activate eval mode for inference
model.eval()

# Load quantized model and tokenizer
def model_fn(model_dir):
    model_8bit = AutoModelForSeq2SeqLM.from_pretrained(model_dir, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model_8bit, tokenizer

def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = data.pop("inputs", data)
    encoded_input = tokenizer(text, return_tensors='pt')
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), **data)
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)