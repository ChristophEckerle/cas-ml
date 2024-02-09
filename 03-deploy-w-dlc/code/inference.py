
# INIT INFERENCE SCRIPT FOR WEB

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sagemaker_inference import encoder, decoder

# Define & load model
MODEL_NAME = "mrm8488/bert2bert_shared-german-finetuned-summarization"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, load_in_8bit=True)

# Activate eval mode for inference
model.eval()

def model_fn(model_dir):
    """
    Load Model for Inference:
    """
    return model

def input_fn(request_body, request_content_type):
    """
    Deseralize Input-Sequences:
    """
    if request_content_type == "application/json":
        input_data = encoder.json_to_dict(request_body)
        text = input_data['text']
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    return text

# Predict function to retrieve prediction from model
def predict_fn(input_data, model):
    """
    Generate prediction based on input sequence:
    """
    # encode input via tokenizer, max_length of the BERT model is 512 inputtokens!
    inputs = tokenizer.encode("summarize: " + input_data, return_tensors="pt", max_length=512, truncation=True)

    # generate summarization
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # decode and return prediction
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def output_fn(prediction_output, accept):
    """
    Seralize model prediction for the web:
    """
    if accept == "application/json":
        output = {'summary': prediction_output}
        return encoder.encode(output, accept)
    raise ValueError(f"Unsupported accept type: {accept}")
