import os
import warnings
import numpy as np
import re
import fitz  # PyMuPDF
from flask import Flask, request, render_template, redirect, url_for
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, BertForQuestionAnswering, BertTokenizer, DPRQuestionEncoderTokenizer,DPRQuestionEncoder 
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at")
warnings.filterwarnings("ignore", message="The tokenizer class you load from this checkpoint")

# Initialize models and tokenizers
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return preprocess_text(text)

def split_into_passages(text, passage_length=100):
    words = text.split()
    passages = [' '.join(words[i:i + passage_length]) for i in range(0, len(words), passage_length)]
    return passages

def encode_passages(passages):
    inputs = context_tokenizer(passages, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = context_encoder(**inputs).pooler_output
    return embeddings, passages


def encode_question(question):
    inputs = question_tokenizer(question, return_tensors='pt')
    with torch.no_grad():
        question_embedding = question_encoder(**inputs).pooler_output
    return question_embedding

def retrieve_passages(question_embedding, passage_embeddings, passages, top_k=5):
    similarities = cosine_similarity(question_embedding.numpy(), passage_embeddings.numpy())[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    top_k_passages = [passages[idx] for idx in top_k_indices]
    return top_k_passages

def extract_best_answer(question, passages):
    best_answer = ""
    highest_score = float('-inf')
    for passage in passages:
        inputs = qa_tokenizer(question, passage, return_tensors='pt')
        with torch.no_grad():
            outputs = qa_model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

        all_tokens = qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1
        answer = qa_tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index])

        score = start_scores[0][start_index] + end_scores[0][end_index - 1]
        if score > highest_score:
            highest_score = score
            best_answer = answer
    return best_answer

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            question = request.form['question']
            
            text = extract_text_from_pdf(file_path)
            passages = split_into_passages(text)
            passage_embeddings, passages = encode_passages(passages)
            question_embedding = encode_question(question)
            top_passages = retrieve_passages(question_embedding, passage_embeddings, passages)
            relevant_passage = top_passages[0]
            best_answer = extract_best_answer(question, top_passages)
            
            return render_template('index.html', answer=best_answer, relevant_passage=relevant_passage)
    
    return render_template('index.html', answer=None, relevant_passage=None)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
