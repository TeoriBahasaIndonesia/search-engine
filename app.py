# app.py
from flask import Flask, render_template, request, jsonify
import time
import os
from bsbi import BSBIIndex
from compression import VBEPostings
import json
import xgboost as xgb
import numpy as np

from ltr import extract_features, rerank_with_model


app = Flask(__name__)

# Configuration Classes
class Config:
    DEBUG = False
    TESTING = False

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

def get_config():
    env = os.getenv('FLASK_ENV', 'production')
    if env == 'development':
        return DevelopmentConfig
    else:
        return ProductionConfig

# Apply configuration
app.config.from_object(get_config())

BSBI_INSTANCE = BSBIIndex(
    data_dir=os.getenv('DATA_DIR', 'data'),
    postings_encoding=VBEPostings,
    output_dir=os.getenv('INDEX_DIR', 'index')
)
BSBI_INSTANCE.load() 

LTR_MODEL_PATH = os.path.join('.', 'ltr_model.xgb')
ltr_model = xgb.Booster()
ltr_model.load_model(LTR_MODEL_PATH)

# corpus.jsonl into a dictionary for quick access
CORPUS_PATH = os.path.join(BSBI_INSTANCE.data_dir, 'corpus.jsonl')
DOCUMENTS = {}

def load_corpus(corpus_path):
    global DOCUMENTS
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                doc_id = doc.get('_id')
                if doc_id:
                    DOCUMENTS[doc_id] = doc
    except FileNotFoundError:
        print(f"Corpus file not found at {corpus_path}. Ensure the path is correct.")

load_corpus(CORPUS_PATH)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    method = request.args.get('method', 'tfidf_taat')  
    k = int(request.args.get('k', 10))  

    if not query:
        return render_template('search_results.html', query=query, results=[], search_time=0, method=method)

    start_time = time.time()

    # Choose retrieval method
    if method == 'tfidf_daat':
        results = BSBI_INSTANCE.retrieve_tfidf_daat(query, k=k)
    elif method == 'tfidf_taat':
        results = BSBI_INSTANCE.retrieve_tfidf_taat(query, k=k)
    elif method == 'bm25_taat':
        results = BSBI_INSTANCE.retrieve_bm25_taat(query, k=k)
    elif method == 'ltr':
        results = rerank_with_model(ltr_model, query, top_k=k)
    else:
        results = rerank_with_model(ltr_model, query, top_k=k) 

    end_time = time.time()
    search_time = end_time - start_time
    formatted_results = []
    for score, doc_id in results:
        doc_id_str = str(doc_id)
        formatted_results.append((score, doc_id_str))

    return render_template(
        'search_results.html',
        query=query,
        results=formatted_results,
        search_time=search_time,
        method=method,
        DOCUMENTS=DOCUMENTS
    )


@app.route('/document/<doc_id>', methods=['GET'])
def document_preview(doc_id):
    """
    Returns the document details for the given doc_id.
    This can be rendered as a separate page or returned as JSON for AJAX.
    """
    doc = DOCUMENTS.get(doc_id)

    if doc:
        doc_name = doc.get('title', 'Untitled Document')
        content = doc.get('text', '')
    else:
        doc_name = "Unknown Document"
        content = "Document content not found."

    preview_length = 100000
    preview = content[:preview_length] + "..." if len(content) > preview_length else content

    return render_template(
        'document_preview.html',
        doc_id=doc_id,
        doc_name=doc_name,
        preview=preview
    )

if __name__ == '__main__':
    debug_mode = app.config['DEBUG']
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
