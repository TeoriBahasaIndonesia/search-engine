import numpy as np
from bsbi import BSBIIndex
from tqdm import tqdm
from compression import VBEPostings
import math
import json
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

BSBI_instance = BSBIIndex(data_dir='data',
                          postings_encoding=VBEPostings,
                          output_dir='index')

def read_qrels(file_path):
    """
    Read qrels data from TSV file and store in a dictionary.

    Parameters
    ----------
    file_path: str
        Path to the qrels file (train.tsv or test.tsv).

    Returns
    -------
    dict(str, List[str])
        Dictionary with query-id as keys and list of relevant corpus-ids as values.
    """
    qrels = pd.read_csv(file_path, sep='\t')
    q_docs_dict = {}
    for _, row in qrels.iterrows():
        q_id, corpus_id, score = row["query-id"], row["corpus-id"], row["score"]
        if score > 0:
            if q_id not in q_docs_dict:
                q_docs_dict[str(q_id)] = []
            q_docs_dict[str(q_id)].append(str(corpus_id))
    return q_docs_dict

def read_queries(file_path):
    """
    Read queries from a JSONL file.

    Parameters
    ----------
    file_path: str
        Path to the queries JSONL file.

    Returns
    -------
    dict
        Dictionary with query IDs as keys and query text as values.
    """
    queries = {}
    with open(file_path, 'r') as f:
        for line in f:
            query = json.loads(line)
            queries[query['_id']] = query['text']
    return queries

def load_corpus(file_path):
    """
    Load corpus from JSONL file and store it in a dictionary.

    Parameters
    ----------
    file_path: str
        Path to the corpus JSONL file.

    Returns
    -------
    dict(str, str)
        Dictionary with document IDs as keys and document text as values.
    """
    corpus = {}
    with open(file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc['_id']
            doc_text = (doc.get('title', '') + ' ' + doc.get('text', '')).strip()
            corpus[doc_id] = doc_text
    return corpus

def retrieve_top_k_bm25(queries, corpus, k=30):
    """
    Retrieve top-k documents using BM25.

    Parameters
    ----------
    queries: dict(str, str)
        Dictionary with query IDs as keys and query text as values.
    corpus: dict(str, str)
        Dictionary with document IDs as keys and document text as values.
    k: int
        Number of top documents to retrieve.

    Returns
    -------
    dict(str, List[Tuple[str, str]])
        Dictionary with query IDs as keys and lists of tuples (doc_id, text) as values.
    """
    retrieved_docs = {}
    for q_id, query in tqdm(queries.items()):
        retrieved = BSBI_instance.retrieve_bm25_taat(query, k)
        retrieved_docs[q_id] = [(doc_id, corpus[doc_id]) for (_, doc_id) in retrieved if doc_id in corpus]
    return retrieved_docs

def prepare_ltr_data(retrieved_docs, q_docs_dict, encoder):
    """
    Prepare data for Learning to Rank (LTR).

    Parameters
    ----------
    retrieved_docs: dict
        Retrieved documents from BM25.
    q_docs_dict: dict
        Ground truth relevance from qrels.
    encoder: CrossEncoder
        Pre-trained CrossEncoder model.

    Returns
    -------
    X, y: np.ndarray, np.ndarray
        Feature matrix and labels for LTR.
    """
    X, y = [], []
    for q_id, docs in tqdm(retrieved_docs.items()):
        for doc_id, doc_text in docs:
            label = 1 if doc_id in q_docs_dict.get(q_id, []) else 0
            query_text = queries[q_id]
            features = encoder.predict([(query_text, doc_text)])
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

def train_ltr_model(X, y):
    """
    Train an LTR model using Random Forest.

    Parameters
    ----------
    X: np.ndarray
        Feature matrix.
    y: np.ndarray
        Labels.

    Returns
    -------
    model: RandomForestClassifier
        Trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

if __name__ == '__main__':
    # Load data
    q_docs_dict = read_qrels('qrels/train.tsv')
    queries = read_queries('queries_train.jsonl')
    corpus = load_corpus('data/corpus.jsonl')

    # Retrieve top-30 documents using BM25
    retrieved_docs = retrieve_top_k_bm25(queries, corpus, k=30)

    # Initialize a CrossEncoder
    encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

    # Prepare data for LTR
    X, y = prepare_ltr_data(retrieved_docs, q_docs_dict, encoder)

    # Train LTR model
    ltr_model = train_ltr_model(X, y)

    # Save the LTR model
    with open('ltr_model_ce.pkl', 'wb') as f:
        pickle.dump(ltr_model, f)

    print("LTR model training complete and saved to ltr_model_ce.pkl!")
