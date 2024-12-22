import json
import pandas as pd
from bsbi import BSBIIndex
from compression import VBEPostings
from exp_evaluation import Metrics

import xgboost as xgb
import numpy as np
from tqdm import tqdm

from index import InvertedIndexReader

def load_queries(jsonl_path="queries.jsonl"):
    """
    Reads queries from a JSONL file with structure:
      {"_id": "0", "text": "sample query", "metadata": {...}}
    Returns a dict: { query_id (str): query_text (str) }
    """
    queries_dict = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["_id"]  
            text = obj["text"]
            queries_dict[qid] = text
    return queries_dict
        
def load_qrels_tsv(qrels_path="qrels/train.tsv"):
    """
    Reads a TSV with columns: query-id, corpus-id, score
    Returns a nested dict: { qid (str): { doc_id (str): score (int) } }
    """
    qrels_dict = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        next(f, None)  # skip header row if it exists
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            qid = parts[0]       
            doc_id = parts[1]      
            score = int(parts[2])  
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][doc_id] = score
    return qrels_dict

def load_qrels(qrels_file=None):
    return load_qrels_tsv(qrels_file)  

def retrieve_and_generate_relevancy_vector(
    qrels_dict,
    queries_dict,
    method="tfidf",
    top_k=20
):
    """
    For each query, we retrieve top-K docs using a retrieval method,
    then build a relevancy vector (1 for relevant, 0 for not relevant)
    based on the real qrels_dict.
    """
    from tqdm import tqdm

    q_ranking_dict = {} 

    for qid, query_text in tqdm(queries_dict.items()):
        if method == "tfidf":
            results = BSBI_instance.retrieve_tfidf_taat(query_text, k=top_k)
        else:
            results = BSBI_instance.retrieve_bm25_taat(query_text, k=top_k)

        docids = [doc for (_, doc) in results]


        relevant_docs_map = qrels_dict.get(qid, {})
        bin_vector = []
        for d in docids:
            # 1 if relevance>0, else 0
            rel = relevant_docs_map.get(d, 0)
            bin_vector.append(1 if rel > 0 else 0)

        q_ranking_dict[qid] = bin_vector

    return q_ranking_dict

def evaluate_qrels(method="tfidf", qrels_file="qrels/train.tsv", queries_file="queries.jsonl"):
    # 1) load queries
    queries_dict = load_queries(queries_file)

    # 2) load real qrels
    qrels_dict = load_qrels_tsv(qrels_file)

    # 3) retrieve & compare
    q_ranking_dict = retrieve_and_generate_relevancy_vector(qrels_dict, queries_dict, method=method, top_k=20)

    # 4) compute metrics
    from math import log2
    eval = { "rbp": [], "dcg": [], "ndcg": [], "prec@5": [], "prec@10": [], "ap": [] }
    for qid, ranking in q_ranking_dict.items():
        metrics = Metrics(ranking)
        eval['rbp'].append(metrics.rbp(p=0.8))
        eval['dcg'].append(metrics.dcg())
        eval['ndcg'].append(metrics.ndcg())
        eval['prec@5'].append(metrics.prec(5))
        eval['prec@10'].append(metrics.prec(10))
        eval['ap'].append(metrics.ap())
    for metric, scores in eval.items():
        print(f"{metric}: {sum(scores)/len(scores) if scores else 0.0}")


def get_bm25_score(query, doc_name):
    """
    Retrieve many docs with BM25, then find doc_name in the results.
    Not very efficient, but simple to illustrate the idea.
    """
    results = BSBI_instance.retrieve_bm25_taat(query, k=999999)
    for (score, doc) in results:
        if doc == doc_name:
            return score
    return 0.0

def get_tfidf_score(query, doc_name):
    """
    Retrieve many docs with TF-IDF, then find doc_name in the results.
    """
    results = BSBI_instance.retrieve_tfidf_taat(query, k=999999)
    for (score, doc) in results:
        if doc == doc_name:
            return score
    return 0.0

def get_doc_length(doc_name):
    return 0.0
 

def extract_features(query, doc_name, bsbi_instance=None):
    from porter2stemmer import Porter2Stemmer
    """
    Extract features directly using InvertedIndexReader for a query-document pair.
    """
    
    if not bsbi_instance:
        bsbi_instance = BSBIIndex(data_dir='data',
                          postings_encoding=VBEPostings,
                          output_dir='index')
        
    query_terms = [Porter2Stemmer().stem(token.lower()) for token in query.split()]
    doc_id = bsbi_instance.doc_id_map[doc_name]

    with InvertedIndexReader(bsbi_instance.index_name, bsbi_instance.postings_encoding, directory=bsbi_instance.output_dir) as index:
        bm25_score, tfidf_score = index.get_document_scores(query_terms, doc_id, bsbi_instance.term_id_map)

    # Feature vector
    doc_length = index.doc_length.get(doc_id, 0)
    return [bm25_score, tfidf_score, doc_length]


# -------------------------------------------------------------------------
# 4. BUILD TRAINING DATA
#    We'll do a "candidate docs" approach for train queries:
#    - For each query, retrieve top-K with BM25 (or both BM25 & TF-IDF).
#    - Mark them relevant if qrels says score>0, else 0.
#    - Extract features, store in X, y.
# -------------------------------------------------------------------------

def build_training_data(queries_dict, qrels_dict, top_k=50):
    """
    queries_dict: { qid -> text }
    qrels_dict: { qid -> { doc_id -> relevance_int } }
    top_k: how many docs to consider from BM25 as training candidates

    Returns X, y arrays for training a binary LTR model, plus a list of query IDs
    so we can do group-based ranking if needed.
    """
    X = []
    y = []
    qid_list = []  # store the query ID for each row

    for qid, query_text in tqdm(queries_dict.items(), desc="Building train data"):
        # 1) get top-K docs from BM25
        results = BSBI_instance.retrieve_bm25_taat(query_text, k=top_k)
        # results is list of (score, docName)

        # 2) for each doc, extract features + label
        relevant_docs_map = qrels_dict.get(qid, {})
        for (_, doc_name) in results:
            feats = extract_features(query_text, doc_name)
            label = relevant_docs_map.get(doc_name, 0)  # if not in qrels => 0
            X.append(feats)
            y.append(label)
            qid_list.append(qid)

    return np.array(X), np.array(y), qid_list

    
def train_xgb_ranker(X, y, qid_list):
    """
    X: (num_samples, num_features)
    y: (num_samples,) 0 or 1
    qid_list: List of query IDs, same length as X.
              We'll group them so XGBoost knows how many docs per query.
    """
    import collections

    # Build group sizes (how many docs per query)
    # This is needed for "rank:pairwise" or "rank:ndcg" objectives
    qid_counts = collections.Counter(qid_list)

    # We must keep them in the same order as X
    # so we'll accumulate in a loop
    current_q = qid_list[0]
    groups = []
    count = 0
    last_q = qid_list[0]
    for i, q in enumerate(qid_list):
        if q == last_q:
            count += 1
        else:
            groups.append(count)
            count = 1
            last_q = q
    groups.append(count)  # for the last group

    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups)

    params = {
        'objective': 'rank:pairwise',  # or 'rank:ndcg'
        'eval_metric': 'ndcg',
        'eta': 0.1,
        'max_depth': 6
    }
    model = xgb.train(params, dtrain, num_boost_round=50)
    return model


# -------------------------------------------------------------------------
# 6. RE-RANK / TEST / EVALUATE
# -------------------------------------------------------------------------

def rerank_with_model(model, query_text, top_k=50, bsbi_instance=None):
    """
    1) Get top-K docs from BM25
    2) For each doc, extract features
    3) Predict with the model
    4) Sort descending by predicted score
    5) Return list of (pred_score, docName)
    """
    if not bsbi_instance:
        bsbi_instance = BSBIIndex(data_dir='data',
                          postings_encoding=VBEPostings,
                          output_dir='index')

    results = bsbi_instance.retrieve_bm25_taat(query_text, k=top_k)
    doc_features = []
    doc_names = []
    for (_, doc_name) in results:
        feats = extract_features(query_text, doc_name)
        doc_features.append(feats)
        doc_names.append(doc_name)

    dtest = xgb.DMatrix(np.array(doc_features))
    preds = model.predict(dtest)

    doc_pred_pairs = list(zip(preds, doc_names))
    doc_pred_pairs.sort(key=lambda x: x[0], reverse=True)  # high→low
    return doc_pred_pairs


def evaluate_model(model, queries_dict, qrels_dict, top_k=20):
    """
    For each query, we:
      1) re-rank with the model
      2) build a binary vector vs. the ground truth
      3) compute metrics
    Then average them across queries.
    """
    ndcgs = []
    maps_ = []

    for qid, query_text in tqdm(queries_dict.items(), desc="Evaluating LTR"):
        # re-rank
        doc_pred_pairs = rerank_with_model(model, query_text, top_k=top_k)
        # doc_pred_pairs => list of (pred_score, doc_name)

        # build binary vector
        relevant_map = qrels_dict.get(qid, {})
        # keep top_k docs
        bin_vector = []
        for _, doc_name in doc_pred_pairs[:top_k]:
            rel = relevant_map.get(doc_name, 0)
            bin_vector.append(1 if rel > 0 else 0)

        M = Metrics(bin_vector)
        ndcgs.append(M.ndcg())
        maps_.append(M.ap())

    print(f"Average nDCG@{top_k}: {sum(ndcgs)/len(ndcgs):.4f}")
    print(f"Average MAP@{top_k}: {sum(maps_)/len(maps_):.4f}")


def train_ltr(features_train, labels_train):
    X = np.array(features_train)
    y = np.array(labels_train)

    train_data = xgb.DMatrix(X, label=y)

    params = {
        "objective": "rank:pairwise",   # or rank:ndcg
        "eval_metric": "ndcg",
        "eta": 0.1,
        "max_depth": 6
    }
    model = xgb.train(params, train_data, num_boost_round=50)
    model.save_model("ltr_model.xgb")
        
def rank_with_model(model, query_text):
    # 1) get top-K docs from BM25
    results = BSBI_instance.retrieve_bm25_taat(query_text, k=100)
    # 2) build feature vectors
    doc_features = []
    doc_ids = []
    for (_, doc) in results:
        feats = extract_features(query_text, doc)
        doc_features.append(feats)
        doc_ids.append(doc)
    
    # 3) predict
    dtest = xgb.DMatrix(np.array(doc_features))
    preds = model.predict(dtest)

    # 4) sort by predicted score descending
    doc_pred_pairs = list(zip(doc_ids, preds))
    doc_pred_pairs.sort(key=lambda x: x[1], reverse=True)
    return doc_pred_pairs  # now in best→worst order




        
def evaluate_ltr(model, test_queries, qrels_test):
    all_scores = {"rbp": [], "dcg": [], "ndcg": [], "prec@5": [], "prec@10": [], "ap": [], "map": []}
    for qid, query_text in test_queries.items():
        # re-rank
        ranked_docs = rank_with_model(model, query_text)  # [(doc, pred_score), ...]

        # build a binary vector vs qrels
        true_rels = qrels_test.get(qid, {})
        bin_vector = [1 if true_rels.get(doc, 0) > 0 else 0 for (doc, _) in ranked_docs[:20]]

        # compute metrics (like your Metrics class)
        M = Metrics(bin_vector)
        all_scores["rbp"].append(M.rbp(p=0.8))
        all_scores["dcg"].append(M.dcg())
        all_scores["ndcg"].append(M.ndcg())
        all_scores["prec@5"].append(M.prec(5))
        all_scores["prec@10"].append(M.prec(10))
        all_scores["ap"].append(M.ap())
    
    # average
    for metric, arr in all_scores.items():
        print(f"{metric} = {sum(arr)/len(arr):.4f}")
        
        
if __name__ == '__main__':
    BSBI_instance = BSBIIndex(data_dir='data',
                          postings_encoding=VBEPostings,
                          output_dir='index')
    
    # evaluate_qrels()
    
    # candidate_docs = {}
    # for qid, query_text in train_queries.items():
    #     # Suppose top_k=100
    #     results = BSBI_instance.retrieve_bm25_taat(query_text, k=100) 
    #     # results is list of (score, docName)
    #     candidate_docs[qid] = results


    # 1) Load queries
    train_queries = load_queries("queries.jsonl")  # adjust path if needed
    # 2) Load qrels (train and test)
    qrels_train = load_qrels_tsv("qrels/train.tsv")
    qrels_test = load_qrels_tsv("qrels/test.tsv")

    # 3) Build training data (we use top-50 from BM25)
    X_train, y_train, qid_list_train = build_training_data(train_queries, qrels_train, top_k=50)

    # 4) Train a ranker
    model = train_xgb_ranker(X_train, y_train, qid_list_train)
    model.save_model("ltr_model.xgb")
    print("Model trained and saved.")

    evaluate_model(model, train_queries, qrels_test, top_k=20)

    print("Done!")
