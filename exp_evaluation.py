import numpy as np
from bsbi import BSBIIndex
from tqdm import tqdm
from compression import VBEPostings
import math
import json
import pandas as pd

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

def retrieve_and_generate_binary_relevancy_vector(q_docs_dict, queries, k=5):
    """
    Perform retrieval using TF-IDF, then compare the results to the ground truth in qrels.

    Parameters
    ----------
    q_docs_dict: dict(str, List[str])
        Dictionary with query IDs as keys and relevant corpus-ids as values.
    queries: dict(str, str)
        Dictionary with query IDs as keys and query text as values.
    k: int
        Top-k results to retrieve.

    Returns
    -------
    dict(str, List[int])
        Dictionary with query ID as keys and binary vectors indicating relevancy as values.
    """
    q_ranking_dict = {}
    for q_id, query in tqdm(queries.items()):
        
        retrieved_docs = [doc for (_, doc) in BSBI_instance.retrieve_tfidf_taat(query, k)]
        ground_truth_docs = q_docs_dict.get((q_id), [])
        binary_relevancy_vector = [1 if doc in ground_truth_docs else 0 for doc in retrieved_docs]
        q_ranking_dict[q_id] = binary_relevancy_vector
    return q_ranking_dict

class Metrics:
    def __init__(self, ranking):
        self.ranking = ranking
    
    def rbp(self, p=0.8):
        rbp_score = 0.0
        for i, rel in enumerate(self.ranking):
            rbp_score += rel * (p ** i)
        return (1 - p) * rbp_score
    
    def dcg(self):
        dcg_score = 0.0
        for i, rel in enumerate(self.ranking):
            dcg_score += rel / math.log2(i + 2)
        return dcg_score
    
    def ndcg(self):
        ideal_ranking = sorted(self.ranking, reverse=True)
        ideal_dcg = 0.0
        for i, rel in enumerate(ideal_ranking):
            ideal_dcg += rel / math.log2(i + 2)
        actual_dcg = self.dcg()
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def prec(self, k):
        relevant = self.ranking[:k]
        return sum(relevant) / k if k > 0 else 0.0

    def ap(self):
        num_relevant = sum(self.ranking)
        if num_relevant == 0:
            return 0.0
        cumulative_precision = 0.0
        relevant_retrieved = 0
        for i, rel in enumerate(self.ranking):
            if rel == 1:
                relevant_retrieved += 1
                cumulative_precision += relevant_retrieved / (i + 1)
        return cumulative_precision / num_relevant

if __name__ == '__main__':
    q_docs_dict = read_qrels('qrels/train.tsv')
    queries = read_queries('queries_train.jsonl')
    
    q_ranking_dict = retrieve_and_generate_binary_relevancy_vector(q_docs_dict, queries)

    eval = {
        "rbp": [],
        "dcg": [],
        "ndcg": [],
        "prec@5": [],
        "ap": []
        # "prec@10": [],
    }

    for _, ranking in q_ranking_dict.items():
        metrics = Metrics(ranking)
        eval['rbp'].append(metrics.rbp())
        eval['dcg'].append(metrics.dcg())
        eval['ndcg'].append(metrics.ndcg())
        eval['prec@5'].append(metrics.prec(5))
        eval['ap'].append(metrics.ap())
        # eval['prec@10'].append(metrics.prec(10))

    for metric, scores in eval.items():
        print(f"Metrik {metric}: {sum(scores)/len(scores)}")
