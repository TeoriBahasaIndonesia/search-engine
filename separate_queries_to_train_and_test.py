import json
import pandas as pd

def split_queries(queries_file, qrels_train_file, qrels_test_file, output_train_file, output_test_file):
    """
    Splits the queries.jsonl file into train and test JSONL files based on qrels data.

    Parameters
    ----------
    queries_file: str
        Path to the input queries.jsonl file.
    qrels_train_file: str
        Path to the qrels/train.tsv file.
    qrels_test_file: str
        Path to the qrels/test.tsv file.
    output_train_file: str
        Path to save the output queries_train.jsonl file.
    output_test_file: str
        Path to save the output queries_test.jsonl file.
    """
    # Load qrels train and test to get query IDs
    train_qids = set(pd.read_csv(qrels_train_file, sep='\t')['query-id'].astype(str))
    test_qids = set(pd.read_csv(qrels_test_file, sep='\t')['query-id'].astype(str))
    
    # Read the queries.jsonl file and split into train and test
    with open(queries_file, 'r') as f:
        queries = [json.loads(line) for line in f]
    
    train_queries = []
    test_queries = []
    
    for query in queries:
        query_id = query["_id"]
        if query_id in train_qids:
            train_queries.append(query)
        elif query_id in test_qids:
            test_queries.append(query)
    
    # Write train queries to JSONL
    with open(output_train_file, 'w') as f:
        for q in train_queries:
            f.write(json.dumps(q) + '\n')
    
    # Write test queries to JSONL
    with open(output_test_file, 'w') as f:
        for q in test_queries:
            f.write(json.dumps(q) + '\n')

    print(f"Train queries written to {output_train_file}")
    print(f"Test queries written to {output_test_file}")



if __name__ == '__main__':
    split_queries(
        queries_file='queries.jsonl',
        qrels_train_file='qrels/train.tsv',
        qrels_test_file='qrels/test.tsv',
        output_train_file='queries_train.jsonl',
        output_test_file='queries_test.jsonl'
    )
