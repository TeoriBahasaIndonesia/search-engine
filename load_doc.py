import json

def load_corpus(corpus_path):
    """
    Load the corpus from a JSONL file into a dictionary.

    Parameters
    ----------
    corpus_path : str
        Path to the corpus JSONL file.

    Returns
    -------
    dict
        The loaded corpus as a dictionary.
    """
    documents = {}
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                doc_id = doc.get('_id')
                if doc_id:
                    documents[doc_id] = doc
    except FileNotFoundError:
        print(f"Corpus file not found at {corpus_path}. Ensure the path is correct.")
    return documents
