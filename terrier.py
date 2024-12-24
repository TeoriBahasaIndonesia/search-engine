import pyterrier as pt
import pandas as pd
import json
import os

# if not pt.started():
#   pt.init(version='snapshot')

pt.terrier.set_version('snapshot')

def create_dataset():
    print("Me-load dataset...")

    # Buka file corpus.jsonl dan baca data
    corpus_path = "data/corpus.jsonl"
    data = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            combined_text = f"{record['title']} {record['text']}"
            data.append({"docno": record["_id"], "text": combined_text})

    # Buat dataframe
    corpus_df = pd.DataFrame(data)

    return corpus_df

def indexing(data):
    # Tentukan lokasi indeks
    index_path = os.path.join(os.getcwd(), "index-terrier")
    os.makedirs(index_path, exist_ok=True)  # Membuat folder jika belum ada

    # Konfigurasi indeks
    pd_indexer = pt.DFIndexer(index_path, \
                              type=pt.index.IndexingType(1), \
                              tokeniser=pt.index.TerrierTokeniser('utf'), \
                              stemmer=pt.index.TerrierStemmer('porter'), \
                              stopwords=pt.index.TerrierStopwords('terrier'), \
                              blocks=True, \
                              verbose=True)

    # Buat indeks
    index_ref = pd_indexer.index(data["text"], data)

    return index_ref

def load_or_create_index(data):
    """
    Fungsi untuk memuat indeks jika sudah ada, atau membuat indeks baru jika belum ada.
    Jika indeks sudah ada, maka indeks akan dihapus dan dibuat ulang.
    """
    index_dir = os.path.join(os.getcwd(), "index-terrier")
    data_properties_path = os.path.join(index_dir, "data.properties")
    print(index_dir)

    if os.path.exists(data_properties_path):
        print("Indeks ditemukan, menghapus indeks lama...")
        # Hapus folder indeks lama dan semua isinya
        import shutil
        shutil.rmtree(index_dir)

    # Setelah indeks dihapus atau belum ada, buat indeks baru
    print("Membuat indeks baru...")
    return indexing(data)

def retrieval(index, query, k=10):
    """
    Model-model retrieval menggunakan indeks yang ada.
    """
    print(f"Query: {query}")

    # Model TF-IDF
    tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF") % k
    tf_idf_results = tf_idf.search(query)
    print("\nHasil TF-IDF:")
    print(tf_idf_results)

    # Model BM25
    bm25 = pt.BatchRetrieve(index, wmodel="BM25") % k
    bm25_results = bm25.search(query)
    print("\nHasil BM25:")
    print(bm25_results)

if __name__ == "__main__":
    # Meload dataset
    data = create_dataset()

    # Memuat atau membuat indeks
    index_ref = load_or_create_index(data)

    # Ambil statistik indeks
    index_fact = pt.IndexFactory.of(index_ref)
    print(index_fact.getCollectionStatistics().toString())

    # Lakukan retrieval
    query = "dna"
    retrieval(index_ref, query)