# BM-25 dengan konfigurasi parameter tertentu bisa diubah behavior-nya
# menjadi seperti TF-IDF

# Hyperparameter dari BM-25 yang bisa diubah-ubah adalah k1 dan b
# Silakan menentukan opsi k1 dan b Anda sendiri

from bsbi import BSBIIndex
from compression import VBEPostings

BSBI_instance = BSBIIndex(data_dir='data',
                          postings_encoding=VBEPostings,
                          output_dir='index')

# Isi dengan kandidat hyperparameter yang Anda inginkan
k1_candidates = [0.1, 0.5, 1.0, 1.2, 1.5, 2.0]
b_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]

query = "neural network"

# TODO: Lakukan hyperparameter tuning

def spearman_rank_correlation(list1, list2):
    """
    Fungsi untuk menghitung spearman rank correlation di antara 2 buah ranking. 
    """
    
    # Membuat sebuah list yang mengandung keseluruhan elemen pada kedua ranking
    all_docs = list(set(list1) | set(list2))
    n = len(all_docs)

    # Meng-assign rank pada setiap dokumen
    rank_dict1 = {doc_id: rank for rank, doc_id in enumerate(list1)}
    rank_dict2 = {doc_id: rank for rank, doc_id in enumerate(list2)}

    # Menyiapkan maximum rank untuk dokumen yang tidak berada di list
    max_rank1 = len(list1)
    max_rank2 = len(list2)

    # Menghitung jumlah squared rank difference
    sum_d_squared = 0
    for doc in all_docs:
        rank1 = rank_dict1.get(doc, max_rank1)
        rank2 = rank_dict2.get(doc, max_rank2)
        d = rank1 - rank2
        sum_d_squared += d * d

    # Perhitungan Spearman's rho
    rho = 1 - (6 * sum_d_squared) / (n * (n * n - 1))
    return rho

    
# TF-IDF results
tfidf_results = BSBI_instance.retrieve_tfidf_taat(query, k=100)
print(f"TF-IDF Results: {tfidf_results}")
tfidf_doc_ids = [doc for (_, doc) in tfidf_results]

if not tfidf_results:
    print("No TF-IDF results found for the query.")
    exit()

best_k1 = None
best_b = None
best_similarity = -2  # -1 <= Spearman's rho <= 1
best_bm25_results = []

print("Starting hyperparameter tuning...\n")

for k1 in k1_candidates:
    for b in b_candidates:

        bm25_results = BSBI_instance.retrieve_bm25_taat(query, k=100, k1=k1, b=b)
        bm25_doc_ids = [doc for (_, doc) in bm25_results]

        # Menghitung Spearman's rank correlation
        
        rho = spearman_rank_correlation(tfidf_doc_ids, bm25_doc_ids)
        print(f"k1: {k1}, b: {b}, Spearman's rho: {rho}")

        if rho > best_similarity:
            best_similarity = rho
            best_k1 = k1
            best_b = b
            best_bm25_results = bm25_results

print(f"\nBest hyperparameters found: k1 = {best_k1}, b = {best_b}, with Spearman's rho = {best_similarity}")

print()
print("Top 10 documents from BM25 with best hyperparameters:")
for score, doc_name in best_bm25_results[:10]:
    print(f"Dokumen: {doc_name}, Skor: {score}")        