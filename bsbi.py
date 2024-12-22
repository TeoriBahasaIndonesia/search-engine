import os
import pickle
import contextlib
import heapq
import math
import re
from porter2stemmer import Porter2Stemmer
import requests
import string

from index import InvertedIndexReader, InvertedIndexWriter
from trie import Trie
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm

# Tambahan imports
import nltk
from nltk.corpus import stopwords
from porter2stemmer import Porter2Stemmer
from nltk.tokenize import RegexpTokenizer
import json

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    trie(Trie): Class Trie untuk query auto-completion
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.trie = Trie()

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []
        
        # Tambahan:
        # Initialize stemmer dan stopwords
        self.stemmer = Porter2Stemmer()
        self.tokenizer = RegexpTokenizer(r'\w+')
                
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))


    def save(self):
        """Menyimpan doc_id_map, term_id_map, dan trie ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'trie.pkl'), 'wb') as f:
            # file ini mungkin agak besar
            pickle.dump(self.trie, f)

    def load(self):
        """Memuat doc_id_map, term_id_map, dan trie dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'trie.pkl'), 'rb') as f:
            self.trie = pickle.load(f)
            
            
    # # Helper function untuk melakukan tokenisasi
    # def tokenize(self, text):
    #     # Lowercase the text
    #     text = text.lower()
    #     # Use regex to find words
    #     tokens = re.findall(r'\b\w+\b', text)
    #     return tokens

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Anda bisa menggunakan stemmer bahasa Inggris yang tersedia, seperti Porter Stemmer
        https://github.com/evandempsey/porter2-stemmer

        Untuk membuang stopwords, Anda dapat menggunakan library seperti NLTK.

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO
        # Hint: Anda dapat mengisi trie di sini
        td_pairs = []
        full_block_path = os.path.join(self.data_dir, block_path)
        for root, dirs, files in os.walk(full_block_path):
            for file in files:
                # Mendapatkan full path menuju file
                file_path = os.path.join(root, file)
                
                # Membaca file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Tokenize content
                tokens_raw = self.tokenizer.tokenize(content)

                # Lowercase tokens
                tokens_lower = [token.lower() for token in tokens_raw]

                # Remove stopwords
                tokens_no_stopwords = [token for token in tokens_lower if token not in self.stopwords]

                # Stemming tokens
                tokens_stemmed = [self.stemmer.stem(token) for token in tokens_no_stopwords]

                # Insert raw tokens ke trie (sebelum preprocessing), agar saran yang diberikan pada pengguna 
                # merupakan kata-kata yang lebih alami
                for token_raw in tokens_raw:
                    self.trie.insert(token_raw, 1)

                # Mendapatkan term IDs
                term_ids = [self.term_id_map[token] for token in tokens_stemmed]
                
                # Mendapatkan doc ID
                relative_file_path = os.path.relpath(file_path, self.data_dir)
                doc_id = self.doc_id_map[relative_file_path]
                
                 # Untuk setiap term ID, add (termID, docID) ke dalam td_pairs
                for term_id in term_ids:
                    td_pairs.append((term_id, doc_id))
                
        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # term_dict merupakan dictionary yang berisi dictionary yang
        # melakukan mapping dari doc_id ke tf
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = dict()
            # Mengupdate juga TF (yang merupakan value dari dictionary yang di dalam)
            term_dict[term_id][doc_id] = term_dict[term_id].get(doc_id, 0) + 1
        
        for term_id in sorted(term_dict.keys()):
            # Sort postings list (dan tf list yang bersesuaian)
            sorted_postings_tf = dict(sorted(term_dict[term_id].items()))
            # Postings list adalah keys, TF list adalah values
            index.append(term_id, list(sorted_postings_tf.keys()), 
                         list(sorted_postings_tf.values()))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def _compute_score_tfidf(self, tf, df, N):
        """
        Fungsi ini melakukan komputasi skor TF-IDF.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        score = w(t, Q) x w(t, D)
        Tidak perlu lakukan normalisasi pada score.

        Gunakan log basis 10.

        Parameters
        ----------
        tf: int
            Term frequency.

        df: int
            Document frequency.

        N: int
            Jumlah dokumen di corpus. 

        Returns
        -------
        float
            Skor hasil perhitungan TF-IDF.
        """
        # TODO
        w_tq = math.log10(N / df)
        w_td = 1 + math.log10(tf) if tf > 0 else 0
        score = w_tq * w_td
        return score
    
    def _compute_score_bm25(self, tf, df, N, k1, b, dl, avdl):
        """
        Fungsi ini melakukan komputasi skor BM25.
        Gunakan log basis 10 dan tidak perlu lakukan normalisasi.
        Silakan lihat penjelasan parameters di slide.

        Returns
        -------
        float
            Skor hasil perhitungan TF-IDF.
        """
        # TODO
        
        # Menggunakan Alternative 2 IDF
        idf = math.log10((N - df + 0.5) / (df + 0.5))
        
        tf_numerator = (k1 + 1) * tf
        tf_denominator = k1 * ((1- b) + b * (dl/avdl)) + tf
        score = idf * (tf_numerator / tf_denominator)
        return score

    def retrieve_tfidf_daat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema DaaT.
        Method akan mengembalikan top-K retrieval results.

        Program tidak perlu paralel sepenuhnya. Untuk mengecek dan mengevaluasi
        dokumen yang di-point oleh pointer pada waktu tertentu dapat dilakukan
        secara sekuensial, i.e., seperti menggunakan for loop.

        Beberapa informasi penting: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        
        """ 
        - Pada DaaT, setiap postings list untuk term akan diinisiasi dengan pointer pada posisi awal
        - Lalu, akan dilakukan pencarian docId minimum di antara pointer saat ini. 
        - Jika docId ada di setiap postings list:
          1. akan dilakukan perhitungan skor untuk dokumen tersebut. Selain itu, juga 
          2. akan diincrement pointer untuk postings list yang mengandung docId tersebut. 
        - Jika docId tidak di setiap postings list:
          1. increment pointer pada postings list dengan docId minimum
        
        """
        
        # TODO      
        self.load()
        
        # Tokenize dan stem query tokens
        tokens = [Porter2Stemmer().stem(token.lower()) for token in query.split()]
        
        # Open inverted index reader
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            postings_data = []
            N = len(index.doc_length)
            for token in tokens:
                term_id = self.term_id_map[token]
                if term_id is None:
                    continue
                postings_list, tf_list = index.get_postings_list(term_id)
                if not postings_list:
                    continue
                df = index.postings_dict[term_id][1]
                postings_data.append((postings_list, tf_list, df))

            if not postings_data:
                return []

            # Inisialisasi pointers untuk setiap postings list
            pointers = [0] * len(postings_data)
            scores = {} 

            while True:
                current_docs = []
                
                # Mendapatkan current document IDs dari setiap postings list
                for i, (postings, _, _) in enumerate(postings_data):
                    if pointers[i] < len(postings):
                        current_docs.append(postings[pointers[i]])
                        
                if not current_docs:
                    break
                
                current_doc = min(current_docs)  

                # Mengakumulasikan skor dari setiap terms untuk current document
                score = 0.0
                for i, (postings, tf_list, df) in enumerate(postings_data):
                    if pointers[i] < len(postings) and postings[pointers[i]] == current_doc:
                        tf = tf_list[pointers[i]]
                        score += self._compute_score_tfidf(tf, df, N)
                        pointers[i] += 1 # Memajukan pointer
                scores[current_doc] = score

            return self.get_top_k_by_score(scores, k)

    def retrieve_tfidf_taat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Beberapa informasi penting: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """

        """ 
        - Pada TaaT, sistem memproses satu term (kata kunci) dalam query pada satu waktu.
        - Setelah memproses satu term, skor relevansi untuk tiap dokumen yang mengandung term tsb akan diperbarui
        - Skor relevansi setiap dokumen akan diakumulasi dengan menjumlahkan kontribusi setiap term
        """
        
        # TODO
        self.load()
        
        # Tokenize dan stem query tokens
        tokens = [Porter2Stemmer().stem(token.lower()) for token in query.split()]
        
        # Open inverted index reader
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            scores = {}
            N = len(index.doc_length)
            for token in tokens:
                term_id = self.term_id_map[token]
                if term_id is None:
                    continue
                postings_list, tf_list = index.get_postings_list(term_id)
                if not postings_list:
                    continue
                df = index.postings_dict[term_id][1]
                for doc_id, tf in zip(postings_list, tf_list):
                    score = self._compute_score_tfidf(tf, df, N)
                    scores[doc_id] = scores.get(doc_id, 0.0) + score
            return self.get_top_k_by_score(scores, k)

    def retrieve_bm25_taat(self, query, k=10, k1=1.2, b=0.75):
        """
        Lakukan retrieval BM-25 dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        # TODO
        self.load()

        tokens = [Porter2Stemmer().stem(token.lower()) for token in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            scores = {}
            N = len(index.doc_length)
            avdl = index.get_average_document_length()
            for token in tokens:
                term_id = self.term_id_map[token]
                if term_id is None:
                    continue
                postings_list, tf_list = index.get_postings_list(term_id)
                if not postings_list:
                    continue
                df = index.postings_dict[term_id][1]
                for doc_id, tf in zip(postings_list, tf_list):
                    dl = index.doc_length[doc_id]
                    score = self._compute_score_bm25(tf, df, N, k1, b, dl, avdl)
                    scores[doc_id] = scores.get(doc_id, 0.0) + score
            return self.get_top_k_by_score(scores, k)


    # def do_indexing(self):
    #     """
    #     Base indexing code
    #     BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
    #     based indexing)

    #     Method ini scan terhadap semua data di collection, memanggil parsing_block
    #     untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
    #     di setiap block dan menyimpannya ke index yang baru.
    #     """
    #     # loop untuk setiap sub-directory di dalam folder collection (setiap block)
    #     for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
    #         td_pairs = self.parsing_block(block_dir_relative)
    #         index_id = 'intermediate_index_'+block_dir_relative
    #         self.intermediate_indices.append(index_id)
    #         with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
    #             self.write_to_index(td_pairs, index)
    #             td_pairs = None

    #     self.save()

    #     with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
    #         with contextlib.ExitStack() as stack:
    #             indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
    #                        for index_id in self.intermediate_indices]
    #             self.merge_index(indices, merged_index)
    
    ### CHANGED - do_indexing tidak lagi loop folder, tapi langsung baca corpus.jsonl ###
    def do_indexing(self):
        """
        Versi simple: hanya 1 blok (corpus.jsonl). 
        Kalau Anda mau multi-block, silakan pecah corpus.jsonl jadi beberapa chunk.
        """
        all_td_pairs = []

        corpus_path = os.path.join(self.data_dir, 'corpus.jsonl')
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading corpus.jsonl"):
                if not line.strip():
                    continue
                doc = json.loads(line)
                # doc["_id"], doc["title"], doc["text"], dsb.
                doc_id_str = doc["_id"]
                # dapatkan integer doc_id
                doc_id = self.doc_id_map[doc_id_str]

                text_content = (doc.get("title", "") + " " + doc.get("text", "")).strip()

                # Tokenisasi
                tokens_raw = self.tokenizer.tokenize(text_content)
                tokens_lower = [t.lower() for t in tokens_raw]
                tokens_nostop = [t for t in tokens_lower if t not in self.stopwords]
                tokens_stemmed = [self.stemmer.stem(t) for t in tokens_nostop]

                # Insert kata-kata mentah ke Trie (untuk autocomplete):
                for token_raw in tokens_raw:
                    self.trie.insert(token_raw, 1)

                for term in tokens_stemmed:
                    term_id = self.term_id_map[term]
                    all_td_pairs.append((term_id, doc_id))

        # Tulis ke index
        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as idx:
            self.write_to_index(all_td_pairs, idx)

        # Simpan mapping dan trie
        self.save()

        print("Indexing done. Index stored at:", self.index_name)
        print("Number of unique terms =", len(self.term_id_map))
        print("Number of docs =", len(self.doc_id_map))
    
    def get_top_k_by_score(self, score_docs, k):
        """
        Method ini berfungsi untuk melakukan sorting terhadap dokumen berdasarkan score
        yang dihitung, lalu mengembalikan top-k dokumen tersebut dalam bentuk tuple
        (score, document). Silakan gunakan heap agar lebih efisien.

        Parameters
        ----------
        score_docs: Dictionary[int -> float]
            Dictionary yang berisi mapping docID ke score masing-masing dokumen tersebut.

        k: Int
            Jumlah dokumen yang ingin di-retrieve berdasarkan score-nya.

        Result
        -------
        List[(float, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        # TODO
        top_docs = heapq.nlargest(k, score_docs.items(), key=lambda x: x[1])
        results = []
        for doc_id, score in top_docs:
            try:
                doc_name = self.doc_id_map[doc_id]
                results.append((score, doc_name))
            except IndexError:
                print(f"Warning: doc_id {doc_id} is out of range.")
        return results
    
    def get_query_recommendations(self, query, k=5):
        # Method untuk mendapatkan rekomendasi untuk QAC
        # Tidak perlu mengubah ini
        self.load()
        last_token = query.split()[-1]
        recc = self.trie.get_recommendations(last_token, k)
        return recc
    
    
    
    

if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='data',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!