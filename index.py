import math
import pickle
import os


class InvertedIndex:
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file; dan juga menyediakan
    mekanisme untuk menulis Inverted Index ke file (storage) saat melakukan indexing.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list)

        postings_dict adalah konsep "Dictionary" yang merupakan bagian dari
        Inverted Index. postings_dict ini diasumsikan dapat dimuat semuanya
        di memori.

        Seperti namanya, "Dictionary" diimplementasikan sebagai python's Dictionary
        yang memetakan term ID (integer) ke 4-tuple:
           1. start_position_in_index_file : (dalam satuan bytes) posisi dimana
              postings yang bersesuaian berada di file (storage). Kita bisa
              menggunakan operasi "seek" untuk mencapainya.
           2. number_of_postings_in_list : berapa banyak docID yang ada pada
              postings (Document Frequency)
           3. length_in_bytes_of_postings_list : panjang postings list dalam
              satuan byte.
           4. length_in_bytes_of_tf_list : panjang list of term frequencies dari
              postings list terkait dalam satuan byte

    terms: List[int]
        List of terms IDs, untuk mengingat urutan terms yang dimasukan ke
        dalam Inverted Index.

    """

    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Nama yang digunakan untuk menyimpan files yang berisi index
        postings_encoding : Lihat di compression.py, kandidatnya adalah StandardPostings,
                        GapBasedPostings, dsb.
        directory (str): directory dimana file index berada
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # Untuk keep track urutan term yang dimasukkan ke index
        # key: doc ID (int), value: document length (number of tokens)
        self.doc_length = {}
        # Ini nantinya akan berguna untuk normalisasi Score terhadap panjang
        # dokumen saat menghitung score dengan TF-IDF atau BM25

    def __enter__(self):
        """
        Memuat semua metadata ketika memasuki context.
        Metadata:
            1. Dictionary ---> postings_dict
            2. iterator untuk List yang berisi urutan term yang masuk ke
                index saat konstruksi. ---> term_iter
            3. doc_length, sebuah python's dictionary yang berisi key = doc id, dan
                value berupa banyaknya token dalam dokumen tersebut (panjang dokumen).
                Berguna untuk normalisasi panjang saat menggunakan TF-IDF atau BM25
                scoring regime; berguna untuk untuk mengetahui nilai N saat hitung IDF,
                dimana N adalah banyaknya dokumen di koleksi

        Metadata disimpan ke file dengan bantuan library "pickle"

        Perlu memahani juga special method __enter__(..) pada Python dan juga
        konsep Context Manager di Python. Silakan pelajari link berikut:

        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        # Membuka index file
        self.index_file = open(self.index_file_path, 'rb+')

        # Kita muat postings dict dan terms iterator dari file metadata
        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms, self.doc_length = pickle.load(f)
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Menutup index_file dan menyimpan postings_dict dan terms ketika keluar context"""
        # Menutup index file
        self.index_file.close()

        # Menyimpan metadata (postings dict dan terms) ke file metadata dengan bantuan pickle
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length], f)


class InvertedIndexReader(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file.
    """

    def __iter__(self):
        return self

    def reset(self):
        """
        Kembalikan file pointer ke awal, dan kembalikan pointer iterator
        term ke awal
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()  # reset term iterator

    def __next__(self):
        """
        Class InvertedIndexReader juga bersifat iterable (mempunyai iterator).
        Silakan pelajari:
        https://stackoverflow.com/questions/19151/how-to-build-a-basic-iterator

        Ketika instance dari kelas InvertedIndexReader ini digunakan
        sebagai iterator pada sebuah loop scheme, special method __next__(...)
        bertugas untuk mengembalikan pasangan (term, postings_list, tf_list) berikutnya
        pada inverted index.

        PERHATIAN! method ini harus mengembalikan sebagian kecil data dari
        file index yang besar. Mengapa hanya sebagian kecil? karena agar muat
        diproses di memori. JANGAN MEMUAT SEMUA INDEX DI MEMORI!
        """
        # TODO
        try:
            term_id = next(self.term_iter)
            postings_list, tf_list = self.get_postings_list(term_id)
            return (term_id, postings_list, tf_list)
        except StopIteration:
            raise StopIteration


    def get_postings_list(self, term):
        """
        Kembalikan sebuah postings list (list of docIDs) beserta list
        of term frequencies terkait untuk sebuah term (disimpan dalam
        bentuk tuple (postings_list, tf_list)).

        PERHATIAN! method tidak boleh iterasi di keseluruhan index
        dari awal hingga akhir. Method ini harus langsung loncat ke posisi
        byte tertentu pada file (index file) dimana postings list (dan juga
        list of TF) dari term disimpan.
        """
        if term not in self.postings_dict:
            return ([], [])
        
        
        # print(f'debug term: {term}')
        start_position_in_file, _, length_in_bytes_of_postings_list, length_in_bytes_of_tf_list = self.postings_dict[term]
        self.index_file.seek(start_position_in_file) # Pindah ke address start_position
        postings_bytes = self.index_file.read(length_in_bytes_of_postings_list) # membaca bytes yang sesuai dari file 
        postings_list = self.postings_encoding.decode(postings_bytes) # Mendecode posting bytes agar menjadi list of docId (integer)
        tf_bytes = self.index_file.read(length_in_bytes_of_tf_list) 
        tf_list = self.postings_encoding.decode_tf(tf_bytes)# Mendecode TF bytes agar menjadi list of tf (integer)
        
        return (postings_list, tf_list)

    def get_average_document_length(self):
        """
        Method untuk menghitung rata-rata panjang dokumen dalam collections.

        Returns
        -------
        float
            Rata-rata panjang dokumen dalam collections.
        """
        total_length = sum(self.doc_length.values())
        num_docs = len(self.doc_length)
        return total_length / num_docs if num_docs > 0 else 0.0
    
    
    def get_document_scores(self, query_terms, doc_id, term_id_map):
        """
        Computes scores for a specific document given a query.
        Returns BM25 and TF-IDF scores.

        Parameters
        ----------
        query_terms: List[str]
            Tokenized query terms (stemmed).
        doc_id: int
            Document ID for which scores are calculated.

        Returns
        -------
        tuple(float, float)
            BM25 and TF-IDF scores for the document.
        """
        bm25_score = 0.0
        tfidf_score = 0.0
        N = len(self.doc_length)
        avdl = self.get_average_document_length()

        for term in query_terms:
            term_id = term_id_map[term] 
            if term_id is None or term_id not in self.postings_dict:
                continue

            postings_list, tf_list = self.get_postings_list(term_id)
            df = self.postings_dict[term_id][1]  # Document frequency
            idf = math.log10((N - df + 0.5) / (df + 0.5))  # BM25 IDF

            if doc_id in postings_list:
                index = postings_list.index(doc_id)
                tf = tf_list[index]
                dl = self.doc_length.get(doc_id, 1)  # Avoid division by zero
                bm25_tf = ((1.2 + 1) * tf) / (1.2 * ((1 - 0.75) + 0.75 * (dl / avdl)) + tf)
                bm25_score += idf * bm25_tf

                # TF-IDF
                tfidf_tf = 1 + math.log10(tf) if tf > 0 else 0
                tfidf_score += idf * tfidf_tf

        return bm25_score, tfidf_score

class InvertedIndexWriter(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya menulis secara
    efisien Inverted Index yang disimpan di sebuah file.
    """

    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """
        Menambahkan (append) sebuah term, postings_list, dan juga TF list 
        yang terasosiasi ke posisi akhir index file.

        Method ini melakukan 4 hal:
        1. Encode postings_list menggunakan self.postings_encoding (method encode),
        2. Encode tf_list menggunakan self.postings_encoding (method encode_tf),
        3. Menyimpan metadata dalam bentuk self.terms, self.postings_dict, dan self.doc_length.
           Ingat kembali bahwa self.postings_dict memetakan sebuah termID ke
           sebuah 4-tuple: - start_position_in_index_file
                           - number_of_postings_in_list
                           - length_in_bytes_of_postings_list
                           - length_in_bytes_of_tf_list
        4. Menambahkan (append) bystream dari postings_list yang sudah di-encode dan
           tf_list yang sudah di-encode ke posisi akhir index file di harddisk.

        Jangan lupa update self.terms dan self.doc_length juga ya!

        SEARCH ON YOUR FAVORITE SEARCH ENGINE:
        - Anda mungkin mau membaca tentang Python I/O
          https://docs.python.org/3/tutorial/inputoutput.html
          Di link ini juga bisa kita pelajari bagaimana menambahkan informasi
          ke bagian akhir file.
        - Beberapa method dari object file yang mungkin berguna seperti seek(...)
          dan tell()

        Parameters
        ----------
        term:
            term atau termID yang merupakan unique identifier dari sebuah term
        postings_list: List[Int]
            List of docIDs dimana term muncul
        tf_list: List[Int]
            List of term frequencies
        """
        # TODO
        # Encode postings and term frequencies
        encoded_postings = self.postings_encoding.encode(postings_list)
        encoded_tf = self.postings_encoding.encode_tf(tf_list)

        # Update term list and postings dictionary
        self.terms.append(term)
        start_position = self.index_file.tell() # mencari current position pada file stream yang akan dijadikan start position dari postings dict
        postings_length = len(postings_list)
        postings_bytes_length = len(encoded_postings)
        tf_bytes_length = len(encoded_tf)
        self.postings_dict[term] = (start_position, postings_length, postings_bytes_length, tf_bytes_length)

        # Update document lengths
        for doc_id, tf in zip(postings_list, tf_list):
            self.doc_length[doc_id] = self.doc_length.get(doc_id, 0) + tf

        # Menulis ke index file
        self.index_file.write(encoded_postings)
        self.index_file.write(encoded_tf)

    

if __name__ == "__main__":

    from compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1, 2], "terms salah"
        assert index.doc_length == {
            2: 2, 3: 38, 4: 25, 5: 56, 8: 3, 10: 30}, "doc_length salah"
        assert index.postings_dict == {1: (0,
                                           5,
                                           len(VBEPostings.encode(
                                               [2, 3, 4, 8, 10])),
                                           len(VBEPostings.encode_tf([2, 4, 2, 3, 30]))),
                                       2: (len(VBEPostings.encode([2, 3, 4, 8, 10])) + len(VBEPostings.encode_tf([2, 4, 2, 3, 30])),
                                           3,
                                           len(VBEPostings.encode([3, 4, 5])),
                                           len(VBEPostings.encode_tf([34, 23, 56])))}, "postings dictionary salah"
        
        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(
            len(VBEPostings.encode([3, 4, 5])))) == [3, 4, 5], "terdapat kesalahan"
        assert VBEPostings.decode_tf(index.index_file.read(
            len(VBEPostings.encode_tf([34, 23, 56])))) == [34, 23, 56], "terdapat kesalahan"
