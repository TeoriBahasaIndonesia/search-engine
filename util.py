from turtle import pos


class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        # TODO
        return len(self.id_to_str)

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        # TODO
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str) - 1
        
        return self.str_to_id[s]

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        # TODO
        return self.id_to_str[i]

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        # TODO
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        return None


def merge_and_sort_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparable, int)]
        Penggabungan yang sudah terurut
    """
    # TODO
    
    result = []
    
    i, j = 0, 0 
    
    # Menambahkan elemen-elemen pada posts_tfs1 dan posts_tfs2 secara terurut berdasarkan document id
    while i < len(posts_tfs1) and j < len(posts_tfs2):
        post_id_1, post_id_1_tf = posts_tfs1[i][0], posts_tfs1[i][1] 
        post_id_2, post_id_2_tf = posts_tfs2[j][0], posts_tfs2[j][1]
        
        if post_id_1 == post_id_2:
            tfs_accumulation = post_id_1_tf + post_id_2_tf
            result.append((post_id_1, tfs_accumulation))
            i += 1
            j += 1
        
        elif post_id_1 < post_id_2:
            result.append((post_id_1, post_id_1_tf))
            i += 1
        else:
            result.append((post_id_2, post_id_2_tf))
            j += 1
    
    # Menambahkan yang tersisa pada posts_tfs1 
    while i < len(posts_tfs1):
        post_id_1, post_id_1_tf = posts_tfs1[i][0], posts_tfs1[i][1] 
        result.append((post_id_1, post_id_1_tf))
        i += 1
    
    
    # Menambahkan yang tersisa pada posts_tfs2
    while j < len(posts_tfs2):
        post_id_2, post_id_2_tf = posts_tfs2[j][0], posts_tfs2[j][1] 
        result.append((post_id_2, post_id_2_tf))
        j += 1
        
    
    return result



if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()

    assert [term_id_map[term]
            for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname]
            for docname in docs] == [0, 1, 2], "docs_id salah"

    assert merge_and_sort_posts_and_tfs([(1, 34), (3, 2), (4, 23)],
                                        [(1, 11), (2, 4), (4, 3), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "merge_and_sort_posts_and_tfs salah"
