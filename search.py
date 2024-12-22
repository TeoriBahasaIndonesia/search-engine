from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='data',
                          postings_encoding=VBEPostings,
                          output_dir='index')

# query = input("Masukkan query Anda: ")

# TODO
# silakan dilanjutkan sesuai contoh interaksi program pada dokumen soal
print("Selamat datang di Search Engine sederhana.")
print("Ketik 'exit' atau kosongkan input untuk keluar.\n")

while True:
    query = input("Masukkan query Anda: ")

    if not query or query.lower() == 'exit':
        print("Terima kasih telah menggunakan search engine ini.")
        break

    # Dapatkan rekomendasi untuk query auto-completion
    last_token = query.split()[-1]
    recommendations = BSBI_instance.get_query_recommendations(last_token)

    # Buat daftar rekomendasi dimulai dengan query asli
    suggestions = [query]
    # Tambahkan rekomendasi dengan melengkapi last_token
    for rec in recommendations:
        # Jika rekomendasi sama dengan last_token, skip untuk menghindari duplikasi
        if rec != '':
            completed_token = last_token + rec
            suggestion = ' '.join(query.split()[:-1] + [completed_token])
            suggestions.append(suggestion)

    # Batasi hingga total 6 rekomendasi (1 asli + 5 dari trie)
    suggestions = suggestions[:6]

    if suggestions:
        print("\nRekomendasi query yang sesuai:")
        for idx, suggestion in enumerate(suggestions, 1):
            print(f"{idx}. {suggestion}")
        choice = input("\nMasukkan nomor query yang Anda maksud: ")
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(suggestions):
                chosen_query = suggestions[choice - 1]
                print(f"\nPilihan Anda adalah '{chosen_query}'.")
                query = chosen_query
            else:
                print("Pilihan tidak valid, melanjutkan dengan query asli.")
        else:
            print("Melanjutkan dengan query asli.")
    else:
        print("Tidak ada rekomendasi query yang tersedia.")

    # Retrieval dengan TF-IDF (DaaT)
    results = BSBI_instance.retrieve_tfidf_daat(query, k=50)
    if results:
        print("\nHasil pencarian:")
        for score, doc_name in results:
            print(f"{doc_name} {score}")
    else:
        print("Tidak ada hasil yang ditemukan.")

    print("\n---\n")
    