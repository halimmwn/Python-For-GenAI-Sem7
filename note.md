.venv\Scripts\activate
-m pip install --upgrade pip

## 10.7_PresistentStorage.py
- **Fungsi Utama**: Mendemonstrasikan penyimpanan (*persistence*) Vector Store Index ke media lokal dalam format JSON dan memuatnya kembali (*reload*) ke memori tanpa perlu melakukan proses embedding ulang dokumen.
- **Alur Kerja**:
  1. Mengonfigurasi LLM (`gpt-4o-mini`) dan model embedding (`text-embedding-3-small`) via OpenRouter pada `Settings`.
  2. Menentukan lokasi direktori penyimpanan lokal melalui variabel `PERSIST_DIR` (`./llama_index_store`).
  3. Membentuk dokumen teks (`Document`) dan membangun indeks vektor awal dengan `VectorStoreIndex.from_documents()`.
  4. Menyimpan indeks dan konteks penyimpanan ke disk menggunakan `index.storage_context.persist()`.
  5. Memuat kembali indeks yang tersimpan dari disk menggunakan `StorageContext.from_defaults()` dan `load_index_from_storage()`.
  6. Membuat query engine dari indeks yang dimuat ulang untuk menjawab pertanyaan pengujian tanpa kalkulasi ulang embedding.
- **Catatan Struktur Folder `llama_index_store`**:
  - Folder ini terbuat secara otomatis saat pemanggilan `persist()`.
  - Berisi file-file JSON penyimpan state:
    - `default__vector_store.json`: Menyimpan data vektor embedding hasil kalkulasi model.
    - `docstore.json`: Menyimpan teks dokumen asli, node chunk, dan metadata.
    - `index_store.json`: Menyimpan ID dan struktur relasi indeks.
    - `graph_store.json` & `image__vector_store.json`: Konteks bawaan untuk relasi graf dan multimodal.
  - **Tujuan Utama**: Efisiensi biaya (*cost*) dan waktu, sehingga aplikasi tidak perlu memanggil API embedding ulang setiap kali dijalankan.

## 10.8_CustomQuery.py
- **Fungsi Utama**: Mendemonstrasikan perakitan query pipeline kustom menggunakan `RetrieverQueryEngine` dengan membandingkan mode sintesis respon yang berbeda (`ResponseMode.COMPACT` dan `ResponseMode.TREE_SUMMARIZE`).
- **Alur Kerja**:
  1. Mengonfigurasi LLM (`gpt-4o-mini`) dan model embedding (`text-embedding-3-small`) via OpenRouter pada `Settings`.
  2. Menyiapkan daftar dokumen teks tentang algoritma pencarian vektor dan membangun `VectorStoreIndex`.
  3. Membentuk retriever berbasis vektor (`VectorIndexRetriever`) dengan `similarity_top_k=2`.
  4. Melakukan iterasi terhadap beberapa mode sintesis respon (`ResponseMode.COMPACT` dan `ResponseMode.TREE_SUMMARIZE`) menggunakan `get_response_synthesizer()`.
  5. Menggabungkan retriever dan synthesizer ke dalam `RetrieverQueryEngine` untuk mengeksekusi query dan membandingkan hasil sintesis teks.


## 11.2.1_ChromaDb.py
- **Fungsi Utama**: Mendemonstrasikan penggunaan ChromaDB sebagai basis data vektor persisten lokal (`PersistentClient`) untuk menyimpan koleksi dokumen, melakukan pra-kalkulasi embedding via OpenRouter, dan mengeksekusi semantic similarity search berbasis jarak kosinus (*cosine distance*).
- **Alur Kerja**:
  1. Mengonfigurasi OpenAI client dengan endpoint OpenRouter untuk embedding teks model `text-embedding-3-small`.
  2. Menginisialisasi `chromadb.PersistentClient` dengan lokasi penyimpanan lokal `./chroma_db`.
  3. Membuat atau membuka koleksi (`ai_knowledge_base`) dengan konfigurasi metrik jarak `{"hnsw:space": "cosine"}`.
  4. Menghasilkan vektor embedding untuk daftar dokumen menggunakan fungsi helper `embed()`.
  5. Menyimpan ID, dokumen, embedding, dan metadata ke dalam koleksi ChromaDB menggunakan `collection.add()`.
  6. Mengubah teks pertanyaan (*query*) menjadi vektor embedding dan mencari 3 tetangga terdekat (`n_results=3`).
  7. Menampilkan dokumen yang paling relevan beserta nilai jarak kedekatannya (*distance*) dan informasi metadata.


## 11.2.2_MetadataFiltering.py
- **Fungsi Utama**: Mendemonstrasikan operasi lanjutan pada koleksi ChromaDB, mencakup penyaringan berdasarkan metadata (*pre-filtering*), pembaruan data dokumen dan vektornya (*update*), serta penghapusan dokumen (*delete*).
- **Alur Kerja**:
  1. Membuka koneksi ke penyimpanan persisten `./chroma_db` dan memuat koleksi `ai_knowledge_base`.
  2. Menghasilkan vektor embedding untuk teks pertanyaan (*query*).
  3. Mengeksekusi pencarian kemiripan semantik dengan filter metadata menggunakan operator `where={"category": {"$eq": "technique"}}`.
  4. Memperbarui dokumen lama (`doc_0`) dengan konten teks dan vektor embedding baru menggunakan `collection.update()`.
  5. Menghapus dokumen tertentu (`doc_4`) dari koleksi menggunakan `collection.delete()`.
  6. Menampilkan jumlah akhir dokumen yang tersisa di dalam koleksi.


## 11.3_Pinecone.py
- **Fungsi Utama**: Mendemonstrasikan integrasi basis data vektor terkelola berbasis cloud (Pinecone Serverless) untuk manajemen indeks jarak jauh, *upsert* vektor embedding beserta metadata, dan pencarian kemiripan semantik dengan filter metadata.
- **Alur Kerja**:
  1. Menyiapkan helper embedding menggunakan OpenAI client yang terhubung ke OpenRouter (`text-embedding-3-small`).
  2. Menghubungkan client Pinecone melalui API key yang tersimpan di environment (`PINECONE_API_KEY`).
  3. Memeriksa dan membuat indeks `ai-knowledge-base` berspesifikasi serverless AWS (`dimension=1536`, `metric="cosine"`).
  4. Melakukan pra-kalkulasi embedding dokumen dan mengunggah (*upsert*) data berupa ID, nilai vektor (*values*), dan metadata teks ke Pinecone.
  5. Memeriksa statistik kondisi indeks menggunakan `index.describe_index_stats()`.
  6. Mengeksekusi pencarian vektor semantik dengan parameter `top_k=3` dan menampilkan teks hasil pencocokan beserta skor kemiripannya.
  7. Menerapkan filter metadata bersyarat (`{"source": {"$eq": "guide"}}`) pada proses query.


## 11.4_Weavite.py
- **Fungsi Utama**: Mendemonstrasikan pola implementasi pustaka Weaviate v4 (`weaviate-client`) sebagai basis data vektor berfitur lengkap yang mendukung pencarian semantik murni (*vector search*), pencarian kata kunci (*BM25*), dan pencarian hibrida (*hybrid search*).
- **Alur Kerja & Komponen Kunci**:
  1. **Deployment & Koneksi Client**:
     - *Local Docker*: Menggunakan `weaviate.connect_to_local()` untuk instance lokal mandiri (`cr.weaviate.io/semitechnologies/weaviate:latest`).
     - *Weaviate Cloud*: Menggunakan `weaviate.connect_to_weaviate_cloud()` dengan otentikasi API Key (`WEAVIATE_API_KEY`).
  2. **Pembuatan Skema & Koleksi (`collections.create`)**:
     - Menggunakan definisi properti terstruktur kuat (*strongly typed schema*) dengan `wvc.config.Property` dan `wvc.config.DataType.TEXT`.
     - Menyetel `vectorizer_config=wvc.config.Configure.Vectorizer.none()` untuk memasukkan vektor embedding sendiri (*bring-your-own-vectors*) dari OpenAI/OpenRouter.
  3. **Penyisipan Objek secara Batch Dinamis (`collection.batch.dynamic`)**:
     - Mengunggah data dalam jumlah banyak secara efisien dengan auto-batching dinamis yang mengoptimalkan memori dan jaringan.
  4. **Mode Pencarian Lanjutan**:
     - **Vector Search (`near_vector`)**: Menemukan dokumen yang paling dekat secara kosinus terhadap vektor input `near_vector`.
     - **Hybrid Search (`hybrid`)**: Menggabungkan algoritma pencarian teks tradisional (BM25) dengan kemiripan vektor. Bobot diatur oleh parameter `alpha` (skala 0.0 sampai 1.0: `0` = murni teks BM25, `1` = murni vektor, `0.5` = kombinasi seimbang 50:50).
     - **Metadata Filtering (`Filter.by_property`)**: Melakukan penyaringan dokumen berdasarkan kriteria metadata (misalnya properti `category == "ai"`) saat eksekusi query.
- **Kelebihan Utama**:
  - Menyediakan *Hybrid Search* bawaan terbaik untuk mengatasi kelemahan pencarian vektor murni pada kata kunci langka/akronim.
  - Mendukung deployment lokal (gratis via Docker) maupun cloud terkelola.
