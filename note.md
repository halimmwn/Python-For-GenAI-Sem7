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


## 12.2_PipelineAPI.py
- **Fungsi Utama**: Mendemonstrasikan penggunaan Hugging Face `pipeline` API tingkat tinggi (*high-level*) untuk klasifikasi teks dan analisis sentimen (*sentiment analysis*) secara lokal menggunakan arsitektur Transformer.
- **Alur Kerja**:
  1. Mengimpor fungsi `pipeline` dari pustaka `transformers`.
  2. Menginisialisasi pipeline untuk tugas `"text-classification"` menggunakan model pra-latih (*pre-trained*) `"distilbert-base-uncased-finetuned-sst-2-english"` pada perangkat CPU (`device=-1`).
  3. Menyiapkan daftar kalimat uji dengan berbagai nada sentimen (positif, negatif, dan netral/moderat).
  4. Menjalankan inferensi klasifikasi teks secara batch menggunakan `classifier(texts)`.
  5. Menampilkan hasil prediksi berupa label sentimen (`POSITIVE`/`NEGATIVE`) dan skor probabilitas keyakinan (*confidence score*).
  6. *Catatan*: Memerlukan runtime PyTorch (`torch`) terpasang pada virtual environment untuk memuat bobot model lokal.


## 12.2.1.py
- **Fungsi Utama**: Mendemonstrasikan penggunaan Hugging Face `pipeline` untuk tugas *Named Entity Recognition* (NER) guna mengidentifikasi dan mengelompokkan entitas bernama (nama orang, organisasi, lokasi) dari teks secara otomatis.
- **Alur Kerja**:
  1. Mengimpor fungsi `pipeline` dari modul `transformers`.
  2. Mengonfigurasi pipeline dengan `task="ner"` menggunakan model pra-latih BERT (`"dbmdz/bert-large-cased-finetuned-conll03-english"`).
  3. Menggunakan parameter `aggregation_strategy="simple"` untuk menggabungkan sub-token kembali menjadi satu kata/frasa entitas yang utuh.
  4. Menerapkan model ke contoh kalimat input yang memuat berbagai jenis entitas.
  5. Melakukan iterasi terhadap hasil ekstraksi dan menampilkan nama entitas, kelompok kategori (`ORG`, `LOC`, `PER`), serta skor probabilitasnya (*confidence score*).


## 12.2.2.py
- **Fungsi Utama**: Mendemonstrasikan penggunaan Hugging Face `pipeline` untuk tugas pembangkitan teks (*text generation*) secara lokal pada CPU menggunakan model bahasa kecil berbasis arsitektur GPT-2 (`distilgpt2`).
- **Alur Kerja**:
  1. Mengimpor fungsi `pipeline` dari modul `transformers`.
  2. Mengonfigurasi pipeline `task="text-generation"` dengan model `"distilgpt2"` pada perangkat CPU (`device=-1`).
  3. Memasukkan teks awalan (*prompt*) `"The future of AI agents is"`.
  4. Mengatur parameter generasi teks:
     - `max_new_tokens=50`: Jumlah maksimal token baru yang dibangkitkan.
     - `num_return_sequences=2`: Menghasilkan 2 variasi teks berbeda.
     - `temperature=0.8` dan `do_sample=True`: Mengaktifkan mode sampling probabilistik untuk hasil yang lebih variatif.
     - `pad_token_id=50256`: Mengatasi peringatan *padding token* pada GPT-2.
  5. Melakukan iterasi dan mencetak seluruh variasi teks hasil generasi model ke terminal.


## 12.2.3.py
- **Fungsi Utama**: Mendemonstrasikan penggunaan Hugging Face `pipeline` untuk tugas peringkasan dokumen teks (*text summarization*) secara otomatis menggunakan model arsitektur encoder-decoder `facebook/bart-large-cnn`.
- **Alur Kerja**:
  1. Mengimpor fungsi `pipeline` dari pustaka `transformers`.
  2. Menginisialisasi pipeline untuk tugas `"summarization"` dengan model `"facebook/bart-large-cnn"` pada CPU (`device=-1`).
  3. Memasukkan teks artikel panjang mengenai pengaruh model bahasa besar (*Large Language Models*) pada pengembangan perangkat lunak.
  4. Menjalankan proses peringkasan dengan parameter batas panjang (`min_length=20`, `max_length=60`) serta `do_sample=False` untuk menghasilkan ringkasan yang ringkas, faktual, dan deterministik.
  5. Menampilkan hasil ringkasan teks akhir (`summary_text`) ke terminal.


## 12.3_Tokenizers.py
- **Fungsi Utama**: Mendemonstrasikan operasi tokenisasi menggunakan `AutoTokenizer` dari Hugging Face, meliputi konversi teks ke token ID (*encoding*), rekontruksi token ke teks (*decoding*), inspeksi sub-token individual (BPE), pembuatan batch tensor dengan *padding* dan *truncation*, serta fungsi penghitung token prompt.
- **Alur Kerja**:
  1. Memuat tokenizer pra-latih model `"gpt2"` menggunakan `AutoTokenizer.from_pretrained()`.
  2. Mengonversi teks string menjadi daftar token ID menggunakan `tokenizer.encode()`.
  3. Mengembalikan token ID ke teks string semula menggunakan `tokenizer.decode()`.
  4. Menginspeksi pecahan sub-token BPE (seperti karakter spasi `Ġ`) menggunakan `tokenizer.convert_ids_to_tokens()`.
  5. Melakukan encoding batch pada beberapa kalimat dengan `padding=True`, `truncation=True`, dan `return_tensors="pt"` untuk menghasilkan tensor PyTorch (`input_ids` dan `attention_mask`) yang siap diinput ke model.
  6. Mendefinisikan fungsi utilitas `count_tokens()` untuk menghitung total token suatu prompt sebelum dikirimkan ke model.


## 12.4_AutoModel.py
- **Fungsi Utama**: Mendemonstrasikan pemuatan dan kontrol inferensi model bahasa kausal (*Causal Language Model*) secara langsung tingkat rendah (*low-level*) menggunakan arsitektur `AutoModelForCausalLM` dan `AutoTokenizer`, bukan melalui abstraksi `pipeline`.
- **Alur Kerja**:
  1. Memuat tokenizer dan model pra-latih `"distilgpt2"` dengan tipe data komputasi `torch.float32` untuk eksekusi CPU.
  2. Mengaktifkan mode inferensi menggunakan `model.eval()` untuk menonaktifkan layer pelatihan (seperti dropout).
  3. Melakukan tokenisasi prompt input menjadi format tensor PyTorch (`return_tensors="pt"`).
  4. Menjalankan fungsi `model.generate()` di dalam blok `torch.no_grad()` dengan strategi sampling decoding (`temperature=0.8`, `top_p=0.9`, `max_new_tokens=40`, dan `pad_token_id=tokenizer.eos_token_id`).
  5. Mengonversi tensor token ID kembali menjadi teks manusia dengan `tokenizer.decode()` dan mengabaikan token khusus (*special tokens*).


## 12.5_PeftLoraFinetunning.py
- **Fungsi Utama**: Mendemonstrasikan pipeline lengkap fine-tuning model bahasa berbasis parameter efisien menggunakan LoRA (*Low-Rank Adaptation*) dan pustaka PEFT, mulai dari pemuatan model dasar, konfigurasi adapter, penyiapan dataset, pelatihan dengan Hugging Face `Trainer`, penyimpanan adapter ringan, hingga inferensi model hasil adaptasi.
- **Alur Kerja**:
  1. **Memuat Model Dasar**: Menginisialisasi `"distilgpt2"` beserta tokenizer dan menyamakan `pad_token` dengan `eos_token`.
  2. **Konfigurasi & Pemasangan LoRA**:
     - Mengonfigurasi `LoraConfig` (`r=8`, `lora_alpha=32`, `target_modules=["c_attn"]`, `lora_dropout=0.1`, `bias="none"`).
     - Membungkus model dasar dengan `get_peft_model()` dan mencetak rasio parameter yang dilatih (~0.18% - 0.36% dari total bobot model).
  3. **Penyiapan & Tokenisasi Dataset**:
     - Menyiapkan contoh dataset teks pasangan Question-Answer.
     - Melakukan tokenisasi batch dengan batas panjang `max_length=128` dan padding konsisten.
     - Menyalin `input_ids` ke dalam `labels` untuk kalkulasi loss bahasa kausal secara otomatis.
  4. **Konfigurasi Pelatihan (`TrainingArguments`)**:
     - Mengatur direktori output (`./lora_output`), `num_train_epochs=3`, batch size per device, `gradient_accumulation_steps=4`, dan learning rate `2e-4`.
  5. **Proses Pelatihan (`Trainer`)**:
     - Menginisialisasi `Trainer` dengan model PEFT dan dataset ter-tokenisasi, lalu menjalankan `trainer.train()`.
  6. **Penyimpanan Adapter Ringan**:
     - Menyimpan hanya bobot adapter LoRA dan tokenizer ke `./lora_adapter` (hanya berukuran beberapa megabyte, bukan puluhan gigabyte bobot model penuh).
  7. **Pemuatan Adapter & Inferensi**:
     - Memuat kembali model dasar bersih, menggabungkannya dengan adapter via `PeftModel.from_pretrained()`, lalu menjalankan pengujian inferensi generasi jawaban teks dengan `torch.no_grad()`.
- **Catatan Evaluasi Hasil & Penjelasan Warning**:
  - **Teks Hasil Generasi Berulang (*"RAG is a term for a term..."*)**:
    - Sangat wajar (*normal*) untuk skala demo ini karena menggunakan model sangat kecil (`distilgpt2` ~82M params), dataset mini (5 kalimat), 3 epoch latihan cepat pada CPU, dan strategi *greedy decoding*. Pada model skala produksi (seperti LLaMA-3) dengan dataset ribuan baris, teks akan koheren dan fasih.
  - **Penjelasan Peringatan (*Warning*)**:
    - `fan_in_fan_out`: GPT-2 menggunakan layer linear `Conv1D` khusus, dan PEFT secara otomatis mengaktifkan `fan_in_fan_out=True` agar perkalian matriks LoRA cocok.
    - `pin_memory`: Fitur untuk akselerasi transfer data ke GPU; diabaikan secara aman saat dijalankan di CPU.
    - `loss_type=None`: Trainer otomatis menggunakan formula loss standar untuk model kausal (`ForCausalLMLoss`).


## 13.1_CoreConcept.py
- **Fungsi Utama**: Mendemonstrasikan konsep dasar arsitektur LangGraph untuk alur kerja agen berbasis state (*stateful graph*), mencakup pembuatan skema State dengan reducer, node pemrosesan, percabangan bersyarat (*conditional edges*), dan siklus eksekusi graf.
- **Alur Kerja**:
  1. Mendefinisikan skema state `AgentState` berbasis `TypedDict` dengan reducer `Annotated[list[str], operator.add]` agar pesan baru digabungkan (*appended*) secara otomatis ke riwayat sebelumnya.
  2. Menyiapkan dua fungsi node:
     - `think_node`: Melakukan simulasi tahapan berpikir dan menambah nilai `step_count`.
     - `answer_node`: Mengembalikan hasil kesimpulan akhir ke dalam state `final_answer`.
  3. Membuat fungsi logika percabangan `should_continue()` yang menentukan rute alur: melakukan looping kembali ke `think` jika `step_count < 2`, atau berpindah ke node `answer`.
  4. Merakit graf dengan `StateGraph`, menghubungkan titik masuk `START`, mendaftarkan *conditional edges*, dan mengalirkan node akhir ke `END`.
  5. Mengompilasi graf menjadi runnable application (`graph.compile()`) dan mengeksekusinya dengan `app.invoke()`.


## 13.2_RecAgent.py
- **Fungsi Utama**: Mendemonstrasikan implementasi arsitektur agen ReAct (*Reason + Act*) berbasis LangGraph yang memiliki kemampuan pemanggilan alat (*tool calling*), mengeksekusi perhitungan melalui alat lokal, dan melakukan siklus penalaran berulang (*cyclic reasoning loop*) hingga menghasilkan jawaban final.
- **Alur Kerja**:
  1. Mendefinisikan fungsi alat lokal (*tools*):
     - `calculator`: Menghitung ekspresi matematika secara aman dengan modul `math`.
     - `token_cost_lookup`: Mengambil data harga tarif token model LLM per 1 juta token.
  2. Menyiapkan skema fungsi JSON (`TOOL_SCHEMAS`) dan mengikatnya ke model `ChatOpenAI` (via OpenRouter) menggunakan `.bind_tools()`.
  3. Mengonfigurasi `ReActState` menggunakan `operator.add` agar riwayat percakapan dan respon tool terus terakumulasi.
  4. Menyiapkan dua node utama dan conditional edge:
     - `agent_node`: Memanggil LLM untuk menganalisis masalah dan memutuskan apakah perlu memanggil alat.
     - `tool_node`: Mengeksekusi alat yang diminta LLM dan mengembalikan hasilnya dalam bentuk `ToolMessage`.
     - `should_use_tools`: Memeriksa apakah respon LLM meminta pemanggilan alat (beralih ke node `tools`) atau selesai (beralih ke `END`).
  5. Membangun struktur graf siklik: `agent` $\rightarrow$ (jika ada tool call) $\rightarrow$ `tools` $\rightarrow$ kembali ke `agent` $\rightarrow$ (jika selesai) $\rightarrow$ `END`.
  6. Menguji agen dengan studi kasus kompleks perhitungan estimasi biaya API harian, dan menampilkan kesimpulan jawaban final.


## 13.4_HumanInTheLoop.py
- **Fungsi Utama**: Mendemonstrasikan implementasi pola intervensi manusia (*Human-in-the-Loop* / HITL) pada LangGraph menggunakan sistem checkpoint (`MemorySaver`) dan fitur interupsi (`interrupt_before`) untuk menghentikan alur kerja otomatis sebelum aksi berisiko dieksekusi sampai mendapatkan konfirmasi/persetujuan.
- **Alur Kerja**:
  1. Mendefinisikan `ReviewState` untuk melacak teks rencana (`plan`), status persetujuan (`approved`), hasil eksekusi (`result`), dan riwayat pesan (`messages`).
  2. Menyiapkan node `plan_node` untuk perancangan alur dan `execute_node` yang memvalidasi bendera `approved` sebelum memproses tugas.
  3. Menginisialisasi `MemorySaver()` untuk menyimpan snapshot state per sesi (`thread_id`) dan mengompilasi graf dengan jeda eksplisit: `interrupt_before=["execute"]`.
  4. Menjalankan eksekusi tahap pertama (`app.invoke()`) hingga alur otomatis terhenti tepat sebelum node eksekusi, lalu menampilkan rencana kerja ke konsol.
  5. Mensimulasikan persetujuan manusia dengan memperbarui state pada checkpoint thread aktif menggunakan `app.update_state(config, {"approved": True})`.
  6. Melanjutkan eksekusi graf yang tertunda dengan memanggil `app.invoke(None, config=config)` untuk menyelesaikan pipeline dan mencetak hasil akhir.


## 14.2_Agent.py
- **Fungsi Utama**: Mendemonstrasikan pendefinisian agen otonom berbasis peran (*role-based autonomous agents*) pada kerangka kerja CrewAI, mencakup penentuan `role`, `goal`, `backstory`, konfigurasi LLM (OpenRouter), dan hak delegasi antar-agen.
- **Alur Kerja**:
  1. Mengonfigurasi LLM menggunakan kelas `LLM` dari CrewAI yang diarahkan ke gateway OpenRouter (`openrouter/openai/gpt-4o-mini`).
  2. Membentuk agen spesialis riset (**AI Research Analyst**): fokus pada pencarian data akurat, dibatasi maksimal 5 iterasi (`max_iter=5`), dan tidak diperbolehkan mendelegasikan tugas (`allow_delegation=False`).
  3. Membentuk agen spesialis penulisan (**Technical Content Writer**): fokus menyusun materi teknis yang lugas dan terstruktur untuk pengembang.
  4. Membentuk agen peninjau kualitas (**Content Quality Reviewer**): fokus memeriksa akurasi fakta serta kesiapan publikasi, dengan izin delegasi aktif (`allow_delegation=True`) agar bisa meminta agen penulis melakukan revisi.
- **Catatan Mengapa Skrip Tidak Menghasilkan Output Terminal**:
  - **Tahap Deklarasi Objek**: Skrip ini murni merupakan tahap deklarasi inisialisasi objek `Agent` di memori komputer dan tidak memiliki pemanggilan fungsi `print()`.
  - **Arsitektur Tiga Tahap CrewAI**:
    1. **Agent (14.2)**: Menentukan *aktor/spesialis* yang terlibat beserta karakteristik dan model LLM-nya.
    2. **Task (14.3)**: Menentukan *pekerjaan spesifik* yang harus diselesaikan beserta ekspektasi keluarannya (*expected output*).
    3. **Crew & Kickoff**: Merakit kumpulan Agent dan Task ke dalam orkestrasi tim (`Crew`), lalu mengeksekusinya via `crew.kickoff()`. Output aktivitas, proses penalaran (*thoughts*), dan hasil kolaborasi agen baru akan muncul di terminal pada fase kickoff ini.
  - **Cara Verifikasi Manual**: Inisialisasi dapat dicek dengan menambahkan baris cetak properti peran di akhir skrip (misal: `print(researcher.role, writer.role, reviewer.role)`).


## 14.3_ToolsForAgent.py
- **Fungsi Utama**: Mendemonstrasikan pembuatan alat kustom (*custom tools*) pada framework CrewAI dengan cara menurunkan (*subclassing*) kelas `BaseTool` serta memanfaatkan validasi skema input Pydantic (`BaseModel`).
- **Alur Kerja**:
  1. Mendefinisikan skema validasi parameter input menggunakan Pydantic `BaseModel` (`TokenCostCalculatorInput` dan `ModelContextLookupInput`) agar LLM memahami tipe dan deskripsi argumen yang diharapkan.
  2. Mengimplementasikan `TokenCostCalculator(BaseTool)`:
     - Menyediakan metadata alat (`name`, `description`, `args_schema`).
     - Menghitung estimasi biaya komputasi API berbasis jumlah token input/output pada metode `_run()` dan mengembalikan hasilnya dalam bentuk string JSON.
  3. Mengimplementasikan `ModelContextLookup(BaseTool)`:
     - Melakukan pencarian spesifikasi batas token *context window* berdasarkan nama model LLM.
  4. Menginstansiasi objek alat (`cost_tool` dan `context_tool`) sehingga siap dipasangkan ke properti `tools` milik agen CrewAI.



## 14.4_Tasks.py
- **Fungsi Utama**: Mendemonstrasikan pendefinisian tugas (*Task*) pada kerangka kerja CrewAI, mencakup spesifikasi deskripsi instruksi (`description`), format luaran yang diharapkan (`expected_output`), penugasan ke agen pelaksana (`agent`), serta penyusunan dependensi data antar-tugas menggunakan parameter konteks (`context`).
- **Alur Kerja**:
  1. Mendefinisikan fungsi perakitan tugas `make_tasks(researcher, writer, reviewer)` yang menerima instans agen CrewAI.
  2. Membentuk tugas riset (**Research Task**):
     - Menugaskan agen peneliti (*AI Research Analyst*) untuk meriset perkembangan RAG (arsitektur, varian, tantangan produksi, dan studi kasus riil).
     - Menetapkan target luaran berupa dokumen Markdown terstruktur ber-header.
  3. Membentuk tugas penulisan (**Writing Task**):
     - Menugaskan agen penulis (*Technical Content Writer*) untuk menyusun panduan teknis pengembang (*developer guide*) berdasar hasil riset.
     - Menyertakan parameter `context=[research_task]`, sehingga luaran tugas riset diteruskan secara otomatis sebagai input bagi tugas penulisan.
  4. Membentuk tugas peninjauan kualitas (**Review Task**):
     - Menugaskan agen peninjau (*Content Quality Reviewer*) untuk mengaudit akurasi teknis, kelengkapan, dan kejelasan teks hasil tulisan.
     - Menyertakan parameter `context=[writing_task]`, sehingga draf panduan dari penulis menjadi input telaah bagi agen peninjau.
  5. Mengembalikan ketiga objek tugas (`research_task`, `writing_task`, `review_task`) yang telah terhubung secara sekuensial (*sequential task pipeline*).
  6. Menginisialisasi agen dan memvalidasi keterhubungan ketiga tugas secara lokal pada blok eksekusi utama.

## 14.5_CrewPuttingTogether.py
- **Fungsi Utama**: Mendemonstrasikan perakitan menyeluruh (*end-to-end integration*) sistem multi-agen menggunakan kelas `Crew` dengan alur eksekusi sekuensial (`Process.sequential`), menggabungkan agen berspesialisasi, alat kustom, dan serangkaian tugas berantai dengan transfer konteks otomatis.
- **Alur Kerja**:
  1. Mengonfigurasi model LLM (`openrouter/openai/gpt-4o-mini`) via gateway OpenRouter menggunakan kelas `crewai.LLM`.
  2. Menyiapkan alat kustom (`TokenCostCalculator` dan `ModelContextLookup`) untuk disematkan pada agen yang membutuhkan referensi komputasi.
  3. Mendefinisikan fungsi pembentuk tugas `make_tasks` yang menghubungkan luaran riset ke penulisan (`context=[research_task]`) dan luaran draf ke peninjauan (`context=[writing_task]`).
  4. Menginisialisasi tiga agen spesialis:
     - `researcher`: AI Research Analyst dilengkapi `context_tool` untuk mencari limit token model.
     - `writer`: Technical Content Writer dilengkapi `cost_tool` untuk estimasi biaya API.
     - `reviewer`: Content Quality Reviewer bertugas memvalidasi kelayakan akhir.
  5. Membangun objek orkestrasi `Crew` dengan parameter `process=Process.sequential` dan `memory=False`.
  6. Mengeksekusi seluruh pipeline kerja agen secara terurut menggunakan `crew.kickoff()` serta mencetak luaran individual dari masing-masing tugas (`task.output`).


## 14.6_HierarchicalProcess.py
- **Fungsi Utama**: Mendemonstrasikan orkestrasi tim multi-agen berbasis proses hierarki (`Process.hierarchical`) yang dipimpin oleh seorang Manajer (`manager_agent`), di mana manajer secara otomatis membedah tugas utama, mendelegasikan subtugas ke agen bawahan (*workers*), dan menyatukan hasil kerja sebelum dipublikasikan.
- **Alur Kerja**:
  1. Mengonfigurasi LLM OpenRouter (`gpt-4o-mini`) untuk agen manajer dan agen pelaksana.
  2. Mendefinisikan agen manajer (**AI Content Production Manager**) dengan hak delegasi aktif (`allow_delegation=True`) untuk mengoordinasikan tim dan mengaudit kualitas.
  3. Mendefinisikan dua agen bawahan (*workers*): `researcher` (riset data mendalam) dan `writer` (penulisan panduan teknis).
  4. Mendefinisikan tugas utama (`main_task`) tentang penyusunan *technical brief* basis data vektor untuk arsitektur RAG produksi.
  5. Membangun `Crew` dengan konfigurasi hierarkis:
     - `process=Process.hierarchical`
     - `manager_agent=manager`
     - `agents=[researcher, writer]` (*Catatan penting: agen manajer tidak boleh dimasukkan ke dalam daftar `agents` bawahan untuk menghindari validasi error CrewAI*).
  6. Menjalankan koordinasi tim otonom via `crew.kickoff()`, di mana manajer merencanakan, menugaskan riset dan penulisan ke pekerja, lalu mengembalikan laporan teknis final yang terstruktur.
