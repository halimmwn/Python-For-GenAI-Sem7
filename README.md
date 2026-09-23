Repository ini digunakan untuk menyimpan source code, hasil percobaan, dan dokumentasi praktikum.

## Identitas Mahasiswa

| Keterangan | Data |
|---|---|
| **Nama** | Mohammad Satria Halim Wicakasana |
| **Kelas** | 4 D4 Teknologi Rekayasa Multimedia - A |
| **NRP** | 5323600010 |

## Deskripsi

Repository ini berisi materi dan percobaan praktikum **Python for Generative AI**, mulai dari dasar pemrograman Python, integrasi API Large Language Model (LLM), prompt engineering, hingga implementasi model AI menggunakan layanan seperti **OpenAI, Anthropic, dan OpenRouter**.

Setiap percobaan disusun berdasarkan bab pembelajaran agar source code lebih terstruktur, mudah dipahami, dan dapat digunakan sebagai dokumentasi selama proses praktikum.

## Struktur Repository

```text
Python For GenAI - Sem7/
│
├── .venv/                         # Virtual environment Python
│
├── Bab_01_Python Core for AI/     # Dasar Python untuk pengembangan AI
│
├── Bab_02_LLM APIs & Prompting/    # API LLM dan Prompt Engineering
│
├── .env                            # API Key dan environment variables
├── .gitattributes                  # Konfigurasi Git attributes
├── .gitignore                      # File/folder yang diabaikan Git
├── README.md                       # Dokumentasi repository
└── requirements.txt                # Daftar dependency Python

## Struktur Repository

Pastikan **Git** dan **Python** sudah terinstall di komputer.

### 1. Clone Repository

Buka Terminal, PowerShell, atau terminal di Visual Studio Code, kemudian jalankan:

```bash
git clone <URL_REPOSITORY_GITHUB>
```

Contoh:

```bash
git clone https://github.com/username/Python-For-GenAI-Sem7.git
```

### 2. Masuk ke Folder Project

```bash
cd Python-For-GenAI-Sem7
```

### 3. Membuat Virtual Environment

Buat virtual environment menggunakan:

```bash
python -m venv .venv
```

Jika menggunakan Python Launcher di Windows:

```bash
py -m venv .venv
```

### 4. Aktifkan Virtual Environment

Untuk **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

Untuk **Windows CMD**:

```cmd
.venv\Scripts\activate
```

Jika berhasil, terminal akan menampilkan `(.venv)` seperti berikut:

```text
(.venv) PS C:\...\Python-For-GenAI-Sem7>
```

### 5. Install Dependencies

Install seluruh library yang terdapat pada `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 6. Membuat File Environment

Buat file baru dengan nama:

```text
.env
```

Kemudian isi API key yang diperlukan. Contoh:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

> **Catatan:** Jangan upload file `.env` ke GitHub karena berisi API key yang bersifat rahasia.

Pastikan `.gitignore` memiliki:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

### 7. Menjalankan Program

Masuk ke folder percobaan yang ingin dijalankan atau jalankan file Python secara langsung:

```bash
python "Bab_02_LLM APIs & Prompting/nama_file.py"
```

Atau pada Windows:

```bash
py "Bab_02_LLM APIs & Prompting/nama_file.py"
```

### 8. Membuka Project di Visual Studio Code

Jika perintah `code` sudah tersedia:

```bash
code .
```

Kemudian pilih Python interpreter dari virtual environment `.venv` melalui:

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ .venv
```