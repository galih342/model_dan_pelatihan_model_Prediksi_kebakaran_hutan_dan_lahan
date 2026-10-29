# 🔥 Prediksi Kebakaran Hutan dan Lahan Menggunakan TensorFlow

Proyek ini merupakan implementasi **Machine Learning menggunakan TensorFlow untuk memprediksi kebakaran hutan dan lahan (Karhutla) di Kalimantan Tengah** berdasarkan data historis dan fitur numerik yang tersedia pada dataset.

Model dibangun menggunakan arsitektur **Multilayer Perceptron (MLP)** untuk menyelesaikan permasalahan regresi. Proses pengembangan model mencakup pengumpulan dataset, preprocessing, normalisasi fitur, pembagian data, pelatihan model, evaluasi performa, dan penyimpanan model yang telah dilatih.

Dataset yang digunakan bersumber dari Hugging Face dan mencakup data periode 2012–2025.

## 📌 Daftar Isi

- [Tujuan Proyek](#-tujuan-proyek)
- [Anggota Kelompok](#-anggota-kelompok)
- [Dataset](#-dataset)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Alur Pengembangan Model](#-alur-pengembangan-model)
- [Arsitektur Model](#-arsitektur-model)
- [Konfigurasi Pelatihan](#-konfigurasi-pelatihan)
- [Hasil Eksperimen](#-hasil-eksperimen)
- [Struktur Repository](#-struktur-repository)
- [Cara Menjalankan Proyek](#-cara-menjalankan-proyek)
- [Penyimpanan Model](#-penyimpanan-model)
- [Catatan dan Batasan](#-catatan-dan-batasan)

## 🎯 Tujuan Proyek

Proyek ini bertujuan untuk:

1. Mengolah dataset historis kebakaran hutan dan lahan untuk kebutuhan pemodelan Machine Learning.
2. Menerapkan preprocessing data menggunakan Pandas dan NumPy.
3. Mengembangkan model regresi berbasis Artificial Neural Network menggunakan TensorFlow.
4. Menerapkan normalisasi fitur menggunakan StandardScaler.
5. Membagi dataset menjadi data training, validation, dan testing.
6. Menerapkan Batch Normalization dan Dropout untuk membantu proses pelatihan dan mengurangi risiko overfitting.
7. Mengevaluasi performa model menggunakan Mean Squared Error (MSE) dan Mean Absolute Error (MAE).
8. Menyimpan model yang telah dilatih agar dapat digunakan kembali pada tahap inferensi.

## 👥 Anggota Kelompok

| No. | Nama | NIM |
|:---:|---|---|
| 1 | Adventus Tangkasiang Senas | 2330305030068 |
| 2 | Yehezkiel Luckyus Tarigan | 2330205030046 |
| 3 | Galih Indra Gunawan | 2330305030077 |
| 4 | Ferdy Afriza Maulana | 2330205030038 |
| 5 | Yulison Wenda | 2330405030066 |

## 📊 Dataset

Dataset yang digunakan dalam proyek ini adalah dataset kebakaran hutan dan lahan Indonesia yang tersedia melalui Hugging Face.

**Sumber dataset:**

[AML Indo Fires — Hugging Face](https://huggingface.co/datasets/jq5522/aml_indo_fires/blob/main/panel_eda_enriched_2012_2025.parquet)

**Nama file:** `panel_eda_enriched_2012_2025.parquet`

**Format:** Apache Parquet

**Periode data:** 2012–2025

Dataset dibaca menggunakan Pandas melalui fungsi `read_parquet()`.

### Seleksi Data

Pada eksperimen yang didokumentasikan, data difokuskan pada wilayah Provinsi Kalimantan Tengah (*Central Kalimantan*).

Proses seleksi wilayah dilakukan dengan memeriksa kolom `province` atau `provinsi`, bergantung pada struktur dataset yang digunakan.

Jika jumlah sampel Kalimantan Tengah kurang dari 5.000 baris, implementasi menyediakan mekanisme alternatif untuk menggunakan data dari seluruh wilayah Kalimantan yang memenuhi kriteria tersebut.

Data yang tidak memiliki nilai target dihapus sebelum proses pelatihan dilakukan.

### Fitur dan Target

- **Fitur (X):** Kolom numerik dalam dataset, kecuali kolom target.
- **Target (y):** Kolom yang nilainya ingin diprediksi oleh model.

Pada eksperimen yang menghasilkan output evaluasi dalam README ini, kolom target yang digunakan adalah `fire_count_lnh`.

Kolom target perlu disesuaikan dengan implementasi preprocessing yang benar-benar digunakan saat training. Jika eksperimen menggunakan `hotspot_count`, target tersebut harus ditetapkan secara konsisten pada seluruh tahapan pemodelan.

Fitur numerik yang memiliki nilai kosong diisi menggunakan nilai median dari masing-masing kolom.

## 🛠️ Teknologi yang Digunakan

| Teknologi | Fungsi |
|---|---|
| Python | Bahasa pemrograman utama |
| TensorFlow | Framework untuk membangun dan melatih model neural network |
| Keras | API untuk membangun arsitektur model |
| Pandas | Membaca dan mengolah dataset |
| NumPy | Manipulasi data numerik |
| Scikit-learn | Pembagian dataset dan normalisasi fitur |
| Matplotlib | Visualisasi hasil pelatihan |
| Google Colab | Lingkungan komputasi untuk menjalankan eksperimen |
| Hugging Face | Sumber dataset |

## 🔄 Alur Pengembangan Model

Pengembangan model dilakukan melalui beberapa tahapan berikut.

### 1. Pengumpulan Data

Dataset diunduh dari Hugging Face dalam format Parquet, kemudian diunggah ke Google Colab agar dapat diproses menggunakan Python.

### 2. Preprocessing Data

Tahapan preprocessing meliputi:

- Membaca dataset menggunakan Pandas.
- Memilih data berdasarkan wilayah yang ditentukan.
- Menghapus baris yang tidak memiliki nilai target.
- Memisahkan fitur numerik dan variabel target.
- Mengisi nilai kosong pada fitur menggunakan median.
- Menyiapkan data untuk proses training.

### 3. Pembagian Dataset

Dataset dibagi menjadi tiga bagian:

| Dataset | Proporsi | Jumlah Sampel |
|---|---:|---:|
| Training | 70% | 27.167 |
| Validation | 15% | 5.819 |
| Testing | 15% | 5.822 |
| **Total** | **100%** | **38.808** |

Pembagian data dilakukan menggunakan `train_test_split` dengan `random_state=42` agar hasil pembagian dapat direproduksi.

Data training digunakan untuk memperbarui parameter model, data validation digunakan untuk memantau performa selama pelatihan, dan data testing digunakan untuk mengevaluasi model setelah pelatihan selesai.

### 4. Normalisasi Fitur

Normalisasi dilakukan menggunakan `StandardScaler` dari Scikit-learn.

Standardisasi membantu menyamakan skala fitur numerik sehingga setiap fitur memiliki skala yang lebih sesuai untuk proses pelatihan neural network.

Scaler hanya di-*fit* pada data training. Parameter hasil fit tersebut kemudian digunakan untuk mentransformasikan data validation dan testing guna menghindari kebocoran informasi dari kedua dataset tersebut.

### 5. Pelatihan Model

Model dilatih menggunakan TensorFlow dengan arsitektur MLP yang terdiri atas tiga hidden layer.

Proses training menggunakan optimizer Adam, loss function Mean Squared Error, serta metrik Mean Absolute Error.

### 6. Evaluasi Model

Setelah pelatihan selesai, model dievaluasi menggunakan data testing yang tidak digunakan untuk memperbarui bobot model selama training.

Metrik evaluasi yang digunakan adalah MAE, sedangkan MSE digunakan sebagai fungsi loss selama proses pelatihan dan evaluasi.

### 7. Penyimpanan Model

Model yang telah dilatih disimpan dalam format HDF5 (`.h5`) agar dapat dimuat kembali menggunakan TensorFlow/Keras.

## 🧠 Arsitektur Model

Model menggunakan arsitektur **Multilayer Perceptron (MLP) untuk regresi**.

Arsitektur yang digunakan adalah sebagai berikut:

| Urutan | Layer | Konfigurasi | Output Shape |
|---:|---|---|---|
| 1 | Dense | 128 neuron, ReLU | `(None, 128)` |
| 2 | BatchNormalization | Normalisasi aktivasi | `(None, 128)` |
| 3 | Dropout | 30% | `(None, 128)` |
| 4 | Dense | 64 neuron, ReLU | `(None, 64)` |
| 5 | BatchNormalization | Normalisasi aktivasi | `(None, 64)` |
| 6 | Dropout | 20% | `(None, 64)` |
| 7 | Dense | 32 neuron, ReLU | `(None, 32)` |
| 8 | Dense | 1 neuron, Linear | `(None, 1)` |

### Penjelasan Komponen

**Dense Layer**

Digunakan untuk mempelajari hubungan nonlinier antara fitur masukan dan target melalui kombinasi bobot, bias, dan fungsi aktivasi.

**ReLU (Rectified Linear Unit)**

Digunakan pada hidden layer untuk membantu model mempelajari pola nonlinier dari data.

**Batch Normalization**

Membantu menstabilkan distribusi aktivasi selama training dan dapat mendukung proses optimasi model.

**Dropout**

Menonaktifkan sebagian unit secara acak selama training untuk membantu mengurangi risiko overfitting.

**Linear Output Layer**

Menghasilkan satu nilai prediksi kontinu sehingga sesuai untuk permasalahan regresi.

### Ringkasan Parameter Model

Berdasarkan output `model.summary()`:

| Parameter | Jumlah |
|---|---:|
| Total Parameters | 23.681 |
| Trainable Parameters | 23.297 |
| Non-trainable Parameters | 384 |
| Perkiraan ukuran parameter | 92,50 KB |

Ukuran tersebut merupakan perkiraan ukuran parameter model dan tidak mencakup seluruh overhead penyimpanan file.

## ⚙️ Konfigurasi Pelatihan

Konfigurasi training yang digunakan pada eksperimen adalah:

| Parameter | Nilai |
|---|---|
| Framework | TensorFlow / Keras |
| Jenis Model | Multilayer Perceptron |
| Jenis Permasalahan | Regresi |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Evaluation Metric | Mean Absolute Error (MAE) |
| Epochs | 50 |
| Batch Size | 64 |
| Fungsi Aktivasi Hidden Layer | ReLU |
| Fungsi Aktivasi Output Layer | Linear |
| Random State | 42 |
| Jumlah Fitur | 97 |
| Jumlah Sampel | 38.808 |

Model dilatih menggunakan data training dan dipantau menggunakan data validation pada setiap epoch.

## 📈 Hasil Eksperimen

Berdasarkan output eksperimen training yang tersedia, diperoleh hasil berikut.

### Ringkasan Dataset

| Komponen | Hasil |
|---|---:|
| Total sampel | 38.808 |
| Jumlah fitur masukan | 97 |
| Sampel training | 27.167 |
| Sampel validation | 5.819 |
| Sampel testing | 5.822 |

### Hasil Evaluasi

| Metrik | Hasil |
|---|---:|
| Test MAE | 3,99 |

Nilai MAE sebesar 3,99 menunjukkan bahwa rata-rata selisih absolut antara nilai aktual dan nilai prediksi adalah sekitar 3,99 **dalam skala target yang digunakan saat evaluasi**.

Interpretasi nilai tersebut bergantung pada definisi variabel target. Jika target menggunakan transformasi logaritmik atau transformasi lainnya, nilai MAE tidak dapat langsung diartikan sebagai jumlah kejadian kebakaran tanpa melakukan interpretasi atau transformasi balik yang sesuai.

Nilai MAE ini merupakan hasil eksperimen yang tercatat, bukan jaminan bahwa model akan menghasilkan performa yang sama pada data baru.

## 📁 Struktur Repository

Struktur repository yang disarankan:

```text
prediksi-karhutla-tensorflow/
│
├── README.md
│
├── notebooks/
│   └── training_karhutla.ipynb
│
├── models/
│   └── mlp_karhutla.h5
│
├── requirements.txt
│
└── .gitignore
```

Keterangan:

- `README.md`: Dokumentasi proyek.
- `notebooks/training_karhutla.ipynb`: Notebook untuk preprocessing, training, dan evaluasi.
- `models/mlp_karhutla.h5`: Model hasil training, jika file tersebut disertakan dalam repository.
- `requirements.txt`: Daftar dependensi Python.
- `.gitignore`: Daftar file dan direktori yang tidak perlu dimasukkan ke Git.

Struktur tersebut merupakan rekomendasi. File model dan notebook perlu ditambahkan ke repository secara terpisah.

## 🚀 Cara Menjalankan Proyek

### 1. Siapkan Repository

Clone repository:

```bash
git clone <URL-REPOSITORY>
cd prediksi-karhutla-tensorflow
```

Ganti `<URL-REPOSITORY>` dengan URL repository GitHub yang sebenarnya.

### 2. Buka Google Colab

Buka [Google Colab](https://colab.research.google.com/) dan unggah notebook `training_karhutla.ipynb`.

Notebook dapat dijalankan pada runtime Python Google Colab.

### 3. Instal Dependensi

Jalankan perintah berikut pada cell Google Colab jika dependensi yang dibutuhkan belum tersedia:

```python
!pip install tensorflow pandas numpy scikit-learn matplotlib pyarrow
```

Library `pyarrow` digunakan untuk membantu membaca file Parquet melalui Pandas.

### 4. Unduh Dataset

Unduh file dataset dari halaman berikut:

[Download Dataset panel_eda_enriched_2012_2025.parquet](https://huggingface.co/datasets/jq5522/aml_indo_fires/blob/main/panel_eda_enriched_2012_2025.parquet)

Unggah file tersebut ke Google Colab melalui panel Files.

Pastikan nama file sesuai dengan nama yang digunakan dalam kode:

```python
file_path = "/content/panel_eda_enriched_2012_2025.parquet"
```

Jika nama file berbeda, sesuaikan variabel `file_path`.

### 5. Jalankan Preprocessing

Jalankan cell preprocessing untuk membaca dataset, memilih wilayah, menangani missing values, memisahkan fitur dan target, membagi dataset, serta melakukan standardisasi.

Pastikan nama kolom target konsisten dengan eksperimen yang akan dijalankan.

### 6. Jalankan Training

Jalankan cell yang membangun model MLP menggunakan TensorFlow/Keras.

Model kemudian dilatih menggunakan konfigurasi yang telah ditentukan:

```python
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=64,
    verbose=1
)
```

### 7. Evaluasi Model

Setelah training selesai, jalankan evaluasi menggunakan data testing:

```python
test_loss, test_mae = model.evaluate(
    X_test_scaled,
    y_test
)

print(f"Test MAE: {test_mae:.2f}")
```

Output evaluasi dapat berbeda apabila versi dataset, pembagian data, konfigurasi model, atau lingkungan training berubah.

### 8. Simpan Model

Untuk menyimpan model di Google Drive, mount Drive terlebih dahulu:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Kemudian simpan model:

```python
import os

model_path = (
    "/content/drive/MyDrive/skripsi/models/mlp_karhutla.h5"
)

os.makedirs(os.path.dirname(model_path), exist_ok=True)

model.save(model_path)

print(f"Model tersimpan di: {model_path}")
```

Pastikan direktori tujuan tersedia dan Google Drive telah berhasil di-mount sebelum menjalankan proses penyimpanan.

## 💾 Penyimpanan Model

Model disimpan dengan nama:

`mlp_karhutla.h5`

Format HDF5 (`.h5`) didukung oleh TensorFlow/Keras, tetapi merupakan format penyimpanan legacy. Untuk pengembangan baru, format native Keras (`.keras`) juga dapat digunakan.

Contoh penyimpanan menggunakan format native Keras:

```python
model.save("mlp_karhutla.keras")
```

Model dapat dimuat kembali menggunakan:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "mlp_karhutla.keras"
)
```

**Catatan:** Agar prediksi pada data baru konsisten dengan proses training, objek `StandardScaler` yang telah di-fit pada data training juga harus disimpan. Urutan dan nama fitur yang digunakan saat inferensi harus sama dengan urutan dan nama fitur ketika model dilatih.

## ⚠️ Catatan dan Batasan

Beberapa hal perlu diperhatikan dalam penggunaan model ini:

1. **Ketergantungan pada dataset:** Kualitas prediksi bergantung pada kualitas, kelengkapan, dan relevansi data historis yang digunakan.
2. **Konsistensi target:** Nama kolom target pada preprocessing harus sama dengan target yang digunakan ketika model dilatih dan dievaluasi.
3. **Konsistensi fitur:** Kolom numerik yang dipilih harus sesuai dengan fitur yang digunakan dalam eksperimen. Kolom numerik yang tidak relevan, seperti pengenal atau kode kategori, perlu ditinjau sebelum training.
4. **Potensi kebocoran data:** Pembagian acak dapat menempatkan observasi dari wilayah atau periode yang saling berkaitan pada training dan testing. Untuk mengukur kemampuan prediksi pada periode mendatang, evaluasi berbasis waktu dapat menjadi pilihan yang lebih sesuai.
5. **Interpretasi MAE:** Nilai MAE harus ditafsirkan berdasarkan skala target yang sebenarnya, terutama jika target telah mengalami transformasi.
6. **Keterbatasan penggunaan:** Model ini merupakan model eksperimen dan belum dapat dianggap sebagai sistem peringatan dini kebakaran yang siap digunakan secara operasional tanpa validasi tambahan.

## 📚 Referensi

- **Dataset:** [AML Indo Fires — Hugging Face](https://huggingface.co/datasets/jq5522/aml_indo_fires/blob/main/panel_eda_enriched_2012_2025.parquet)
- **TensorFlow Documentation:** [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **Keras Documentation:** [https://keras.io/](https://keras.io/)
- **Scikit-learn Documentation:** [https://scikit-learn.org/](https://scikit-learn.org/)

---

**Dikembangkan oleh kelompok mahasiswa sebagai proyek Machine Learning untuk prediksi kebakaran hutan dan lahan menggunakan TensorFlow.**