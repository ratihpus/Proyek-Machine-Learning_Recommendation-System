# Laporan Proyek Machine Learning - Ratih Puspitasari

>## Project Overview

Domain proyek Machine Learning untuk sistem rekomendasi ini adalah mengenai rekomendasi destinasi wisata kota Semarang dengan judul "Recommend top Semarang Touristic Destination based on preference". 

Kota Semarang, sebagai ibu kota Provinsi Jawa Tengah, merupakan salah satu destinasi wisata utama di Indonesia. Dengan beragam tempat wisata, seperti Lawang Sewu, Kota Lama, Sam Poo Kong, dan Pantai Marina, Semarang menarik perhatian wisatawan lokal maupun mancanegara. Namun, tantangan yang sering muncul adalah bagaimana wisatawan dapat menemukan destinasi yang paling sesuai dengan minat mereka, terutama jika waktu kunjungan terbatas.

Tren digitalisasi di sektor pariwisata menunjukkan peningkatan signifikan dalam penggunaan sistem rekomendasi untuk membantu wisatawan merencanakan perjalanan. Sistem rekomendasi berbasis data memungkinkan analisis preferensi pengguna, ulasan, serta tren kunjungan, sehingga menghasilkan saran destinasi yang optimal. Dengan mengintegrasikan data tentang tempat wisata, ulasan wisatawan, serta pola kunjungan, sistem rekomendasi ini diharapkan dapat membantu wisatawan menjelajahi keindahan Kota Semarang secara lebih terarah.

Tentunya dengan perkembangan teknologi yang semakin pesat, dapat membawa dampak yang baik terhadap perekonomian. Pada sebuah destinasi wisata, kita dapat menerapkan yang namanya sistem rekomendasi. Apa itu sistem rekomendasi? [Sistem Rekomendasi](https://en.wikipedia.org/wiki/Recommender_system)  adalah subkelas sistem penyaringan informasi yang berupaya memprediksi "peringkat" atau "preferensi" yang akan diberikan kepada pengguna pada suatu item/produk. Sistem ini dapat digunakan untuk memberikan rekomendasi secara personal tergantung pola interaksi masing-masing pengguna. Sangat bermanfaat bukan?

Proyek ini penting untuk diselesaikan agar pelanggan/user mendapatkan rekomendasi destinasi wisata di Kota Semarang berdasarkan data user, rating, dan place.

>## Business Understanding

### Problem Statements

Kita ingin mempergunakan data preferensi dari pengguna/user untuk meningkatkan kualitas layanan agar memperoleh penilaian yang baik dari pengguna/user. Namun, kita memiliki  masalah terkait:
- Bagaimana membuat model machine learning untuk merekomendasikan destinasi wisata yang sesuai dengan rating atau penilaian user?
- Bagaimana melakukan tahapan pra-pemrosesan data sebelum data tersebut dimasukkan ke dala model machine learning?
- Bagaimana menyiapkan data places dan ratings untuk digunakan dalam melatih model machine learning sistem rekomendasi?

### Goals

Berdasarkan masalah yang telah disebutkan pada bagian `Problem Statements`, berikut adalah tujuan yang ingin kita capai:
- Membuat model Machine learning yang dapat memberikan inside rekomendasi place atau destinasi wisata di kota Semarang yang terbaik sesuai dengan ratings dan pengunjung tersebut.
- Melakukan tahapan pra-pemrosesan/mengolah data sebelum data tersebut dimasukkan kedalam model machine learning.
- Melakuka tahapan persiapan data sehingga data siap digunakan untuk melatih model machine learning sistem rekomendasi.

### Solution statements

Untuk menyelesaikan masalah yang telah disebutkan pada bagian `Problem Statements`, kita akan menggunakan pendekatan sistem rekomendasi dengan teknik **Content-based filtering Recommendation** dan **Collaborative Filtering Recommedation**. 
- `Content-based Filtering Recommendation`: Salah satu metode dalam sistem rekomendasi yang berfokus pada analisis konten atau fitur dari item yang akan direkomendasikan dan mencocokkannya dengan preferensi pengguna. Sistem ini bekerja dengan menganalisis karakteristik item (seperti deskripsi, kategori, atau fitur lainnya) dan mencocokkannya dengan profil pengguna yang dibuat berdasarkan item yang sudah disukai atau digunakan.

- `Collaborative Filtering Recommedation`: Sistem merekomendasikan sejumlah destinasi wisata berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna, kita akan mengidentifikasi destinasi wisata yang mirip dan belum pernah dikunjungi oleh pengguna untuk direkomendasikan. Teknik ini menggunakan model based collaborative filtering : SVD (Singular Value Decomposition)

>## Data Understanding
<p align="center">
  <img src="https://github.com/ratihpus/Proyek-Machine-Learning_Recommendation-System/blob/eef2ec7907eeb1a2b693eef3bdac21adfe579277/img/datasetinfo.PNG?raw=true"/>  
</p>

Dataset ini digunakan untuk keperluan studi/pendidikan dimana diharapkan dapat menghasilkan daftar restoran teratas sesuai dengan preferensi konsumen dan menemukan fitur signifikan. 
Dataset diperoleh dari [Kaggle](https://www.kaggle.com/). ***Kaggle*** merupakan platform penyedia dataset untuk data science. Untuk proyek ini, dataset yang kita pakai yaitu:
[Recommend top restaurants based on preference](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

Pada proyek ini, kita hanya akan menggunakan file `tourism_with_id.csv` dan `tourism_rating.csv` sebagai dataset yang akan digunakan. 

- tourism_with_id.csv - mengandung informasi tempak wisata di 5 kota besar di Indonesia (hanya kota Bandung yang dipakai)
- user.csv - mengandung informasi pengguna untuk membuat rekomendasi fitur berdasar pengguna
- tourism_rating.csv - mengandung informasi pengguna, tempat wisata, dan rating untuk membuat sistem rekomendasi berdasar rating

Berikut adalah Exploratory Data Analysis (EDA) yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

# Place 
  <br> Informasi Dataset tourism_with_id
  <p align="center">
  <img src="https://github.com/ratihpus/Proyek-Machine-Learning_Recommendation-System/blob/main/img/dataplace_info.PNG? raw=true"/>  
  </p>
  
tourism_with_id terdiri dari 437 baris dan 10 kolom sebagai berikut:

- Place_Id: kolom yang menunjukkan id dari setiap tempat wisata.
- Place_Name: kolom yang menunjukkan nama dari setiap tempat wisata.
- Description: kolom yang menunjukkan deskripsi dari setiap tempat wisata.
- Category: kolom yang menunjukkan kategori dari setiap tempat wisata.
- City: kolom yang menunjukkan kota dimana tempat wisata tersebut berada.
- Price: kolom yang menunjukkan harga tiket masuk ke tempat wisata tersebut.
- Rating: kolom yang menunjukkan rating dari setiap tempat wisata.
- Time_Minutes: kolom yang menunjukkan waktu yang diperlukan untuk mengunjungi tempat wisata tersebut.
- Coordinate: kolom yang menunjukkan koordinat dari setiap tempat wisata.
- Lat: kolom yang menunjukkan latitude dari setiap tempat wisata.
- Long: kolom yang menunjukkan longitude dari setiap tempat wisata.

Berikut adalah visualisasi dari dataset tourism_with_id:
<p align="center">
  <img src="https://github.com/ratihpus/Proyek-Machine-Learning_Recommendation-System/blob/main/img/sampel_dataplace.PNG? raw=true"/>  
</p>

# Rating
  <br> Informasi Dataset tourism_rating
  <p align="center">
  <img src="https://github.com/ratihpus/Proyek-Machine-Learning_Recommendation-System/blob/main/img/sampel_dataplace.PNG? raw=true"/>  
  </p>
  tourism_rating terdiri dari 10000 baris dan 3 kolom sebagai berikut:
  - User_Id: identitas unik dari setiap pengguna.
  - Place_Id: identitas unik dari setiap tempat wisata.
  - Place_Ratings: penilaian atau rating yang diberikan oleh pengguna terhadap tempat wisata tertentu.

  Berikut adalah visualisasi dari dataset tourism_rating:
  
   <p align="center">
  <img src="https://github.com/ratihpus/Proyek-Machine-Learning_Recommendation-System/blob/main/img/datarating_info.PNG? raw=true"/>  
  </p>

>## Exploratory Data Analysis (EDA)

Tahap eksplorasi penting untuk memahami variabel-variabel pada data serta korelasi antar variabel. Pemahaman terhadap variabel pada data dan korelasinya akan membantu kita dalam menentukan pendekatan atau algoritma yang cocok untuk data kita. Idealnya, kita melakukan eksplorasi data terhadap seluruh variabel.

Exploratory Data Analysis (EDA) memiliki peranan penting untuk dapat memahami dataset secara baik dan detail.

>## Data Preparation
