# Laporan Proyek Machine Learning - Ratih Puspitasari

>## Project Overview

Domain proyek Machine Learning untuk sistem rekomendasi ini adalah mengenai rekomendasi destinasi wisata kota Semarang dengan judul "Recommend top Semarang Touristic Destination based on preference".

Kota Semarang, sebagai ibu kota Provinsi Jawa Tengah, merupakan salah satu destinasi wisata utama di Indonesia. Dengan beragam tempat wisata, seperti Lawang Sewu, Kota Lama, Sam Poo Kong, dan Pantai Marina, Semarang menarik perhatian wisatawan lokal maupun mancanegara. Namun, tantangan yang sering muncul adalah bagaimana wisatawan dapat menemukan destinasi yang paling sesuai dengan minat mereka, terutama jika waktu kunjungan terbatas.

Tren digitalisasi di sektor pariwisata menunjukkan peningkatan signifikan dalam penggunaan sistem rekomendasi untuk membantu wisatawan merencanakan perjalanan. Sistem rekomendasi berbasis data memungkinkan analisis preferensi pengguna, ulasan, serta tren kunjungan, sehingga menghasilkan saran destinasi yang optimal. Dengan mengintegrasikan data tentang tempat wisata, ulasan wisatawan, serta pola kunjungan, sistem rekomendasi ini diharapkan dapat membantu wisatawan menjelajahi keindahan Kota Semarang secara lebih terarah.

Tentunya dengan perkembangan teknologi yang semakin pesat, dapat membawa dampak yang baik terhadap perekonomian. Pada sebuah destinasi wisata, kita dapat menerapkan yang namanya sistem rekomendasi. Apa itu sistem rekomendasi? [Sistem Rekomendasi](https://en.wikipedia.org/wiki/Recommender_system)  adalah subkelas sistem penyaringan informasi yang berupaya memprediksi "peringkat" atau "preferensi" yang akan diberikan kepada pengguna pada suatu item/produk. Sistem ini dapat digunakan untuk memberikan rekomendasi secara personal tergantung pola interaksi masing-masing pengguna. Sangat bermanfaat bukan?

Proyek ini penting untuk diselesaikan agar pelanggan/user mendapatkan rekomendasi destinasi wisata di Kota Semarang berdasarkan data user, rating, dan place.

>## Data Understanding
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/recommendation-system/blob/main/img/kaggle-dataset-preview.png?raw=true"/>
</p>

Dataset ini digunakan untuk keperluan studi/pendidikan dimana diharapkan dapat menghasilkan daftar restoran teratas sesuai dengan preferensi konsumen dan menemukan fitur signifikan. 
Dataset diperoleh dari [Kaggle](https://www.kaggle.com/). ***Kaggle*** merupakan platform penyedia dataset untuk data science. Untuk proyek ini, dataset yang kita pakai yaitu:
[Recommend top restaurants based on preference](https://www.kaggle.com/uciml/restaurant-data-with-consumer-ratings)

Pada proyek ini, kita hanya akan menggunakan file `rating_final.csv`. Kita menggunakan dataset `rating_final.csv` untuk merekomendasikan restoran berdasarkan popularitas & berdasarkan Collaborative yaitu peringkat/rating yang diberikan oleh setiap pengguna.
Pada  `rating_final.csv` terdapat 5 fitur :
- `userID`  Nominal, merupakan ID Pengguna yang memberikan penilaian 
- `placeID` Nominal, merupakan ID Restoran yang akan dinilai 
- `rating`  Numeric 0 - 2, merupakan penilaian restoran menurut user 
- `food_rating`  Numeric 0 - 2, merupakan penilaian makanan restoran menurut user


