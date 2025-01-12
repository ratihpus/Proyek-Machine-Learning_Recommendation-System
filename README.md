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

Dataset ini digunakan untuk keperluan studi/pendidikan dimana diharapkan dapat menghasilkan daftar restoran teratas sesuai dengan preferensi konsumen dan menemukan fitur signifikan. 

Dataset diperoleh dari [Kaggle](https://www.kaggle.com/). ***Kaggle*** merupakan platform penyedia dataset untuk data science. Untuk proyek ini, dataset yang kita pakai yaitu:
[Recommend top Tourism Destination based on preference](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

Pada proyek ini, kita hanya akan menggunakan file `tourism_with_id.csv` dan `tourism_rating.csv` sebagai dataset yang akan digunakan. 

- tourism_with_id.csv - mengandung informasi tempak wisata di 5 kota besar di Indonesia (hanya kota Bandung yang dipakai)
- user.csv - mengandung informasi pengguna untuk membuat rekomendasi fitur berdasar pengguna
- tourism_rating.csv - mengandung informasi pengguna, tempat wisata, dan rating untuk membuat sistem rekomendasi berdasar rating

Berikut adalah Exploratory Data Analysis (EDA) yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

### Place 
  <br> Informasi Dataset tourism_with_id.csv
  
Jumlah data pada dataset tersebut yaitu jumlah place sebanyak 437 sedangkan jumlah rating sebanyak 10000. 

Pada tourism_with_id terdiri dari 437 baris dan 10 kolom sebagai berikut:

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
| Place_Id | Place_Name        | Description                                                                                                                                                               | Category    | City      | Price | Rating | Time_Minutes | Coordinate                                          | Lat        | Long       |
|----------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|-----------|-------|--------|--------------|-----------------------------------------------------|------------|------------|
| 335      | Candi Gedong Songo| Candi Gedong Songo (bahasa Jawa: ꦕꦤ꧀ꦝꦶ​ꦒꦼꦝꦺꦴꦁ​...)                                                                                                              | Budaya      | Semarang | 10000 | 4.5    | NaN          | {'lat': -7.209886700000002, 'lng': 110.3421119}      | -7.209887  | 110.342112 |
| 336      | Grand Maerakaca   | Masyarakat Jawa Tengah mungkin sudah tidak asi...                                                                                                                          | Taman Hiburan | Semarang | 15000 | 4.4    | NaN          | {'lat': -6.9605225, 'lng': 110.3863941}             | -6.960522  | 110.386394 |
| 337      | Kampung Pelangi    | Kampung pelangi atau dalam bahasa Inggris dise...                                                                                                                           | Taman Hiburan | Semarang | 3000  | 4.3    | 30.0         | {'lat': -6.988881200000001, 'lng': 110.4083781}     | -6.988881  | 110.408378 |
| 338      | Lawang Sewu       | Lawang Sewu ("Seribu Pintu") (bahasa Jawa: ꦭꦮꦁ...)                                                                                                                        | Budaya      | Semarang | 10000 | 4.6    | NaN          | {'lat': -6.9839099, 'lng': 110.4104342}             | -6.983910  | 110.410434 |
| 339      | Sam Poo Kong Temple| Sam Poo Kong (Hanzi: ; Pinyin: Sānbǎo Dòng), j...                                                                                                                          | Budaya      | Semarang | 35000 | 4.5    | NaN          | {'lat': -6.996236599999999, 'lng': 110.398122}      | -6.996237  | 110.398122 |

### User 
  <br> Informasi Dataset user.csv
  Tahapan ini meliputi melihat gambaran data user, merubah data user agar hanya berisi user yang pernah megunjungi wisata di Kota Semarang, melihat dataset user yang pernah memberi rating pada wisata di Kota Semarang.
  <br>
  | **User_Id** | **Location**                 | **Age** |
|-------------|------------------------------|---------|
| 1           | Semarang, Jawa Tengah        | 20      |
| 2           | Bekasi, Jawa Barat           | 21      |
| 3           | Cirebon, Jawa Barat          | 23      |
| 4           | Bekasi, Jawa Barat           | 21      |
| 5           | Lampung, Sumatera Selatan    | 20      |

<br>
  Pada data user terdiri 297 jumlah pengguna atau entri data dan terdiri dari tiga kolom: User_Id, Location, dan Age.
  <br>
  - User_Id:
  <br>
  Merupakan ID unik untuk setiap pengguna. Angka ini bersifat incremental atau acak untuk membedakan satu pengguna dengan pengguna lainnya.
  <br>
  - Location:
  <br>
  Menunjukkan lokasi pengguna, termasuk nama kota dan provinsi. Dalam data ini terlihat pengguna berasal dari kota-kota seperti Semarang, Bekasi, Cirebon, dan Lampung.
  <br>
  - Age: 
  <br>
  Menyediakan informasi tentang usia pengguna. Dalam contoh ini, usia berkisar antara 20 hingga 23 tahun, yang menunjukkan kelompok usia muda atau awal dewasa.
  
### Rating
  <br> Informasi Dataset tourism_rating.csv
<br>
tourism_rating terdiri dari 1317 baris dan 3 kolom sebagai berikut: 
<br> - User_Id: identitas unik dari setiap pengguna.
<br> - Place_Id: identitas unik dari setiap tempat wisata.
<br> - Place_Ratings: penilaian atau rating yang diberikan oleh pengguna terhadap tempat wisata tertentu.

Berikut adalah visualisasi dari dataset tourism_rating:
  
| User_Id | Place_Id | Place_Ratings |
|---------|----------|---------------|
| 5       | 335      | 3             |
| 20      | 335      | 4             |
| 41      | 335      | 5             |
| 55      | 335      | 2             |
| 70      | 335      | 3             |

>## Kondisi Data

Pada bagian kondisi data:
1. Pada dataset places yag terdiri dari kolom place_id, place_name, category, price, dan rating, tidak ditemukan missing value.
2. Pada dataset ratings yang terdiri dari kolom user_id, place_id, dan place_ratings, tidak ditemukan missing value.
3. Setelah melakukan pengecekan data duplikat, hasil analisis adalah :
   - Dataset places : tidak ditemukan data duplikat.
   - Dataset ratings : ditemukan 6 data duplikat.

>## Exploratory Data Analysis (EDA)

Tahap eksplorasi penting untuk memahami variabel-variabel pada data serta korelasi antar variabel. Pemahaman terhadap variabel pada data dan korelasinya akan membantu kita dalam menentukan pendekatan atau algoritma yang cocok untuk data kita. Idealnya, kita melakukan eksplorasi data terhadap seluruh variabel.

Exploratory Data Analysis (EDA) memiliki peranan penting untuk dapat memahami dataset secara baik dan detail.

  Hasil visualisasi dan wawasan :
  1. Menampilkan informasi terkait tempat yang paling sering dirating
     <br>
     
     dari hasil visualisasi hasil top 10 tempat-tempat berikut yang memiliki jumlah ulasan (rating) yang relatif tinggi :
     <br>
      - Pantai Marina dan Grand Maerakaca memiliki jumlah ulasan tertinggi, masing-masing sebanyak 33 ulasan.
      - Tempat-tempat lainnya dengan jumlah ulasan signifikan (31 ulasan) termasuk :
        
         * Tirto Argo Siwarak
         * Wisata Lereng Kelir
         * Kampoeng Rawa
         * Monumen Palagan Ambarawa
         * La Kana Chapel
         * Umbul Sidomukti
         * Tugu Muda Semarang
         * Hutan Pinus Kayon
    
  Wawasan dari hasil tersebut :
    <br>
  * Pantai Marina dan Grand Maerakaca:
       <br>
    - Kedua tempat ini berpotensi menjadi destinasi favorit wisatawan di wilayah tersebut, karena memiliki daya tarik khusus atau fasilitas yang sesuai dengan kebutuhan pengunjung.
    - Daya tarik ini bisa berupa lokasi strategis, aksesibilitas, keindahan tempat, atau kegiatan yang ditawarkan.
    <br>
  * Tempat dengan ulasan tinggi lainnya:
      <br>
    - Tempat-tempat seperti Tirto Argo Siwarak, Wisata Lereng Kelir, dan lainnya menunjukkan bahwa ada persebaran minat yang cukup luas dari wisatawan di berbagai jenis destinasi, seperti wisata alam, sejarah, dan budaya.
      

  2. Menampilkan visualisasi data distribusi rating
     <br>
     Dari visualisasi tersebut, berikut adalah beberapa wawasan berdasarkan grafik:
     <br>
     - Tempat Wisata dengan Rating Tertinggi:
         * Pantai Marina dan Grand Maerakaca terlihat memiliki jumlah rating terbanyak (33), mengindikasikan popularitasnya di kalangan pengunjung.
         * Tempat ini mungkin memiliki daya tarik yang lebih besar dibandingkan tempat lain, seperti fasilitas yang baik, keindahan alam, atau akses yang mudah.
    <br>
    - Kelompok Tempat dengan Popularitas yang Serupa:
         * Terdapat beberapa tempat dengan jumlah rating yang hampir sama (31), seperti Tirto Argo Siwarak, Wisata Lereng Kelir, Kampoeng Rawa, Monumen Palagan Ambarawa, dan La Kana Chapel. Hal ini menunjukkan bahwa mereka memiliki daya tarik yang kompetitif satu sama lain.

  3. Menampilkan visualisasi perbandingan jumlah kategori wisata di Kota Semarang
     <br>
     Hasil dan wawasan yang dapat diperoleh :
     - Kategori Cagar Alam memiliki jumlah tempat wisata terbanyak dibandingkan kategori lainnya, dengan sekitar 20 lokasi.
     - Kategori Taman Hiburan juga cukup dominan, berada di posisi kedua dengan jumlah lokasi yang sedikit lebih rendah dibandingkan Cagar Alam.
     - Kategori Budaya berada di posisi ketiga, menunjukkan bahwa wisata berbasis budaya juga memiliki daya tarik yang signifikan.
     - Kategori Bahari dan Tempat Ibadah memiliki jumlah tempat wisata paling sedikit dibandingkan kategori lainnya.
    <br>
    Wawasan :
      - Dominasi Cagar Alam dan Taman Hiburan:
        Kota Semarang tampaknya memiliki daya tarik wisata alam yang kuat, dengan fokus pada pelestarian lingkungan (Cagar Alam) dan hiburan keluarga (Taman Hiburan). Potensi wisata alam ini dapat diperkuat melalui promosi keberlanjutan dan pelibatan masyarakat lokal.
      - Kesempatan untuk Kategori Bahari dan Tempat Ibadah:
        Kategori Bahari dan Tempat Ibadah memiliki jumlah lokasi yang lebih sedikit, yang dapat menjadi peluang untuk eksplorasi lebih lanjut. Pemerintah atau pengelola wisata dapat mempertimbangkan pengembangan infrastruktur atau promosi wisata bahari, mengingat Semarang adalah kota pesisir.
      - Wisata Budaya:
        Wisata budaya memiliki potensi untuk menarik wisatawan lokal maupun internasional yang ingin mengeksplorasi sejarah dan tradisi kota Semarang. Meningkatkan program budaya atau event tahunan dapat memperkuat daya tarik wisata budaya.
      - Strategi Pengembangan:
        Penting untuk memahami kebutuhan dan preferensi wisatawan di setiap kategori. Misalnya, apakah wisatawan mencari pengalaman edukasi, rekreasi, atau spiritual. Mengintegrasikan kategori wisata (seperti taman hiburan di sekitar cagar alam atau acara budaya di tempat wisata bahari) bisa menjadi strategi untuk meningkatkan kunjungan wisata.
        
  4. Menampilkan visualisasi distribusi usia pengunjung (user)
     <br>
     berikut adalah analisis distribusi usia pengguna:
     - Median Usia:
       Median usia pengguna berada di sekitar 30 tahun, menunjukkan bahwa sebagian besar pengguna berusia mendekati angka tersebut.
     - Rentang Usia:
       Rentang usia pengguna terlihat berkisar antara 20 tahun (minimum) hingga 40 tahun (maksimum).
     - Interquartile Range (IQR):
       Sebagian besar data usia pengguna berada dalam rentang 25 hingga 35 tahun, yang merupakan kuartil pertama (Q1) hingga kuartil ketiga (Q3). Ini menunjukkan bahwa mayoritas pengguna berada dalam usia produktif.
       
  5. Membuat visualisasi distribusi harga masuk tempat wisata
     <br>
     Dari hasil visualisasi menunjukkan bahwa harga masuk wisata di Kota Semarang dari harga 0 (gratis) sampai kurang dari 50.000
     
  6. Menggabungkan Harga dan Waktu_Menit untuk tujuan Kategori.
     <br>
     berikut adalah analisis data mengenai harga (Price) dan waktu kunjungan (Time_Minutes) berdasarkan kategori tempat wisata:
     - Kategori Bahari :
       * Harga rata-rata: Rp 4.000
       * Total harga: Rp 16.000
       * Waktu kunjungan rata-rata: 90 menit
       * Total waktu kunjungan: 90 menit
       * Wawasan: Bahari memiliki rata-rata harga yang paling rendah dibandingkan kategori lainnya. Namun, waktu kunjungan rata-rata cukup lama, menunjukkan wisatawan cenderung menikmati aktivitas yang lebih santai di lokasi ini.
     - Kategori Budaya:
       * Harga rata-rata: Rp 13.166
       * Total harga: Rp 197.500
       * Waktu kunjungan rata-rata: 60,63 menit
       * Total waktu kunjungan: 485 menit
       * Wawasan: Wisata budaya memiliki harga rata-rata yang menengah. Waktu kunjungan rata-rata lebih pendek, mengindikasikan kegiatan mungkin lebih fokus pada eksplorasi situs budaya tertentu.
     - Kategori Cagar Alam:
       * Harga rata-rata: Rp 12.025
       * Total harga: Rp 240.500
       * Waktu kunjungan rata-rata: 61,88 menit
       * Total waktu kunjungan: 495 menit
       * Wawasan: Cagar alam memiliki harga rata-rata yang mirip dengan kategori budaya, dengan waktu kunjungan yang sedikit lebih lama. Hal ini menunjukkan daya tarik berupa aktivitas alam yang memakan waktu tetapi tetap terjangkau.
     - Kategori Taman Hiburan:
       * Harga rata-rata: Rp 34.400
       * Total harga: Rp 516.000
       * Waktu kunjungan rata-rata: 81,82 menit
       * Total waktu kunjungan: 900 menit
       * Wawasan: Taman hiburan memiliki harga rata-rata yang paling tinggi. Waktu kunjungan rata-rata juga cukup lama, menunjukkan bahwa wisatawan menghabiskan waktu lebih banyak untuk menikmati fasilitas dan aktivitas di lokasi ini.
     - Kategori Tempat Ibadah:
       * Harga rata-rata: Rp 0
       * Total harga: Rp 0
       * Waktu kunjungan rata-rata: Tidak ada data
       * Wawasan: Tempat ibadah biasanya tidak memiliki biaya masuk, sehingga harga rata-rata adalah Rp 0. Namun, data waktu kunjungan tidak tercatat, yang bisa menjadi area untuk ditingkatkan dalam pengumpulan data.
       
  7. Memfilter asal kota dari user
     <br>
     Hasil dari visualisasi menunjukkan bahwa jumlah asal kota dari pengguna (pengunjung) yaitu ;
     - Bekas (kota yang paling banyak)
     - Semarang
     - Yogyakarta
     - Lampung
     - Bogor
       
  8. Menghitung frekuensi kemunculan setiap nilai unik dalam kolom User_Id pada DataFrame user
      <br>
      Distribusi Pengguna: Setiap User_Id muncul satu kali, menunjukkan bahwa data ini bersifat unik untuk setiap pengguna.
     
  9. Menghitung frekuensi kemunculan setiap nilai unik dalam kolom Place_Id pada DataFrame rating
      <br>
      - Lokasi dengan Rating Tertinggi:
        <br>
        Place_Id 344 dan 336 memiliki jumlah rating tertinggi, yaitu 33 rating. Setelah melakukan Exploratory Data Analysis (EDA), kita memperoleh hasil:
      - Distribusi Rating:
        <br>
        Lokasi dengan Place_Id 366, 367, 368, 369, dan 377 memiliki jumlah rating yang sama, yaitu 31 rating, menempatkannya pada posisi berikutnya setelah Place_Id 344 dan 336.
      - Jumlah Lokasi Berdasarkan Kategori Rating:
        <br>
        * Lokasi dengan jumlah rating tinggi (≥30 rating): Ada 7 lokasi.
        * Lokasi dengan jumlah rating menengah (20–29 rating): Mayoritas tempat berada di kategori ini.
        * Lokasi dengan jumlah rating rendah (<20 rating): Beberapa lokasi memiliki popularitas yang relatif rendah.

<br>
* Semua (33) tempat yang paling sering dirating
<br>
* Semua (297) pengguna telah memberi peringkat minimal 1 kali
<br>
Untuk merekomendasikan tempat wisata dengan preferensi teratas, kita dapat meminta setiap pengguna memberi peringkat terhadap semua tempat wisata. Namun tentunya hal tersebut sedikit sulit dicapai. Solusinya kita akan mencoba memprediksi peringkat yang akan diberikan pengguna terhadap tempat wisata.

>## Data Preparation
Tahap ini bertujuan untuk mempersiapkan data yang akan digunakan untuk proses training model. Di sini dilakukan dengan langkah berikut :

- Menghapus kolom yang tidak diperlukan 
  <br>
  Menghapus kolom Unnamed: 11 dan Unnamed: 12, selain itu pada dataset tourism_with_id, data yang diperlukan hanya ada pada   kolom Place_Id, Place_Name, dan Category, jadi hapus yang lain.

  Pada dataset tourism_rating, semua kolom diperlukan, jadi tidak ada kolom yang dihapus.
  
- Mengecek Missing Value
  <br>
  Proses pengecekan data yang hilang atau missing value dilakukan pada masing-masing dataset tourism_with_id dan tourism_rating. Berdasarkan hasil pengecekan, ternyata tidak ada data yang hilang dari kedua dataset tersebut.

- Membuat kamus (dictionary) yang dapat digunakan untuk encoding dan decoding nilai-nilai dalam suatu kolom pada DataFrame.
  
- Melakukan encoding kolom user_id dan memetakan hasil encoding tersebut ke dalam DataFrame.
  
- Melakukan encoding kolom Place_id dan memetakan hasil encoding tersebut ke DataFrame.

- Melakukan pengacakan urutan baris dalam DataFrame, dengan tujuan menghindari bias urutan data, meningkatkan generalisasi model, mempersiapkan data untuk split.
  
Tahapan pada Data Preparation :
  1. Filter data : proses untuk menyaring subset data dari DataFrame berdasarkan suatu kondisi atau kriteria tertentu. Proses ini berguna untuk memfokuskan analisis pada data yang relevan atau memenuhi syarat tertentu.
  2. Merge data : menggabungkan dua DataFrame berdasarkan kolom yang sama atau kunci yang umum
  3. Handling missing value : proses menangani data yang hilang atau kosong (missing data) dalam dataset.
  4. Content Based Filtering Preparation : Ekstrasi fitur TF-IDF : TF-IDF Vectorizer digunakan untuk menemukan representasi fitur yang penting dari setiap kategori destinasi wisata. Alat ini dari library scikit-learn akan mengubah nilai-nilai tersebut menjadi vektor dengan menggunakan metode fit_transform dan transform, serta melakukan pemecahan data menjadi bagian-bagian yang lebih kecil secara langsung.
  5. Collaborative Filtering Preparation : teknik rekomendasi yang memanfaatkan interaksi pengguna dengan item (misalnya, rating yang diberikan pengguna terhadap produk atau tempat).
     Encode label : encoding label berarti mengubah kategori atau string menjadi angka yang dapat digunakan oleh model. 
     Split data : data harus dibagi menjadi training set dan test set. Umumnya, 80% data digunakan untuk pelatihan, dan 20% sisanya untuk pengujian.

>## Modelling

Pada tahap pengembangan model machine learning sistem rekomendasi, teknik content-based filtering recommendation dan collaborative filtering recommendation digunakan untuk memberikan rekomendasi tempat terbaik kepada pengguna berdasarkan rating atau penilaian yang telah mereka berikan pada tempat tersebut. Tujuannya adalah untuk memberikan hasil rekomendasi yang tepat sesuai dengan keinginan pengguna. 

-1. Content-based Filtering Recommendation  menggunakan Cosine Similarity.
<br>
Beberapa tahap yang dilakukan untuk membuat sistem rekomendasi dengan pendekatan content-based filtering adalah TF-IDF Vectorizer, cosine similarity, dan pengujian sistem rekomendasi.

* TF-IDF Vectorizer
  TF-IDF Vectorizer akan melakukan transformasi teks nama tempat menjadi bentuk angka berupa matriks.
  
* Cosine Similarity
  Cosine similarity digunakan untuk menghitung tingkat kesamaan antara dua data place dengan mengukur sudut antara kedua data tersebut. Teknik ini menghitung tingkat kesamaan dengan menggunakan sudut antara data place yang dianalisis. Hasil perhitungan ini akan memberikan nilai yang menunjukkan tingkat kesamaan antara dua data place, dimana nilai yang mendekati 1 menunjukkan tingkat kesamaan yang tinggi, dan nilai yang mendekati 0 menunjukkan tingkat kesamaan yang rendah.
  
* Hasil Top-N Recommendation
  Setelah data tempat wisata dikonversi menjadi matriks dengan menggunakan TF-IDF Vectorizer, dan tingkat kesamaan antar nama tempat ditentukan dengan menggunakan cosine similarity, selanjutnya dilakukan pengujian terhadap sistem rekomendasi yang menggunakan pendekatan content-based filtering recommendation.
<br>
Hasil dari Top-N Recomendation :

| Place_Name               | Category |
|--------------------------|----------|
| Candi Gedong Songo       | Budaya   |
| Semarang Chinatown       | Budaya   |
| Kampoeng Djadhoel Semarang | Budaya   |
| Pura Giri Natha          | Budaya   |
| Benteng Pendem           | Budaya   |

Berdasarkan hasil rekomendasi di atas, dapat dilihat bahwa sistem yang dibuat berhasil memberikan rekomendasi tempat berdasarkan sebuah tempat, yaitu 'Tugu Muda Semarang' dan dihasilkan rekomendasi tempat dengan kategori yang sama, yaitu budaya.

-2. Collaborative Filtering Recommendation menggunakan RecomenderNet.
<br>
Tahap-tahap yang dilakukan untuk membuat sistem rekomendasi dengan pendekatan collaborative filtering meliputi data preparation, pembagian data menjadi data latih dan data validasi, serta pembangunan model dan pengujian sistem rekomendasi.

* Data Preparation

Tahap data preparation dilakukan dengan proses encoding fitur User_Id pada dataset ratings dan fitur Place_Id pada dataset ratings menjadi sebuah array. Lalu hasil encoding tersebut akan dilakukan pemetaan atau mapping fitur yang telah dilakukan encoding tersebut ke dalam dataset ratings. Berdasarkan hasil encoding dan mapping tersebut, diperoleh jumlah user sebesar 297, jumlah tempat sebesar 57, nilai rating minimal sebesar 1.0, dan nilai rating maksimal yaitu 5.0

* Membagi Data Latih dan Data Validasi

Tahap pembagian dataset diawali dengan mengacak dataset ratings, kemudian melakukan pembagian menjadi data latih dan data validasi, yaitu dengan rasio data latih banding data validasi sebesar 80:20.

* Model Development dan Hasil Rekomendasi

Dari model machine learning yang telah dibangun menggunakan layer embedding dan regularizer, serta adam optimizer, binary crossentropy loss function, dan metrik RMSE (Root Mean Squared Error), diperoleh hasil pengujian sistem rekomendasi tempat wisata dengan pendekatan collaborative filtering. 

Melakukan pendefinisian kelas RecommenderNet untuk membangun model klasifikasi teks tersebut. Model ini akan memberikan rekomendasi kepada pengguna berdasarkan preferensi atau kecenderungan pengguna di masa lalu. Model ini dapat digunakan dalam berbagai bidang, seperti rekomendasi film, musik, produk, dan lain-lain. RecommenderNet menggunakan algoritma pembelajaran mesin seperti collaborative filtering atau content-based filtering untuk menentukan rekomendasi yang tepat untuk pengguna.

Parameter yang digunakan dalam model ini adalah:

1. users_count: jumlah user yang akan jadi input dimension pada user embedding, tepatnya sebagai jumlah elemen dari vocabulary atau kata-kata yang digunakan dalam input data
2. place_count: jumlah tempat yang akan jadi input dimension pada tempat embedding, tepatnya sebagai jumlah elemen dari vocabulary atau kata-kata yang digunakan dalam input data
3. embedding_size: ukuran embedding akan jadi output dimension pada user embedding dan tempat embedding, yaitu jumlah fitur yang dihasilkan oleh Embedding layer, yang merupakan hasil pengurangan dimensi dari input data.
Embedding layer ini akan mengubah representasi numerik dari input data menjadi representasi vektor yang lebih bermakna dan dapat dipahami oleh model machine learning.

Proses kompilasi atau compile dengan:

1. binary crossentropy loss function: loss function untuk menghitung loss pada model klasifikasi biner.
2. adam optimizer: algoritma optimisasi yang digunakan untuk mengupdate bobot pada model machine learning secara efisien.
3. metrik RMSE (Root Mean Square Error): metrik yang digunakan untuk mengukur seberapa jauh hasil prediksi dari model dari nilai aktual. RMSE dihitung dengan mencari rata-rata dari kuadrat error yang diakumulasikan dari seluruh data.

Berdasarkan hasil rekomendasi tempat di atas, dapat dilihat bahwa sistem rekomendasi mengambil pengguna acak (208), lalu dilakukan pencarian tempat dengan rating terbaik dari user tersebut.

- Water Blaster Bukit Candi Golf: Taman Hiburan
- La Kana Chapel: Taman Hiburan
- Masjid Agung Ungaran: Tempat Ibadah
- Air Terjun Semirang: Cagar Alam

Selanjutnya, sistem akan menampilkan 10 daftar tempat yang direkomendasikan berdasarkan kategori yang dimiliki terhadap data pengguna acak tadi. Dapat dilihat bahwa sistem merekomendasikan beberapa tempat dengan kategori yang sama, seperti

1. Candi Gedong Songo : Budaya
   Harga Tiket Masuk: 10000
   Rating Wisata: 4.5

2. Grand Maerakaca : Taman Hiburan
   Harga Tiket Masuk: 15000
   Rating Wisata: 4.4

3. Desa Wisata Lembah Kalipancur : Taman Hiburan
   Harga Tiket Masuk: 0
   Rating Wisata: 3.9

4. Hutan Wisata Tinjomoyo Semarang : Cagar Alam
   Harga Tiket Masuk: 3000
   Rating Wisata: 4.3

5. Indonesia Kaya Park : Taman Hiburan
   Harga Tiket Masuk: 0
   Rating Wisata: 4.6

6. Pantai Cipta : Bahari
   Harga Tiket Masuk: 5000
   Rating Wisata: 4.0

7. Old City 3D Trick Art Museum : Budaya
   Harga Tiket Masuk: 50000
   Rating Wisata: 4.4

8. Taman Srigunting : Taman Hiburan
   Harga Tiket Masuk: 0
   Rating Wisata: 4.7

9. Wisata Alam Wana Wisata Penggaron : Cagar Alam
   Harga Tiket Masuk: 10000
   Rating Wisata: 4.1

10. Masjid Kapal Semarang : Tempat Ibadah
    Harga Tiket Masuk: 0
    Rating Wisata: 4.1

## Evaluasi

1. **Content-based Filtering Recommendation**

   Tahap evaluasi untuk sistem rekomendasi dengan _content-based filtering_ dapat menggunakan metrik _precision_.

   $$precision = \frac{TP}{TP + FP}$$

   Di mana:
   $TP =$ _True Positive_; rekomendasi yang sesuai
   $FP =$ _False Positive_; rekomendasi yang tidak sesuai

   Berdasarkan hasil rekomendasi tempat wisata dengan pendekatan _content-based filtering_ dapat dilihat bahwa hasil yang diberikan oleh sistem rekomendasi berdasarkan tempat wisata **Tugu Muda Semarang** dengan kategori **Budaya**, menghasilkan 5 rekomendasi judul tempat wisata yang tepat. Tetapi secara keseluruhan sistem merekomendasikan tempat wisata dengan tepat.

   $$precision = \frac{5}{5 + 0} = 100\%$$

   Dengan begitu, diperoleh nilai _precision_ sebesar **100%**.

2. **Collaborative Filtering Recommendation**

   Tahap evaluasi untuk sistem rekomendasi dengan _collaborative filtering_ menggunakan metrik RMSE (Root Mean Squared Error). Rumus untuk mencari nilai RMSE sebagai berikut,

   $$RMSE=\sqrt{\sum^{n}_{i=1} \frac{y_i - y\\_pred_i}{n}}$$

   Di mana:
   $n =$ jumlah _dataset_
   $i =$ urutan data dalam _dataset_
   $y_i =$ nilai yang sebenarnya
   $y_{pred} =$ nilai prediksi terhadap $i$

   Nilai RMSE dari sistem rekomendasi dengan pendekatan _collaborative filtering_ adalah 0.3437 pada _Training RMSE_, dan 0.3695 pada _Validation RMSE_. Sedangkan untuk nilai _training loss_ sebesar 0.6909, dan _validation loss_ sebesar 0.7109.

## Kesimpulan

Dengan begitu, dapat disimpulkan bahwa sistem berhasil melakukan rekomendasi baik dengan pendekatan _content-based filtering_ maupun _collaborative filtering_. _Collaborative filtering_ membutuhkan data penilaian tempat dari pengguna, sedangkan pada _content-based filtering_, data rating tidak dibutuhkan karena sistem akan merekomendasikan berdasarkan konten tempat tersebut, yaitu kategori.

