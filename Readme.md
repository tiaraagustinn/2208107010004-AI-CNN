

## 1. Mengimpor library yang diperlukan
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```
Pada bagian ini, kita mengimpor kelas dan fungsi dari Keras untuk membangun model CNN. `Sequential` digunakan untuk mendefinisikan urutan lapisan (layers) model, sementara `Conv2D`, `MaxPooling2D`, `Flatten`, dan `Dense` adalah lapisan-lapisan yang umum digunakan dalam CNN.

## 2. Inisialisasi CNN

```python
MesinKlasifikasi = Sequential()
```
`MesinKlasifikasi` adalah objek model CNN yang akan dibangun secara berurutan (sequential).

## 3. Langkah 1 - Convolution
```python
MesinKlasifikasi.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape = (128, 128, 3), activation = 'relu'))
```
- `Conv2D` adalah lapisan konvolusi, yang berfungsi untuk mengekstrak fitur dari gambar.
- `filters = 32`: Menentukan jumlah filter (atau kernel) yang akan digunakan dalam lapisan konvolusi. Di sini, ada 32 filter.
- `kernel_size = (3, 3)`: Ukuran kernel atau filter adalah 3x3 piksel.
- `input_shape = (128, 128, 3)`: Menentukan ukuran input gambar, di mana gambar yang diterima berukuran 128x128 piksel dengan 3 saluran warna (RGB).
- `activation = 'relu'`: Fungsi aktivasi ReLU (Rectified Linear Unit) digunakan untuk memperkenalkan non-linearitas dalam model.
## 4. Langkah 2 - Pooling
```python
MesinKlasifikasi.add(MaxPooling2D(pool_size = (2, 2)))
```
- `MaxPooling2D`: Lapisan pooling digunakan untuk mengurangi dimensi gambar setelah konvolusi, meminimalkan kompleksitas dan mempercepat proses komputasi.
- `pool_size = (2, 2)`: Ukuran pooling 2x2 berarti akan mengambil nilai maksimum dalam setiap blok 2x2 piksel dari hasil konvolusi.
# 5. Menambah convolutional layer
```python
MesinKlasifikasi.add(Conv2D(32, (3, 3), activation = 'relu'))
MesinKlasifikasi.add(MaxPooling2D(pool_size = (2, 2)))
```
Di sini, lapisan konvolusi dan pooling ditambahkan lagi untuk mengekstrak lebih banyak fitur dari gambar dan mengurangi dimensi gambar lebih lanjut.

## 6. Langkah 3 - Flattening
```python
MesinKlasifikasi.add(Flatten())
```
`Flatten` digunakan untuk meratakan hasil dari lapisan konvolusi dan pooling menjadi satu dimensi, agar bisa diproses oleh lapisan-lapisan berikutnya yang bersifat fully connected.

## 7. Langkah 4 - Full connection (Dense)
```python
MesinKlasifikasi.add(Dense(units = 128, activation = 'relu'))
MesinKlasifikasi.add(Dense(units = 1, activation = 'sigmoid'))
```
- `Dense(units = 128)`: Lapisan fully connected dengan 128 neuron dan fungsi aktivasi ReLU.
- `Dense(units = 1, activation = 'sigmoid')`: Lapisan output dengan satu neuron yang menggunakan fungsi aktivasi sigmoid untuk tugas klasifikasi biner (output berupa 0 atau 1).
## 8. Menjalankan CNN
```python
MesinKlasifikasi.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```
- `optimizer = 'adam'`: Optimizer Adam digunakan untuk mempercepat proses pelatihan.
- `loss = 'binary_crossentropy'`: Fungsi loss yang digunakan untuk klasifikasi biner.
- `metrics = ['accuracy']`: Metrik yang digunakan untuk mengevaluasi model adalah akurasi.

## 9. Preprocessing dan Augmentasi Gambar
```python
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
```
- `ImageDataGenerator`: Digunakan untuk melakukan augmentasi gambar (misalnya rotasi, zoom, flipping) pada data pelatihan untuk meningkatkan jumlah data pelatihan dan membuat model lebih robust.
- `rescale = 1./255`: Menormalisasi gambar dengan membagi nilai piksel dengan 255 agar berada dalam rentang [0, 1].
  
## 10. Menyiapkan Set Data Pelatihan dan Pengujian
```python
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (128, 128), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (128, 128), batch_size = 32, class_mode = 'binary')
```
- `flow_from_directory`: Digunakan untuk memuat gambar dari direktori. Set data pelatihan dan pengujian dibaca dari folder yang masing-masing berisi gambar untuk kelas 0 dan 1.
- `target_size = (128, 128)`: Gambar akan diubah ukurannya menjadi 128x128 piksel.
- `batch_size = 32`: Jumlah gambar yang diproses dalam satu batch.
- `class_mode = 'binary'`: Digunakan untuk klasifikasi biner.

## 11. Melatih Model
```python
MesinKlasifikasi.fit_generator(training_set, steps_per_epoch = 8000/32, epochs = 50, validation_data = test_set, validation_steps = 2000/32)
```
- `fit_generator`: Melatih model menggunakan generator gambar.
- `steps_per_epoch = 8000/32`: Menentukan berapa langkah yang dilakukan dalam setiap epoch (total gambar pelatihan dibagi dengan batch size).
- `epochs = 50`: Model akan dilatih selama 50 epoch.
- `validation_data = test_set`: Menggunakan data pengujian untuk evaluasi setelah setiap epoch.
- `validation_steps = 2000/32`: Menentukan jumlah langkah untuk evaluasi validasi.

