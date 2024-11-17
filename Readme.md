# Review
Berikut adalah review dari code yang ada pada link

---
Link 2-1
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


---
Link 2-2
##1. Import Library
```python
import numpy as np
from keras.preprocessing import image
```
- `numpy`: Digunakan untuk manipulasi array, termasuk konversi gambar menjadi array untuk diproses oleh model.
- `keras.preprocessing.image`: Digunakan untuk memuat dan memproses gambar sebelum dikirim ke model.

##2. Inisialisasi Variabel
```python
count_dog = 0
count_cat = 0
```
`count_dog` dan `count_cat`: Variabel penghitung jumlah prediksi untuk kategori anjing dan kucing.

##3. Perulangan untuk Memproses Gambar
```python
for i in range(4001, 5001):
```
Melakukan iterasi untuk 1000 gambar dengan nama berformat `dog.<nomor>.jpg` (dari 4001.jpg hingga `5000.jpg`) di `folder test_set/dogs/`.

##4. Memuat dan Mengolah Gambar
```python
test_image = image.load_img('dataset/test_set/dogs/dog.' + str(i) + '.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
```
- `image.load_img()`: Memuat gambar dari path yang diberikan dan mengubah ukuran menjadi (128, 128).
- `image.img_to_array()`: Mengonversi gambar menjadi array numerik.
- `np.expand_dims()`: Menambahkan dimensi baru (batch dimension) sehingga array memiliki bentuk (1, 128, 128, 3), yang diperlukan oleh model untuk prediksi.

##5. Prediksi Gambar
```python
result = MesinKlasifikasi.predict(test_image)
training_set.class_indices
```
- `MesinKlasifikasi.predict(test_image)`: Menggunakan model yang sudah dilatih (MesinKlasifikasi) untuk memprediksi label gambar.
- `training_set.class_indices`: Menyimpan mapping antara label kelas dan indeksnya (misal, {'cat': 0, 'dog': 1}).

##6. Menginterpretasikan Hasil Prediksi
```python
if result[0][0] == 0:
    prediction = 'cat'
    count_cat = count_cat + 1
else:
    prediction = 'dog'
    count_dog = count_dog + 1
```
-`result[0][0] == 0`: Jika hasil prediksi untuk gambar adalah 0, gambar dianggap sebagai kucing.
- `else`: Jika hasil prediksi adalah 1, gambar dianggap sebagai anjing.
- Variabel penghitung diperbarui berdasarkan kategori prediksi.

##7. Mencetak Hasil
```python
print("count_dog:" + str(count_dog))    
print("count_cat:" + str(count_cat))
```
- `count_dog`: Total gambar yang diprediksi sebagai anjing.
- `count_cat`: Total gambar yang diprediksi sebagai kucing.

---

Link 3-1
## 1. Memuat Dataset CIFAR-10
```python
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
```
- CIFAR-10 adalah dataset yang terdiri dari 60.000 gambar berukuran 32x32 piksel dengan 3 channel warna (RGB).
- Terdiri dari 10 kelas, seperti pesawat, mobil, burung, dll.
- Dataset dibagi menjadi:
    - 50.000 gambar untuk pelatihan (train_images, train_labels).
    - 10.000 gambar untuk pengujian (test_images, test_labels).

## 2. Normalisasi Data
```python
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
```
- Nilai piksel gambar awalnya dalam rentang 0-255.
- Normalisasi membagi semua nilai piksel dengan 255 untuk merubahnya menjadi rentang 0-1, sehingga lebih cocok untuk digunakan dalam pelatihan model.

## 3. Konversi Label ke Bentuk Kategorikal
```python
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
```
- `to_categorical` mengubah label integer (0 hingga 9) menjadi bentuk one-hot encoding.
Misalnya, label 3 menjadi [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].

## 4. Membangun Model CNN
```python
model = tf.keras.Sequential([
    ...
])
```
Model CNN terdiri dari beberapa lapisan:


## 5. Ringkasan Model
```python
model.summary()
```
Menampilkan arsitektur model, termasuk jumlah parameter yang perlu dilatih.

## 6. Kompilasi Model
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
- Optimizer: adam, digunakan untuk memperbarui bobot model selama pelatihan.
- Loss: categorical_crossentropy, digunakan karena ini adalah masalah klasifikasi dengan lebih dari dua kelas.
- Metrics: accuracy, untuk mengukur akurasi selama pelatihan dan pengujian.

## 7. Pelatihan Model
```python
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
```
- `epochs=10`: Model dilatih selama 10 iterasi penuh melalui dataset.
- `batch_size=64`: Data dilatih dalam batch berukuran 64 gambar sekaligus.
- `validation_data`: Data pengujian digunakan untuk mengevaluasi kinerja model pada setiap epoch.

## 8. Evaluasi Model
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
- Model diuji pada data pengujian untuk menghitung loss dan akurasi.
- `test_acc` menunjukkan akurasi model pada data pengujian.

---

Link 3-2
## 1. Mengimpor Library
```python
from google.colab import files
from keras.models import load_model
from PIL import Image
import numpy as np
```
- `google.colab.files`: Digunakan untuk mengunggah file ke lingkungan Google Colab.
- `keras.models.load_model`: Untuk memuat model yang sudah dilatih.
- `PIL.Image`: Untuk membuka dan memproses file gambar.
- `numpy`: Digunakan untuk manipulasi array, termasuk normalisasi data gambar.

## 2. Fungsi untuk Memuat dan Memproses Gambar
```python
def load_and_prepare_image(file_path):
    img = Image.open(file_path)          # Membuka file gambar
    img = img.resize((32, 32))           # Mengubah ukuran gambar menjadi 32x32 piksel
    img = np.array(img) / 255.0          # Mengonversi gambar menjadi array dan menormalisasi nilainya ke rentang 0-1
    img = np.expand_dims(img, axis=0)    # Menambahkan dimensi batch (bentuk array menjadi (1, 32, 32, 3))
    return img
```
Fungsi ini memastikan gambar dalam format dan ukuran yang sesuai dengan input model, yaitu ukuran 32x32 piksel dengan 3 channel warna (RGB), dan memiliki dimensi batch.

## 3. Daftar Nama Kelas
```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```
- CIFAR-10 memiliki 10 kelas, masing-masing diberi nama deskriptif, seperti "airplane", "automobile", dll.
- Kode ini menggunakan daftar tersebut untuk menerjemahkan indeks prediksi menjadi nama kelas.

## 4. Memuat Model
```python
model = load_model('path_to_your_model.h5')
```
- `load_model` digunakan untuk memuat model yang sudah dilatih dan disimpan dalam format .h5.
- Path ke model harus diganti dengan lokasi sebenarnya dari model Anda.

## 5. Mengunggah File Gambar
```python
uploaded = files.upload()
```
- Membuka antarmuka unggah file di Google Colab untuk mengunggah gambar yang akan diproses.
- Semua file yang diunggah akan tersedia di dictionary uploaded, dengan nama file sebagai kunci.

## 6. Memproses Gambar dan Membuat Prediksi
```python
for filename in uploaded.keys():
    img = load_and_prepare_image(filename)             # Memuat dan memproses gambar
    prediction = model.predict(img)                   # Melakukan prediksi pada gambar
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Mendapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class_name = class_names[predicted_class_index] # Mendapatkan nama kelas dari daftar `class_names`
    print(f'File: {filename}, Predicted Class Index: {predicted_class_index}, Predicted Class Name: {predicted_class_name}')
```
- `model.predict(img)`: Model memberikan output berupa probabilitas untuk setiap kelas.
 `np.argmax(prediction, axis=1)[0]`: Mendapatkan indeks kelas dengan probabilitas tertinggi dari hasil prediksi.
- `class_names[predicted_class_index]`: Mengonversi indeks kelas menjadi nama kelas.

