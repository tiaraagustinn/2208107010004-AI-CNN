# Uji coba lagi di test_set
import numpy as np
from keras.preprocessing import image
count_dog = 0
count_cat = 0
for i in range(4001, 5001): 
    test_image = image.load_img('dataset/test_set/dogs/dog.' + str(i) + '.jpg', target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = MesinKlasifikasi.predict(test_image)
    training_set.class_indices
    if result[0][0] == 0:
        prediction = 'cat'
        count_cat = count_cat + 1
    else:
        prediction = 'dog'
        count_dog = count_dog + 1

# Mencetak hasil prediksinya agar bisa dibaca
print("count_dog:" + str(count_dog))    
print("count_cat:" + str(count_cat))
