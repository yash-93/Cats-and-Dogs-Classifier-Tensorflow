import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def prepare(filepath):
    photo = load_img(filepath, target_size=(200, 200))
    photo = img_to_array(photo)
    return photo.reshape(-1, 200, 200, 3)

print(prepare('D:/Yashdeep/Kaggle_Datasets/dataset_dogs_vs_cats/test/cats/cat.0.jpg'))
new_model = load_model("testmodel.h5")
new_model.summary()
# print(new_model.optimizer)
# print(new_model.get_weights())
prediction = new_model.predict(prepare('D:/Yashdeep/Kaggle_Datasets/dataset_dogs_vs_cats/test/dogs/dog.1.jpg'))
print(int(prediction))