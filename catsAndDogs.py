from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras imports optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

folder = 'D:/Yashdeep/Kaggle_Datasets/cats_and_dogs/train/'
dataset_home = 'D:/Yashdeep/Kaggle_Datasets/dataset_dogs_vs_cats/'


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_test_harness():
    model = define_model()

    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_it = datagen.flow_from_directory(dataset_home + 'train/', class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = datagen.flow_from_directory(dataset_home + 'test/', class_mode='binary', batch_size=64, target_size=(200, 200))

    history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)

    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    model.save('testmodel.h5')

    # prediction = model.predict(prepare('D:/Yashdeep/Kaggle_Datasets/dataset_dogs_vs_cats/test/dogs/dog.1.jpg'))
    # print(prediction)


run_test_harness()