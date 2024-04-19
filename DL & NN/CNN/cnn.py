from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

classifier = Sequential()

classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 32

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='binary'
        )

test_set = test_datagen.flow_from_directory(
            'dataset/test_set',
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='binary'
        )

# fine-tune the model
classifier.fit(
            train_set,
            steps_per_epoch=8000 // batch_size,
            epochs=20,
            validation_data=test_set,
            validation_steps=2000 // batch_size
        )