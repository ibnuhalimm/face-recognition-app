import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
import os
from keras.preprocessing import image
import numpy as np


class Recognition:
    def __init__(self):
        self.photo_image = None
        self.names = np.array([
            "Ibnu",
            "Iin",
            "Kriswanto",
            "Mustofa"
        ])

        self.img_width = 128
        self.img_height = 128

        self.model_file = os.path.join(os.getcwd(), 'model.h5')
        self.model = tf.keras.models.load_model(self.model_file)

        #self.model_weight_file = os.path.join(os.getcwd(), 'model_weight.h5')
        #self.model = self.create_model()
        #self.model.load_weights(self.model_weight_file)

        self.predictions = None

    def get_model_summary(self):
        return self.model.summary()

    def set_photo_image(self, photo_image):
        self.photo_image = photo_image

    def get_name(self):
        img = image.load_img(self.photo_image,
                             target_size=(self.img_width, self.img_height))
        image_to_predict = np.expand_dims(img, axis=0)
        self.predictions = self.model.predict(image_to_predict)
        class_index = np.argmax(self.predictions, axis=1)

        return self.names[class_index][0]

