import tensorflow as tf
import os
from keras.preprocessing import image
import numpy as np


class Recognition:
    def __init__(self):
        self.photo_image = None
        self.names = np.array([
            "Ibnu",
            "Iin",
            "Irfan",
            "Kriswanto",
            "Pandu",
            "Ula",
            "Zainul"
        ])

        self.img_width = 96
        self.img_height = 96

        self.model_file = os.path.join(os.getcwd(), 'model.h5')
        self.model = tf.keras.models.load_model(self.model_file)
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

