import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.utils import Sequence # type: ignore
import numpy as np

@tf.keras.utils.register_keras_serializable(package="face_detection")
def regression_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]
    h_pred = yhat[:,3] - yhat[:,1]
    w_pred = yhat[:,2] - yhat[:,0]
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_coord + delta_size

@tf.keras.utils.register_keras_serializable(package="face_detection")
def classification_loss():
    return tf.keras.losses.BinaryCrossentropy()

@tf.keras.utils.register_keras_serializable(package="face_detection")
def optimizerpro():
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return opt

@tf.keras.utils.register_keras_serializable(package="face_detection")
class FaceDetection(Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compile(self, optimizer, classloss, regressloss):
        super().compile()
        self.opt = optimizer
        self.closs = classloss
        self.rloss = regressloss

    def train_step(self, batch):
        X, y = batch
        
        y_class = tf.reshape(y[0], (-1, 1))
        y_bbox = tf.cast(y[1], tf.float32)
        
        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y_class, classes)
            batch_regressloss = self.rloss(y_bbox, coords)
            total_loss = 1.5 * batch_regressloss + 0.5 * batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_regressloss}

    def test_step(self, batch):
        X, y = batch

        y_class = tf.reshape(y[0], (-1, 1))
        y_bbox = tf.cast(y[1], tf.float32)

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y_class, classes)
        batch_regressloss = self.rloss(y_bbox, coords)
        total_loss = 1.5 * batch_regressloss + 0.5 * batch_classloss
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_regressloss}


    def call(self, X):
        return self.model(X)

    def get_config(self):
        return {
            "model": self.model.get_config(),
            "optimizer": tf.keras.utils.serialize_keras_object(self.opt),
            "classloss": tf.keras.utils.serialize_keras_object(self.closs),
            "regressloss": tf.keras.utils.serialize_keras_object(self.rloss),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = Model.from_config(config["model"], custom_objects=custom_objects)
        optimizer = tf.keras.utils.deserialize_keras_object(config["optimizer"], custom_objects=custom_objects)
        classloss = tf.keras.utils.deserialize_keras_object(config["classloss"], custom_objects=custom_objects)
        regressloss = tf.keras.utils.deserialize_keras_object(config["regressloss"], custom_objects=custom_objects)
        instance = cls(model)
        instance.compile(optimizer=optimizer, classloss=classloss, regressloss=regressloss)
        return instance

    def get_compile_config(self):
        return {
            "optimizer": self.opt,
            "classloss": self.closs,
            "regressloss": self.rloss,
        }

    @classmethod
    def compile_from_config(cls, config):
        optimizer = tf.keras.utils.deserialize_keras_object(config["optimizer"])
        classloss = tf.keras.utils.deserialize_keras_object(config["classloss"])
        regressloss = tf.keras.utils.deserialize_keras_object(config["regressloss"])
        return optimizer, classloss, regressloss
    
def load_model(path):
    return tf.keras.models.load_model(path)

class FACE_DETECTION:
    def __init__(self, image, model):
        self.image = image
        self.model = model
        self.classes = None
        self.coords = None
        self.original_image = image.copy()

    def preprocess(self):
        self.image = cv2.resize(self.image, (224, 224))
        self.image = self.image / 255.0
        self.image = np.expand_dims(self.image, axis=0)
        return self.image
    
    def predict(self):
        self.image = self.preprocess()
        self.classes, self.coords = self.model.predict(self.image)
        return self.classes, self.coords
    
    def draw(self):
        h, w, _ = self.original_image.shape
        if self.classes is None or self.coords.all() is None:
            self.predict()

        if self.classes[0][0] > 0.5:
            x1, y1, x2, y2 = self.coords[0]
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            cv2.rectangle(self.original_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.show()

    def get_coords(self):
        h, w, _ = self.original_image.shape
        if self.classes is None or self.coords.all() is None:
            self.predict()
        x1, y1, x2, y2 = self.coords[0]
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        return [x1, y1, x2, y2]

if __name__ == "__main__":
    model = load_model("Models/FaceDetectionModelV15.keras")
    image = cv2.imread("ImagesTest/face1.jpg")
    face_detection = FACE_DETECTION(image, model)

    # Get coordinates and classes
    #classes, coords = face_detection.predict()

    # Draw the bounding box
    face_detection.draw()
    