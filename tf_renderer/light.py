import tensorflow as tf
import numpy as np

class Light(object):
    def ray(self, points):
        raise NotImplementedError()

    def intensity(self, points):
        raise NotImplementedError()

class PointLight(Light):
    def __init__(self, intensity, position):
        self._intensity = intensity
        self.position = position

    def ray(self, points):
        return tf.math.l2_normalize(
            points - self.position, axis=-1)
    
    def intensity(self, points):
        dist = tf.reduce_sum(tf.square(points - self.position), axis=-1, keepdims=True)
        dist = tf.maximum(dist, 0.01)
        return self._intensity / (4. * np.pi * dist)
