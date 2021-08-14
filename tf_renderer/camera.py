import tensorflow as tf
import numpy as np
from tensorflow_graphics.rendering.camera import perspective

class Camera(object):
    def project(self, points3d):
        raise NotImplementedError()
    def ray(self, points2d):
        raise NotImplementedError()
    
class PerspectiveCamera(Camera):
    def __init__(self, focal, principal_point, fov=30.):
        self.focal = focal
        self.principal_point = principal_point
        fov = fov * np.pi / 180.
        f = 1. / np.tan(fov/2.)
        self.f = tf.constant([f, f, -1.], dtype=tf.float32)

    def project(self, points3d):
        points3d = tf.convert_to_tensor(points3d, dtype=tf.float32) * self.f
        points2d = perspective.project(points3d, self.focal, self.principal_point)
        depth = points3d[...,2]
        return points2d, depth

    def ray(self, points2d):
        vectors3d = perspective.ray(points2d, self.focal, self.principal_point)
        vectors3d = vectors3d / self.f
        return vectors3d
