# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:35:13 2017

@author: DELL
"""

import tensorflow as tf
import random

def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    """
    return tf.image.convert_image_dtype((image+1.0)/2, tf.uint8)

def convert2float(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image/127.5)-1.0

def batch_convert2int(images):
    return tf.map_fn(convert2int, images, dtype=tf.uint8)

def batch_convert2float(images):
    return tf.map_fn(convert2float, images, dtype=tf.float32)

class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []
        
    def query(self, image):
        if self.pool_size ==0:
            return image
        
        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image
        
        
        
        