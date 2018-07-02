# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:56:36 2017

@author: Administrator
"""

import tensorflow as tf
import ops

class Discriminator:
    def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
        self.name = name
        self.norm = norm
        self.is_training = is_training
        self.use_sigmoid = use_sigmoid
        self.reuse = False
        
    def __call__(self, input):
        with tf.variable_scope(self.name):
            C64 = ops.Ck(input, 64, reuse=self.reuse, norm=None,
                         is_training=self.is_training, name='C64')
            C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C128')
            C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C256')
            C512 = ops.Ck(C256, 512, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C512')
            
            output = ops.last_conv(C512, reuse=self.reuse, 
                                   use_sigmoid=self.use_sigmoid, name='output')
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        return output
    



