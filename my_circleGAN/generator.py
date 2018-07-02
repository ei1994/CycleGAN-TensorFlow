# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:37:43 2017

@author: universe
"""

import tensorflow as tf
import ops
import utils

class Generator:
    def __init__(self, name, is_training, ngf=64, norm='instance', image_size=128):
        self.name = name
        self.is_training = is_training
        self.ngf = ngf
        self.norm = norm
        self.image_size = image_size
        self.reuse = False
        
    def __call__(self, input):
       """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
       """
       with tf.variable_scope(self.name):
           c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
                                reuse=self.reuse, name='c7s1_32')
           d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                        reuse=self.reuse, name='d64')
           d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='d128')
           
           if self.image_size <=128:
               res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6)
           else:
               res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)
               
           # fractional-strided convolution
           u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                        reuse=self.reuse, name='u64')
           u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm, 
                        reuse=self.reuse, name='u32', output_size=self.image_size)
           
           output = ops.c7s1_k(u32, 3, norm=None,
                               activation='tanh', reuse=self.reuse, name='output')
         
       self.reuse = True
       self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
       return output
    def sample(self, input):
        image = utils.batch_convert2int(self.__call__(input))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image
           
       
                        
                   



           
           
           
           
           
           
           
           
           
           



