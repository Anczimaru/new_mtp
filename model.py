
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import sys
import os
import config

NUM_CLASSES = len(config.CLASS)
SIZE = config.FILTER_SIZE

def params():
    params = {}
   
    #first layer
    params['w_conv1'] = tf.get_variable(
        'w_conv1',
        shape = [SIZE,SIZE, config.CNN_IN_CH, 32],
        initializer = tf.contrib.layers.xavier_initializer())
    params['b_conv1'] = tf.Variable(tf.constant(0.1, shape=[32]))
    
    #second layer
    params['w_conv2'] = tf.get_variable(
        'w_conv2',
        shape = [SIZE,SIZE, 32, 64],
        initializer = tf.contrib.layers.xavier_initializer())
    params['b_conv2'] = tf.Variable(tf.constant(0.1, shape=[64]))
    
        
        
    #third layer    
    params['w_conv3'] = tf.get_variable(
        'w_conv3',
        shape = [SIZE,SIZE, 64, 128],
        initializer = tf.contrib.layers.xavier_initializer())
    params['b_conv3'] = tf.Variable(tf.constant(0.1, shape=[32]))
    
    
    #fc layer
    params['w_fc1'] = tf.get_variable(
        'w_fc1',
        shape = [16*4*128, 2048],
        initializer = tf.contrib.layers.xavier_initializer())
    params['b_fc1'] = tf.Variable(tf.constant(0.1, shape=[2048]))
    
    #fc layer 2
    params['w_fc2'] = tf.get_variable(
        'w_fc2',
        shape = [2048, NUM_CLASSES],
        initializer = tf.contrib.layers.xavier_initializer())
    params['b_fc2'] = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
    
    
    
def my_conv_layer(x, name, params, padding = "SAME"):
        w = "w_"+ name
        b = "b_"+ name
        with tf.name_scope(name):
            layer = tf.nn.relu(tf.nn.conv2d(x,
                                            params[w], 
                                            [1,1,1,1],
                                            padding = padding)) + params[b]
        return layer


def my_pool_layer(x, ksize=(2, 2), stride=(2, 2)):
    layer = tf.nn.max_pool(x, 
                          ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1],
                          padding='SAME')
         
    return layer 
        
def my_fc_layer(x,name,params):
    w = "w_"+ name
    b = "b_"+ name
    with tf.name_scope(name):
        layer = tf.nn.relu(tf.matmul(x, model_params[w]) + params[b])
        return layer
        
        
        
    return
def cnn(data, model_params, keep_prob = config.KEEP_PROB):
    conv_1 = my_conv_layer(data, "conv1", model_params)
    pool_1 = my_pool_layer(conv_1)
        
        
    conv_2 = my_conv_layer(pool_1, "conv2", model_params)
    pool_2 = my_pool_layer(con_2)
        
    conv_3 = my_conv_layer(pool_2, "conv3", model_params)
    pool_3 = my_pool_layer(conv_3)
        
        
    #fc layers
    conv3_layer_flat = tf.reshape(h_pool3, [-1, 16 * 4 * 128])
    
    fc_1 = my_fc_layer(conv3_layer_flat, "fc1", params)
    dropout = tf.nn.dropout(fc_1, keep_prob)
    
    fc_2 = my_fc_layer(dropout, "fc2", params)
        
    return fc_2

