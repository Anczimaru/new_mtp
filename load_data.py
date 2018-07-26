import sys
import os
import numpy as np
import tensorflow as tf

def load_dataset_fn(nth, src_dir="./data"):
    """
    Handles importing data from file to memory, also randomizes it and its labels
    """
    
    nth=str(int(nth))
    src_dir = os.path.join(src_dir,"Dataset")
    dataset_name = "data"+nth+".npy"
    label_name = "label"+nth+".npy"
    dataset_file = os.path.join(src_dir,dataset_name)
    label_name = os.path.join(src_dir,label_name)
    dataset = np.load(dataset_file)
    labels = np.load(label_name)
    dataset = dataset.astype(np.float32)
    labels_cl = labels[:,[0,1]].astype(np.float32)
    labels_loc = labels[:,[2,3,4,5]].astype(np.float32)
    del labels
    dataset,labels_cl, labels_loc = randomize(dataset,labels_cl,labels_loc)
    return dataset, labels_cl, labels_loc
   
def randomize(dataset, labels_cl, labels_loc):
    permutation = np.random.permutation(labels_cl.shape[0])
    shuffled_dataset = tf.convert_to_tensor(dataset[permutation])
    shuffled_labels_cl = tf.convert_to_tensor(labels_cl[permutation])
    shuffled_labels_loc = tf.convert_to_tensor(labels_loc[permutation])
    
    return shuffled_dataset, shuffled_labels_cl, shuffled_labels_loc
