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
    labels = labels.astype(np.float32)
    dataset, labels = randomize(dataset,labels)
    return dataset, labels
   
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = tf.convert_to_tensor(dataset[permutation])
    shuffled_labels = tf.convert_to_tensor(labels[permutation])
    
    return shuffled_dataset, shuffled_labels
