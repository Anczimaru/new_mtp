
# coding: utf-8

# In[1]:


import tensorflow as tf
import sys
import os
import config
import shutil
import my_tools
from model import Model as Model

print("loaded libs")


# In[2]:


params = {"result_dir": config.RESULT_DIR,
          "learning_rate": 1e-4,
          "img_size": config.CNN_IN_HEIGHT,
          "num_channels":config.CNN_IN_CH,
          "num_classes": config.CNN_OUTPUT_SIZE,
          "batch_size": config.BATCH_SIZE,
          "keep_probability": config.KEEP_PROB,
          "print_nth_step": config.PRINT_NTH}
print("loaded params")


"""def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):

     The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate)
    return log_dir
"""



def main():
    result_dir = config.RESULT_DIR
    x = input("Do you want to rerun datacreation?[y/n]")
    if x == 'y' :
        os.system("python3 gen.py")
        os.system("python3 dataset_creator.py")

    x = input("Do you want to remove previous results?[y/n]")
    if x == 'y':
        remove_results = True
    else: remove_results = False
    if remove_results == True:
        shutil.rmtree(result_dir, ignore_errors=True)
    #Check for dirs, if not present make them
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    ckpt_dir=os.path.join(result_dir,"ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)


    graph = tf.Graph()
    #Get name as default graph
    with graph.as_default():


    #Initialization of Model, load all Model functions returning variables
        model = Model(params,result_dir, ckpt_dir)
        n_times = 2
        model.train_n_times(result_dir, n_times)


# In[6]:


### tf.reset_default_graph()
if __name__ == '__main__':

    main()





"""
tensorboard --logdir="results/"
"""
