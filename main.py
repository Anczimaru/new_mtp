
# coding: utf-8

# In[1]:


import tensorflow as tf
import sys
import os
import config
import shutil
import my_tools
import numpy as np
from model import Model as Model
import random

print("loaded libs")


# In[2]:


params = {"root_result_dir": config.RESULT_DIR,
          "learning_rate": 1e-4,
          "img_size": config.CNN_IN_HEIGHT,
          "num_channels":config.CNN_IN_CH,
          "num_classes": config.CNN_OUTPUT_SIZE,
          "batch_size": config.BATCH_SIZE,
          "keep_probability": config.KEEP_PROB,
          "print_nth_step": config.PRINT_NTH}
print("loaded params")


def gen_log_dir_name(n=0,learning_rate = params["learning_rate"],
                     num_classes = params["num_classes"],
                     root_result_dir = config.RESULT_DIR):
    """
    Generate new name for directory
    """
    r = n
    word = ["alpha", "beta", "gamma", "delta", "kappa", "epsilon"]
    if r == 6:
        print("Please review your results, number of directories is pretty high")
        symbol = random.randint
    else:
        symbol = word[n]
    s = "lr_{0:.0e}_classes_{1}_{2}"
    # Insert all the hyper-parameters in the dir-name.
    log_dir_name = str(s.format(learning_rate,num_classes,symbol))
    log_dir = os.path.join(root_result_dir, log_dir_name)
    if not os.path.exists(log_dir):
        return log_dir
    else:
        return gen_log_dir_name(n=r+1)

def get_log_dir_name(result_root_dir = config.RESULT_DIR):
    """
    Generete new directory or load existing one
    """
    #### LOAD EXISTING DIR
    x = input("do you want to load existing directory?[y/n]")
    if x == "y":
        dir_list = os.listdir(result_root_dir)
        print(dir_list)
        dir_num = int(input("please input number of desired directory"))
        try:
            log_dir_name = str(dir_list[(dir_num-1)])
            print("Selected directory: ",log_dir_name)
            x1 = input("proceed?[y/n]")
            if x1 == "y":
                log_dir = os.path.join(result_root_dir, log_dir_name)
                return log_dir
            else: get_log_dir_name()
        except IndexError as e:
            print(e)
            print("calling function again, put proper number please")
            get_log_dir_name()

    ####CREATE NEW DIR
    else:
        x2 = input("Do you wish to create new directory?[y/n]")
        if x2 == "y":
            log_dir = gen_log_dir_name()
            print(log_dir)
        else: get_log_dir_name()

def main(quick_start = False):
    quick_start = True
    if quick_start == False:
        result_dir = config.RESULT_DIR
        x = input("Do you want to rerun datacreation?[y/n]")
        if x == 'y' :
            os.system("python3 gen.py")
            os.system("python3 dataset_creator.py")

        if not os.path.exists(config.RESULT_DIR):
            os.makedirs(config.RESULT_DIR)

        log_dir = get_log_dir_name()
        x = input("How many times you want to go through dataset?")
        n_times = int(x)


    else:
        n_times = 2
        log_dir = gen_log_dir_name()


    #Check for dirs, if not present make them
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)



    tf.reset_default_graph()

    graph = tf.Graph()
    #Get name as default graph
    with graph.as_default():



    #Initialization of Model, load all Model functions returning variables
        model = Model(params,log_dir,log_dir)
        model.train_n_times(log_dir, n_times)





if __name__ == '__main__':

    main()





"""
tensorboard --logdir="results/"
"""
