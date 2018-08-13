
# coding: utf-8



import os




_all_ = ('CLASS',
         "CNN_IN_WIDTH", "CNN_IN_HEIGHT", "CNN_IN_CH", "CNN_SHAPE",
         "CNN_OUTPUT_SIZE",
         "FILTER_SIZE", "DATA_DIR", "PIC_SRC_DIR", "BG_PIC_SRC_DIR",
         "TRAIN_DIR", "TEST_DIR", "CROOPED_DIR","RESULT_DIR",
         "LABEL_ORG", "TFRECORD_NAMES", "GEN_NUM_PIC","PUT_EFFECTS","ONLY_INJECTED",
         "KEEP_PROB", "PRINT_NTH","BATCH_SIZE")
#NN settings
CLASS = {'MARKER'}
CNN_IN_WIDTH = 256
CNN_IN_HEIGHT = 256
CNN_IN_CH = 3
CNN_SHAPE = (CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)
CNN_OUTPUT_SIZE = 3
FILTER_SIZE = 5

#Directories
DATA_DIR = "data"
PIC_SRC_DIR = os.path.join(DATA_DIR, "pictures")
BG_INSTALL_DIR = os.path.join(DATA_DIR,"background")
BG_PIC_SRC_DIR = os.path.join(BG_INSTALL_DIR,"images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VAL_DIR = os.path.join(DATA_DIR,"validation")
RESULT_DIR = "results"

#Data generation settings
LABEL_ORG = os.path.join(DATA_DIR, "label_org.npy") #orginal labels
TFRECORD_NAMES = ['train.tfrecords','test.tfrecords','validation.tfrecords']
GEN_NUM_PIC = 10
PUT_EFFECTS = True
ONLY_INJECTED = True

#NN params
KEEP_PROB = 0.7
PRINT_NTH = 100
BATCH_SIZE = 20
