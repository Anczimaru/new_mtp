
# coding: utf-8



import os




_all_ = ('CLASS', 
         "CNN_IN_WIDTH", "CNN_IN_HEIGHT", "CNN_IN_CH", "CNN_SHAPE",
         "CNN_OUTPUT_SIZE",
         "FILTER_SIZE", "DATA_DIR", "PIC_SRC_DIR", "BG_PIC_SRC_DIR",
         "TRAIN_DIR", "TEST_DIR", "CROOPED_DIR","RESULT_DIR", 
         "LABEL_ORG", "TFRECORD_NAMES", 
         "LR","KEEP_PROB", "PRINT_NTH","MAX_STEPS","BATCH_SIZE")

CLASS = {'MARKER', 'BACKGROUND'}
CNN_IN_WIDTH = 256
CNN_IN_HEIGHT = 256
CNN_IN_CH = 3
CNN_SHAPE = (CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)
CNN_OUTPUT_SIZE = 5
FILTER_SIZE = 5
DATA_DIR = "data"
PIC_SRC_DIR = os.path.join(DATA_DIR, "pictures")
BG_PIC_SRC_DIR = os.path.join(DATA_DIR,"background")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VAL_DIR = os.path.join(DATA_DIR,"validation")
RESULT_DIR = "results"
LABEL_ORG = os.path.join(DATA_DIR, "label_org.npy") #orginal labels
TFRECORD_NAMES = ['train.tfrecords','test.tfrecords','validation.tfrecords']


LR = 1e-4
KEEP_PROB = 0.7
PRINT_NTH = 10
MAX_STEPS = 100
BATCH_SIZE = 20

