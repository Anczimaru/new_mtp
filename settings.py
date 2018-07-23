
# coding: utf-8



import os




_all_ = ('CLASS', 
         "CNN_IN_WIDTH", "CNN_IN_HEIGHT", "CNN_IN_CH", "CNN_SHAPE", 
         "DATA_DIR", "PIC_SRC_DIR", "BG_PIC_SRC_DIR", "BATCHES_SRC_DIR",
         "TRAIN_DIR", "TEST_DIR", "CROOPED_DIR","RESULT_DIR", 
         "LABEL_ORG", "LABEL_FILE", 
         "LR","KEEP_PROB", "PRINT_NTH")

CLASS = {'MARKER', 'BACKGROUND'}
CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32
CNN_IN_CH = 3
CNN_SHAPE = (CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)
DATA_DIR = "./data"
PIC_SRC_DIR = os.path.join(DATA_DIR, "pictures")
BG_PIC_SRC_DIR = os.path.join(DATA_DIR,"background")
BATCHES_SRC_DIR = os.path.join(DATA_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
CROPPED_DIR = os.path.join(TRAIN_DIR, "cropped")
RESULT_DIR = "./results"
LABEL_ORG = os.path.join(DATA_DIR, "label_org") #orginal labels
LABEL_FILE = os.path.join(TRAIN_DIR, "labels_with_bg") #proccesed labels

LR = 1e-4
KEEP_PROB = 0.7
PRINT_NTH = 10

