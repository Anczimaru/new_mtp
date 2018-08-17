# coding: utf-8
from PIL import Image, ImageDraw
import numpy as np
import config
import os
from matplotlib import pyplot as plt
import functools
import tensorflow as tf

def show_image(n):
    """
    Give index get photo
    """

    img, label = get_image_with_label(n-1)
    print(label)
    show_opened_image(img)


def get_image_with_label(n):
    """
    Returns opened Image module and corresponding label
    """
    labels = np.load(config.LABEL_ORG)
    label = labels[n]
    img_name = str(n+1)+".png"
    path_to_file = os.path.join(config.PIC_SRC_DIR,img_name)
    img = Image.open(path_to_file)
    return img, label


def show_opened_image(image):

    plt.imshow(image)
    plt.show()



def show_label_on_img(n):
    """
    Give image index, returns ploted ground truth on image
    """
    labels = np.load(config.LABEL_ORG)
    label = labels[n-1]
    img_name = str(n)+".png"
    path_to_file = os.path.join(config.PIC_SRC_DIR,img_name)
    n=256

    coordinates = ((int(n*label[1]),int(n*label[2])),(int(n*label[3]),int(n*label[4])))
    base = Image.open(path_to_file).convert('RGBA')
    box = Image.new('RGBA',base.size,(255,255,255,0))
    d = ImageDraw.Draw(box)
    d.rectangle(coordinates,outline=(255,0,0,255))
    out = Image.alpha_composite(base, box)
    show_opened_image(out)

def plot_result_on_img(n, label):
    """
    Give coordinates, returns ploted results on image
    """
    img_name = str(n)+".png"
    path_to_file = os.path.join(config.PIC_SRC_DIR,img_name)
    n=1/config.MU

    coordinates = ((int(n*label[0]),int(n*label[1])),(int(n*label[2]),int(n*label[3])))
    base = Image.open(path_to_file).convert('RGBA')
    box = Image.new('RGBA',base.size,(255,255,255,0))
    d = ImageDraw.Draw(box)
    d.rectangle(coordinates,outline=(255,0,0,255))
    out = Image.alpha_composite(base, box)
    return out


def box_iou(boxA, boxB):

	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA) * max(0, yB - yA)

	boxAArea = (boxA[2] - boxA[0]+1) * (boxA[3] - boxA[1]+1)
	boxBArea = (boxB[2] - boxB[0]+1) * (boxB[3] - boxB[1]+1)

	union = float(boxAArea + boxBArea - interArea)

	return interArea/(union + 0.0001) #added epsilon for safer calculations

#Define wrappers for memmory loading funcitons and variables by model.py

def tf_box_iou(bboxes1, bboxes2):
    """
    get tf iou calculated
    """

    x11, y11, x12, y12 = tf.split(bboxes1, [1,1,1,1])
    x21, y21, x22, y22 = tf.split(bboxes2, [1,1,1,1])

    xI1 = tf.maximum(x11, tf.transpose(x21))
    xI2 = tf.minimum(x12, tf.transpose(x22))

    yI1 = tf.minimum(y11, tf.transpose(y21))
    yI2 = tf.maximum(y12, tf.transpose(y22))

    inter_area = tf.maximum((xI2 - xI1), 0) * tf.maximum((yI1 - yI2), 0)

    bboxes1_area = (x12 - x11) * (y11 - y12)
    bboxes2_area = (x22 - x21) * (y21 - y22)

    union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    return inter_area / (union+0.0001)



def doublewrap(function):
    """
    A decorator of decorator, allowing use of lazy property if no arguments are provided
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope = None, *args, **kwargs):
    """
    Lazy decorator, optimizes code by loading class Model parts only once to memory
    Also its groups tf.Graph, in tensorboard into smaller, more readable parts
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            #Sorting Graph by Var_scope
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
                print("Initialized Model.{}".format(name))
        return getattr(self, attribute)

    return decorator
