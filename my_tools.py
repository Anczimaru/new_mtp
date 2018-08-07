import PIL
import numpy as np
import config
import os
from matplotlib import pyplot as plt

def show_image(n):
    """
    Give index get photo
    """
    
    img, label = get_image_with_label(n)
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
    img = PIL.Image.open(path_to_file)
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
    
    
    coordinates = ((int(label[1]),int(label[2])),(int(label[3]),int(label[4])))
    base = PIL.Image.open(path_to_file).convert('RGBA')
    box = PIL.Image.new('RGBA',base.size,(255,255,255,0))
    d = PIL.ImageDraw.Draw(box)
    d.rectangle(coordinates,outline=(0,0,0,255))
    out = PIL.Image.alpha_composite(base, box)
    show_opened_image(out)