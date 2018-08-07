import PIL
import numpy as np
import config
import os
from matplotlib import pyplot as plt

def show_image(n):
    img, label = get_image_with_label(n)
    print(label)
    plt.imshow(img)
    plt.show()
    
    
    
    
def get_image_with_label(n):
    """
    Returns img with Image module and corresponding label
    """
    labels = np.load(config.LABEL_ORG)
    label = labels[n]
    img_name = str(n)+".png"
    path_to_file = os.path.join(config.PIC_SRC_DIR,img_name)
    img = PIL.Image.open(path_to_file)
    return img, label