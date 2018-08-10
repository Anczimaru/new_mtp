
# coding: utf-8

# In[ ]:


import tensorflow
import os
import config
import my_tools
import numpy as np 

def main():
    path_to_record = os.path.join(config.DATA_DIR,config.TFRECORD_NAMES[0])
    record_iterator = tf.python_io.tf_record_iterator(path=path_to_record)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        x1 = (example.features.feature['x1']
                                     .float_list
                                     .value[0])


        x2 = (example.features.feature['x2']
                                     .float_list
                                     .value[0])

        y1 = (example.features.feature['y1']
                                     .float_list
                                     .value[0])


        y2 = (example.features.feature['y2']
                                     .float_list
                                     .value[0])

        index =(example.features.feature['index']
                                        .int64_list
                                        .value[0])

        img_string = (example.features.feature['image']
                                      .bytes_list
                                      .value[0])
    
    
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((256, 256, -1))
    
    print(index)
    print(x1,y1,x2,y2)
    my_tools.show_opened_image(reconstructed_img)

