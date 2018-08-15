# coding: utf-8
import tensorflow as tf
import os
import config
import my_tools
import numpy as np



def parser(serialized_data):
    """
    Decode and map data properly for model.py
    """

    features = {"image" : tf.FixedLenFeature((),tf.string),
                   "class" : tf.FixedLenFeature((),tf.int64),
                   "x1" : tf.FixedLenFeature((),tf.float32),
                   "y1" : tf.FixedLenFeature((),tf.float32),
                   "x2" : tf.FixedLenFeature((),tf.float32),
                   "y2" : tf.FixedLenFeature((),tf.float32),
                   "x" : tf.FixedLenFeature((),tf.float32),
                   "y" : tf.FixedLenFeature((),tf.float32),
                   "index" : tf.FixedLenFeature((),tf.int64)}
    parsed_features = tf.parse_single_example(serialized_data, features)

    #Convert data to proper formats
    image_string = parsed_features["image"]
    image = tf.decode_raw(image_string, tf.uint8)
    image = tf.cast(image,dtype=tf.float32)

    label_class = tf.convert_to_tensor(parsed_features["class"])
    label_class = tf.cast(label_class,tf.float32)

    label_loc = tf.convert_to_tensor((
                     parsed_features["x"],
                     parsed_features["y"]))
    box = tf.convert_to_tensor((
                     parsed_features["x1"],
                     parsed_features["y1"],
                     parsed_features["x2"],
                     parsed_features["y2"]))
    index = tf.cast(parsed_features["index"], tf.int32)

    return image, label_class, label_loc, index, box



def main():
    """
    If you with to check structure of TFRecord run this funciton, it prints out every information from TFRecords created by dataset_creator.py
    """
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

        cl =(example.features.feature['class']
                                        .int64_list
                                        .value[0])

        img_string = (example.features.feature['image']
                                      .bytes_list
                                      .value[0])
        x = (example.features.feature['x']
                                     .float_list
                                     .value[0])

        y = (example.features.feature['y']
                                     .float_list
                                     .value[0])

        print(index)
        print(cl)
        print(x1,y1,x2,y2)
        print(x,y)



if __name__ == '__main__':

    main()
