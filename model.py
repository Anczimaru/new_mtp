# coding: utf-8
from my_tools import define_scope
import my_tools
import tensorflow as tf
import time
import os
import config
from dataset_helper import parser as parser


class Model(object):
    """
    Model of NN
    """

### INITIALIZATION ####
    def __init__(self, params, log_dir,ckpt_dir):
        with tf.name_scope("initialize"):
            # main initialization for whole model, reads directed params, creates neccesary variables
            #Variables
            self.result_dir = log_dir
            self.ckpt_dir = ckpt_dir
            self.img_size = params["img_size"]
            self.num_channels = params["num_channels"]
            self.num_classes = params["num_classes"]
            self.lr = params["learning_rate"]
            self.keep_prob = params["keep_probability"]
            self.write_step = params["print_nth_step"]
            self.is_training = True
            self.global_step = tf.train.get_or_create_global_step(graph=tf.get_default_graph())


        #FUNCTIONS AND OPS INITIALIZATION, IMPORTANT!!
        self.data_pipeline
        self.prediction
        self.loss_op
        self.summary_op
        self.optimize



#FUNCTION DEFINITIONS
    @define_scope
    def data_pipeline(self):
        """
        Function handling loading into memery previosly creater TFRecords file, it maps variables due to rules specified in parser, and creates initializers for datasets
        """
        #Get datasets
        train_dataset = tf.data.TFRecordDataset(os.path.join(config.DATA_DIR,config.TFRECORD_NAMES[0]))
        test_dataset = tf.data.TFRecordDataset(os.path.join(config.DATA_DIR,config.TFRECORD_NAMES[2]))
        val_dataset = tf.data.TFRecordDataset(os.path.join(config.DATA_DIR, config.TFRECORD_NAMES[1]))
        train_dataset =train_dataset.map(parser)
        test_dataset = test_dataset.map(parser)
        val_dataset = val_dataset.map(parser)

        #Create main iterator stucture
        iterator = tf.data.Iterator.from_structure( train_dataset.output_types,
                                                    train_dataset.output_shapes)

        #Get next command
        self.data, target_class, target_center, self.index, target_loc = iterator.get_next()

        #Data reshaping operations
        self.target_class = tf.cast(tf.reshape(target_class, shape= [1]),dtype=tf.float32)
        self.target_loc = tf.reshape(target_loc, shape = [4])
        self.data = tf.reshape(self.data,[-1, self.img_size, self.img_size, self.num_channels])

        #Creation of sub initializers for datasets
        self.train_init = iterator.make_initializer(train_dataset)
        self.test_init = iterator.make_initializer(test_dataset)
        self.val_init = iterator.make_initializer(val_dataset)


 ##### NEURAL NETWORK #####
    @define_scope
    def prediction(self):
        """
        Main body of neural network, takes data and labels as input,
        returns feature map of photo
        """
        #1 conv layer
        conv1 = tf.layers.conv2d(inputs = self.data,
                             filters = 32,
                             kernel_size = [5,5],
                             strides = 1,
                             padding = "same",
                             activation = tf.nn.relu,
                             name = 'conv1')

        #1 pool layer, img size reduced by 1/4
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size = 2,
                                        strides = 2,
                                        padding = "same",
                                        name = 'pool1')

        #2 conv layer
        conv2 = tf.layers.conv2d(inputs = pool1,
                             filters = 64,
                             kernel_size = 5,
                             strides = 1,
                             padding = "same",
                             activation = tf.nn.relu,
                             name = 'conv2')

        #2 pool overal image size reduced totaly by factor of 1/16
        pool2 = tf.layers.max_pooling2d(inputs = conv2,
                                        pool_size = 2,
                                        strides = 2,
                                        padding = "same",
                                        name = 'pool2')

        #Pool layer reshaping for fully conected layer
        pool2_flat = tf.reshape(pool2,[-1, 64*64*64])

        #first fully connected layer
        dense = tf.layers.dense(inputs = pool2_flat,
                            units = 512,
                            activation = tf.nn.relu,
                            name = 'fc')

        #Dropout layer for better learning
        dropout = tf.layers.dropout(dense,
                                    rate = self.keep_prob,
                                    training = self.is_training,
                                    name = 'dropout')

        #2nd fc layer
        self.loc_pred = tf.layers.dense(inputs = dropout,
                                        units = (self.num_classes),
                                        activation = tf.nn.relu,
                                        name = "fc_localizer")

        self.loc_pred = tf.reshape(self.loc_pred, shape = [4])
        return  self.loc_pred





#### LOSS ####
    @define_scope
    def loss_op(self):
        """
        Loss function calculated by adding losses for localizer and classifier
        """


        #loss_cl = tf.reshape(tf.cast(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_class, logits = self.class_pred),dtype=tf.float64),shape =[1])
        self.loss = tf.losses.absolute_difference(labels = self.target_loc, predictions = self.loc_pred)
        return self.loss



#### SUMMARY CREATOR ###
    @define_scope
    def summary_op(self):
        """
        Summary creator
        """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary = tf.summary.merge_all()
        return self.summary



##### OPTIMIZER #####
    @define_scope
    def optimize(self):
        """
        Optimizer of model
        """

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=tf.train.get_global_step())
        return self.opt



#### TRAIN ###
    def train_one_time(self, sess, writer, ckpt_dir, saver, step, epoch):
        """
        Train on whole training dataset
        """

        start_time = time.time()
        sess.run(self.train_init) #initialize proper dataset
        self.is_training = True #for dropout
        total_loss = 0
        num_batches = 0
        recent_loss = 0
        self.training = True
        try :
            while True:
                #train,get loss op, then gen summary, then optimize
                l, summary, _ = sess.run([self.loss_op, self.summary_op, self.optimize])
                writer.add_summary(summary, global_step=step)
                total_loss += l
                num_batches += 1
                recent_loss +=l
                step+=1
                if (step) % self.write_step == 0:
                    print('{0}: Avg loss for recent steps {1}'.format(self.global_step.eval(), recent_loss/self.write_step))
                    recent_loss = 0
        except tf.errors.OutOfRangeError:
            pass
        except KeyboardInterrupt:
            print("keyboard interrupt")
            pass
        saver.save(sess, ckpt_dir, global_step = self.global_step.eval()) #save checkpoint
        #Print some info
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/num_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step



 #### EVALUATE ####
    def evaluate(self, sess, writer, step, epoch):
        """
        Evaluate once from test dataset
        """

        start_time = time.time()
        sess.run(self.val_init) #initazlize proper dataset
        self.is_training = False #switch dropout off
        iou = 0
        total_loss = 0
        eval_step = 0
        eval_iou_t = tf.placeholder(dtype=tf.float32)
        eval_summary_op = tf.summary.scalar("val/iou", eval_iou_t)
        try:
            while True:
                #test, get loss then do summary
                l, new_pred, new_label = sess.run([self.loss_op, self.loc_pred, self.target_loc])
                eval_step+=1
                eval_iou = my_tools.box_iou(new_pred, new_label)

                eval_summary = sess.run(eval_summary_op, feed_dict={eval_iou_t:eval_iou})
                writer.add_summary(eval_summary, global_step = eval_step)



                iou += eval_iou
                total_loss +=l
        except tf.errors.OutOfRangeError:
            pass
        #Print some info
        print('Average loss at validaiton epoch {0}: {1}'.format(epoch, total_loss/eval_step))
        print('Average IoU at validation epoch {0}: {1}'.format(epoch, iou/eval_step))
        print('Took: {0} seconds'.format(time.time() - start_time))



###TEST
    def test_after_train(self, sess, writer):
        sess.run(self.test_init) #initialize proper datasets
        self.is_training = False
        test_iou = 0
        try:
            while True:
                new_pred, new_label, index = sess.run([self.loc_pred, self.target_loc, self.index])
                iou = my_tools.box_iou(new_pred, new_label)
                print("Test dataset IoU: {0}".format(iou))
                if (iou > 0.5):
                    image = my_tools.plot_result_on_img(index, new_pred)
                    image_tensor = tf.image.convert_image_dtype(image,dtype=tf.uint8)
                    image_tensor = tf.reshape(image_tensor,[1,256,256,4])
                    test_summary_op = tf.summary.image("IoU {0}: ".format(iou), image_tensor)
                    test_summary = sess.run(test_summary_op)
                    writer.add_summary(test_summary)
                test_iou+=iou
        except tf.errors.OutOfRangeError:
            pass
        print("Average Test IoU: {}".format(test_iou))


#### TO DO ####
    def train_n_times(self,result_dir, n_times):
        """
        training handling funciton
        """
        #Open session, required for tensorflow
        print("### Starting Session ###")
        with tf.Session() as sess:

            #Creating summary writer
            writer = tf.summary.FileWriter(self.ckpt_dir)

            #Add graph
            writer.add_graph(tf.get_default_graph())

            #Creating save for model session for future saving and restoring model
            saver = tf.train.Saver()

            #Check for checkpoint, if present load variables from it
            ckpt = tf.train.get_checkpoint_state(self.result_dir)
            if ckpt and ckpt.model_checkpoint_path:
                #if ckpt found load it and load global step
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("# Found checkpoint")
                step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])

            else:
                #If no checkpoint present, run clean session with clean initialization of variables
                init = tf.global_variables_initializer()
                sess.run(init)
                step = 0

            print("Current step: {0}".format(step))
            #Training
            print("### Starting Training ###")
            for epoch in range(1,n_times+1):
                try:
                    #Train one epoch
                    step = self.train_one_time(sess, writer, self.ckpt_dir, saver, step, epoch)
                    #Evaluate learning
                    self.evaluate(sess, writer, step,epoch)
                except KeyboardInterrupt:
                    print("keyboard interrupt")
                    pass
            print("### Testing netowrk ###")
            self.test_after_train(sess, writer)
            print("Closing session and saving results")
            print(self.global_step.eval(), step)
            #Save variables
            saver.save(sess, self.ckpt_dir, global_step = step)
            #Make sure that everything is recorded by writer

            writer.flush()
            #writer.add_graph(graph)
            writer.close()

        print("###Closed summary, work finnished###")
