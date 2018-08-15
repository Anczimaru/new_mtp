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


            self.writer = tf.contrib.summary.create_file_writer(self.ckpt_dir)
            with self.writer.as_default():
                tf.contrib.summary.always_record_summaries()


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
        Summary creator, uses tf.contrib.summary for summary creation
        """

        with tf.name_scope("summary"):
            with self.writer.as_default():
                with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                    tf.contrib.summary.scalar('loss', self.loss)
                self.summary = tf.contrib.summary.all_summary_ops()
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
    def train_one_time(self, sess, ckpt_dir, saver, step, epoch):
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
                l, _, _ = sess.run([self.loss_op, self.summary_op, self.optimize])
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
        saver.save(sess, ckpt_dir, global_step = step) #save checkpoint
        #Print some info
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/num_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step


 #### EVALUATE ####
    def evaluate(self, sess, step, epoch):
        """
        Evaluate once from test dataset
        """

        start_time = time.time()
        sess.run(self.val_init) #initazlize proper dataset
        self.is_training = False #switch dropout off
        total_loss = 0
        iou = 0
        eval_step = 0
        try:
            #test, get loss then do summary
            l, _, new_pred, new_label = sess.run([self.loss_op, self.summary_op, self.loc_pred, self.target_loc])
            iou += my_tools.box_iou(new_pred, new_label)
            total_loss +=l
            eval_step+=1
        except tf.errors.OutOfRangeError:
            pass
        #Print some info
        print('Average loss at validaiton epoch {0}: {1}'.format(epoch, total_loss/eval_step))
        print('Average IoU at validation epoch {0}: {1}'.format(epoch, iou/eval_step))
        print('Took: {0} seconds'.format(time.time() - start_time))



###TEST
    def test_after_train(self,sess):
        sess.run(self.test_init) #initialize proper datasets
        self.is_training = False
        iou = 0
        try:
            new_pred, new_label = sess.run([self.loc_pred, self.target_loc])
            iou = my_tools.box_iou(new_pred, new_label)
            print("Test dataset IoU: {0}".format(iou))
        except tf.errors.OutOfRangeError:
            pass



#### TO DO ####
    def train_n_times(self,result_dir, n_times):
        """
        Main training handling funciton
        """
        #Open session, required for tensorflow
        print("Starting Session")
        with tf.Session() as sess:
            with self.writer.as_default():

                #Creating save for model session for future saving and restoring model
                saver = tf.train.Saver()

                #Check for checkpoint, if present load variables from it
                ckpt = tf.train.get_checkpoint_state(self.result_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    #if ckpt found load it and load global step
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("Found checkpoint")
                    step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])

                else:
                    #If no checkpoint present, run clean session with clean initialization of variables
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    step = 0

                #initialize writer for data summary
                tf.contrib.summary.initialize(graph=tf.get_default_graph())

                print("Current step: {0}".format(step))
                #Training
                print("Starting Training")
                for epoch in range(1,n_times+1):
                    try:
                        #Train one epoch
                        step = self.train_one_time(sess, self.ckpt_dir, saver, step, epoch)
                        #Evaluate learning
                        self.evaluate(sess,step,epoch)
                    except KeyboardInterrupt:
                        print("keyboard interrupt")
                        pass
                self.test_after_train(sess)
                print("Closing session and saving results")
                print(self.global_step.eval(), step)
                #Save variables
                saver.save(sess, self.ckpt_dir, global_step = step)
                #Make sure that everything is recorded by writer
                self.writer.flush()

        print("Closed summary, work finnished")
