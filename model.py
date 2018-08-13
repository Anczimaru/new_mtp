from my_tools import define_scope
import tensorflow as tf
import time
import os
import config
from dataset_helper import parser as parser


class Model(object):
    """
    Model of NN
    """
    #Variable of model
    #accuracy = tf.reduce


### INITIALIZATION ####
    def __init__(self, params, log_dir,ckpt_dir):

        self.result_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.img_size = params["img_size"]
        self.num_channels = params["num_channels"]
        self.num_classes = params["num_classes"]
        self.lr = params["learning_rate"]
        self.keep_prob = params["keep_probability"]
        self.write_step = 1#params["print_nth_step"]
        self.is_training = True
        self.global_step = tf.Variable(0, False, dtype = tf.int64, name="global_step")
        self.writer = tf.contrib.summary.create_file_writer(self.ckpt_dir)

        #FUNCTIONS
        self.data_pipeline
        self.prediction
        self.classifier
        self.localizer
        self.loss_op
        self.optimize
        self.summary




#FUNCTION DEFINITIONS
    @define_scope
    def data_pipeline(self):
        """
        loading of TFRecords
        """
        train_dataset = tf.data.TFRecordDataset(os.path.join(config.DATA_DIR,config.TFRECORD_NAMES[0]))
        test_dataset = tf.data.TFRecordDataset(os.path.join(config.DATA_DIR,config.TFRECORD_NAMES[1]))
        train_dataset =train_dataset.map(parser)
        test_dataset = test_dataset.map(parser)


        iterator = tf.data.Iterator.from_structure( train_dataset.output_types,
                                                    train_dataset.output_shapes)




        self.data, target_class, target_loc, self.index, _ = iterator.get_next()
        self.target_class = tf.cast(tf.reshape(target_class, shape= [1]),dtype=tf.float32)
        self.target_loc = tf.reshape(target_loc, shape = [2])
        self.data = tf.reshape(self.data,[-1, self.img_size, self.img_size, self.num_channels])

        self.train_init = iterator.make_initializer(train_dataset)
        self.test_init = iterator.make_initializer(test_dataset)



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


        pool2_flat = tf.reshape(pool2,[-1, 64*64*64])


        dense = tf.layers.dense(inputs = pool2_flat,
                            units = 512,
                            activation = tf.nn.relu,
                            name = 'fc')

        dropout = tf.layers.dropout(dense,
                                    rate = self.keep_prob,
                                    training = self.is_training,
                                    name = 'dropout')

        dense2 = tf.layers.dense(inputs = dropout,
                                    units = 128,
                                    activation = tf.nn.relu,
                                    name = "fc_2")
        return dense2

    @define_scope
    def classifier(self):
        self.class_pred = tf.layers.dense(inputs = self.prediction,
                             units = 1,
                             activation = tf.nn.relu,
                             name = "fc_classifier")

        return self.class_pred

    @define_scope
    def localizer(self):
        self.loc_pred = tf.layers.dense(inputs = self.prediction,
                                        units = (self.num_classes-1),
                                        activation = tf.nn.relu,
                                        name = "fc_localizer")
        self.loc_pred = tf.reshape(self.loc_pred, shape = [2])
        return self.loc_pred



#### LOSS ####
    @define_scope
    def loss_op(self):
        """
        loss
        """
        loss_loc =  tf.reshape(tf.cast(tf.losses.mean_squared_error(labels = self.target_loc, predictions = self.loc_pred),dtype=tf.float64),shape =[1])
        loss_cl = tf.reshape(tf.cast(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_class, logits = self.class_pred),dtype=tf.float64),shape =[1])
        self.loss = tf.add(loss_cl,loss_loc)
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', self.loss)
        return self.loss

##### OPTIMIZER #####
    @define_scope
    def optimize(self):
        """
        Optimizer of model
        """
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)
        a = tf.cast(1,tf.int32) #trash line so fetcher doesnt return errors
        return a




###summary####
    @define_scope
    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''

        with tf.name_scope('summaries'):
            tf.contrib.summary.scalar('loss', self.loss)
            pass



#### TRAIN ###
    def train_one_time(self, sess, ckpt_dir, saver, step, epoch):
        """
        Training op for model
        """
        start_time = time.time()
        sess.run(self.train_init)
        self.is_training = True
        total_loss = 0
        num_batches = 0
        self.training = True
        try :
            while True:
                #train
                l, _,summary = sess.run([self.loss_op, self.optimize, tf.contrib.summary.all_summary_ops()])
                #writer.add_summary(self.summary, global_step=step)
                total_loss += l
                num_batches += 1
                step+=1
                self.global_step = tf.convert_to_tensor(step)
                if (step) % self.write_step == 0:
                    print('Loss at step {0}: {1}'.format(self.global_step.eval(), l))
        except tf.errors.OutOfRangeError:
            pass
        except KeyboardInterrupt:
            print("keyboard interrupt")
            pass
        saver.save(sess, ckpt_dir, global_step = step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/num_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step


 #### EVALUATE ####
    def evaluate(self, sess, step, epoch):
        """
        Evaluate once from test
        """
        start_time = time.time()
        sess.run(self.test_init)
        self.is_training = False
        total_loss = 0
        try:
            l, summary = sess.run([self.loss_op, tf.contrib.summary.all_summary_ops()])
            #writer.add_summary(self.summary, global_step=step)
            total_loss +=l
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss at validaiton epoch {0}: {1}'.format(epoch, total_loss))
        print('Took: {0} seconds'.format(time.time() - start_time))



 ### TO DO ####
    def train_n_times(self,result_dir, n_times):


            print("Starting Session")
            #Assign name to session, assign it's default graph as graph
            with tf.Session() as sess:


                #Creating save for model session for future saving and restoring model
                saver = tf.train.Saver()

                #Loading last checkpoint
                ckpt = tf.train.get_checkpoint_state(self.result_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    #if ckpt found load it and load global step
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("Found checkpoint")
                    step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])

                else:
                    #Assign Initializer
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    step = 0

                self.global_step = tf.cast(step, dtype = tf.int64)
                with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.initialize(graph=tf.get_default_graph())


                    #Creating summary writer



                    print("Current step: {0}".format(step))
                    #Training
                    print("Starting Training")
                    for epoch in range(1,n_times+1):
                        try:
                            step = self.train_one_time(sess, self.ckpt_dir, saver, step, epoch)
                            self.evaluate(sess,step,epoch)
                        except KeyboardInterrupt:
                            print("keyboard interrupt")
                            pass
                    print("Closing session and saving results")
                    print(self.global_step.eval(), step)
                    saver.save(sess, self.ckpt_dir, global_step = step)

            self.writer.flush()
            print("Closed summary, work finnished")
