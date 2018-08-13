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
        #self.permutation = np.random.permutation(range(len(os.listdir(config.PIC_SRC_DIR))))
        self.result_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.img_size = params["img_size"]
        self.num_channels = params["num_channels"]
        self.num_classes = params["num_classes"]
        self.lr = params["learning_rate"]
        self.keep_prob = params["keep_probability"]
        self.write_step = 10#params["print_nth_step"]
        self.is_training = True
        self.global_step = tf.Variable(0, False, dtype = tf.int32, name="global_step")
        #remove bellow
        #self.data = tf.placeholder(dtype= tf.float32, shape=[None,256,256,3])
        #self.target = tf.placeholder(dtype = tf.float32, shape=[None,5])
        #self.index = tf.placeholder(dtype= tf.int32,shape=[None,1])

        #FUNCTIONS
        self.data_pipeline
        self.prediction
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


        self.train_init = iterator.make_initializer(train_dataset)
        self.test_init = iterator.make_initializer(test_dataset)


        self.data, target, self.index = iterator.get_next()
        self.target = tf.reshape(target, shape = [5])
        self.data = tf.reshape(self.data,[-1, self.img_size, self.img_size, self.num_channels])




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
                             units = self.num_classes,
                             activation = tf.nn.relu,
                             name = "logits")

        return tf.nn.softmax(tf.reshape(dense2, shape=[5]))

#### LOSS ####
    @define_scope
    def loss_op(self):
        """
        loss
        """
        self.loss = tf.losses.mean_squared_error(labels = self.target, predictions = self.prediction)

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
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()



#### TRAIN ###
    def train_one_time(self, sess, ckpt_dir, writer, saver, step, epoch):
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
                l, _,summary = sess.run([self.loss_op, self.optimize, self.summary_op])
                writer.add_summary(summary, global_step=step)
                if (step) % self.write_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                total_loss += l
                num_batches += 1
                step+=1
                self.global_step = tf.convert_to_tensor(step)
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
    def evaluate(self, sess, writer, step, epoch):
        """
        Evaluate once from test
        """
        start_time = time.time()
        sess.run(self.test_init)
        self.is_training = False
        try:
            l, summary = sess.run([self.loss_op, self.summary_op])
            writer.add_summary(summary, global_step=step)
        except tf.errors.OutOfRangeError:
            pass
        print('Loss at validaiton epoch {0}: {1}'.format(epoch, l))
        print('Took: {0} seconds'.format(time.time() - start_time))



 ### TO DO ####
    def train_n_times(self,result_dir, n_times):


            print("Starting Session")
            #Assign name to session, assign it's default graph as graph
            with tf.Session() as sess:

                #Creating summary writer
                writer = tf.summary.FileWriter(self.ckpt_dir)

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


                print("Current step: {0}".format(step))
                #Add graph
                writer.add_graph(tf.get_default_graph())
                #Training
                print("Starting Training")
                for epoch in range(1,n_times+1):
                    try:
                        step = self.train_one_time(sess, self.ckpt_dir, writer, saver, step, epoch)
                        self.evaluate(sess,writer,step,epoch)
                    except KeyboardInterrupt:
                        print("keyboard interrupt")
                        pass
                print("Closing session and saving results")
                print(self.global_step.eval(), step)
                saver.save(sess, self.ckpt_dir, global_step = step)

            #Merge all summaries
            writer.flush()
            #writer.add_graph(graph)
            writer.close()
            print("Closed summary, work finnished")
