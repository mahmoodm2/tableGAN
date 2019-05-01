"""
Paper: http://www.vldb.org/pvldb/vol11/p1071-park.pdf
Authors: Mahmoud Mohammadi, Noseong Park Adopted from https://github.com/carpedm20/DCGAN-tensorflow
Created : 07/20/2017
Modified: 10/15/2018
"""

from __future__ import division

import time
import tensorflow as tf
from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class TableGan(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None,
                 alpha=1.0, beta=1.0, delta_mean=0.0, delta_var=0.0
                 , label_col=-1, attrib_num=0
                 , is_shadow_gan=False
                 , test_id=''
                 ):
        """
        :param sess: TensorFlow session
        :param batch_size:  The size of batch. Should be specified before training.
        :param sample_num:
        :param output_height:
        :param output_width:
        :param y_dim: (optional) Dimension of dim for y. [None]
        :param z_dim: (optional) Dimension of dim for Z. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
        :param df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
        :param gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
        :param dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param dataset_name: Name of dataset, Required.
        """

        self.test_id = test_id

        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.feature_size = 0
        self.attrib_num = 1

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        # Classifier
        self.c_bn1 = batch_norm(name='c_bn1')
        self.c_bn2 = batch_norm(name='c_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.g_bn3 = batch_norm(name='g_bn3')

        self.alpha = alpha  # Info Loss Weigh
        self.beta = beta  # Class Loss Weigh

        self.delta_mean = delta_mean
        self.delta_var = delta_var

        self.label_col = label_col
        self.attrib_num = attrib_num

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        # mm if self.dataset_name in ["LACity", "Health", "Adult", "Ticket"]:

        self.data_X, self.data_y, self.data_y_normal = self.load_dataset(is_shadow_gan)
        self.c_dim = 1

        self.grayscale = (self.c_dim == 1)
        print("c_dim 1= " + str(self.c_dim))

        self.build_model()

    def build_model(self):

        self.y = tf.placeholder(
            tf.float32, [self.batch_size, self.y_dim], name='y')

        self.y_normal = tf.placeholder(
            tf.int16, [self.batch_size, 1], name='y_normal')

        # if self.crop:
        #     image_dims = [self.output_height, self.output_width, self.c_dim]
        # else:
        #     image_dims = [self.input_height, self.input_width, self.c_dim]

        data_dims = [self.input_height, self.input_width, self.c_dim]
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + data_dims, name='inputs')

        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + data_dims, name='sample_inputs')

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')

        self.z_sum = histogram_summary("z", self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits, self.D_features = self.discriminator(inputs, self.y, reuse=False)
            self.sampler = self.sampler(self.z, self.y)
            self.sampler_disc = self.sampler_discriminator(self.inputs, self.y)
            self.D_, self.D_logits_, self.D_features_ = self.discriminator(self.G, self.y, reuse=True)

            # Classifier
            if self.label_col > 0:  # We have duplicate attribute in input matrix and the label column should be masked
                inputs_C = masking(inputs, self.label_col, self.attrib_num)
            else:
                inputs_C = inputs

            self.C, self.C_logits, self.C_features = self.classification(inputs_C, self.y, reuse=False)

            if self.label_col > 0:  # We have duplicate attribute in input matrix and the label column should be masked
                self.GC = self.G
            else:
                self.GC = masking(self.G, self.label_col, self.attrib_num)

            self.C_, self.C_logits_, self.C_features = self.classification(self.GC, self.y, reuse=True)

        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits, self.D_features = self.discriminator(inputs)
            self.sampler = self.sampler(self.z)
            self.sampler_disc = self.sampler_discriminator(self.inputs)
            self.D_, self.D_logits_, self.D_features_ = self.discriminator(self.G, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)

        # Classifier
        if self.y_dim:
            self.c_sum = histogram_summary("c", self.C)
            self.c__sum = histogram_summary("c_", self.C_)
        #
        self.G_sum = image_summary("G", self.G)

        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        y_normal = tf.to_float(self.y_normal)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))

        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        # Classifier :Loss Funciton
        if self.y_dim:
            self.c_loss = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.C_logits, y_normal))
            self.g_loss_c = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.C_logits_, y_normal))

        # Original Loss Function
        self.g_loss_o = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        # Loss function for Information Loss
        self.D_features_mean = tf.reduce_mean(self.D_features, axis=0, keep_dims=True)
        self.D_features_mean_ = tf.reduce_mean(self.D_features_, axis=0, keep_dims=True)

        self.D_features_var = tf.reduce_mean(tf.square(self.D_features - self.D_features_mean), axis=0, keep_dims=True)

        self.D_features_var_ = tf.reduce_mean(tf.square(self.D_features_ - self.D_features_mean_), axis=0,
                                              keep_dims=True)

        dim = self.D_features_mean.get_shape()[-1]

        self.feature_size = dim

        print("Feature Size = %s" % (self.D_features_mean.get_shape()[-1]))

        # Previous Global Mean for real Data
        self.prev_gmean = tf.placeholder(tf.float32, [1, dim], name='prev_gmean')

        # Previous Global Mean  for fake Data
        self.prev_gmean_ = tf.placeholder(tf.float32, [1, dim], name='prev_gmean_')

        # Previous Global Variance for real Data
        self.prev_gvar = tf.placeholder(tf.float32, [1, dim], name='prev_gvar')

        # Previous Global Variance for fake Data
        self.prev_gvar_ = tf.placeholder(tf.float32, [1, dim], name='prev_gvar_')

        # Moving Average Contributions
        mac = 0.99

        self.gmean = mac * self.prev_gmean + (1 - mac) * self.D_features_mean

        self.gmean_ = mac * self.prev_gmean_ + (1 - mac) * self.D_features_mean_

        self.gvar = mac * self.prev_gvar + (1 - mac) * self.D_features_var

        self.gvar_ = mac * self.prev_gvar_ + (1 - mac) * self.D_features_var_

        self.info_loss = tf.add(tf.maximum(x=0.0, y=tf.reduce_sum(tf.abs(self.gmean - self.gmean_) - self.delta_mean))
                                , tf.maximum(x=0.0, y=tf.reduce_sum(tf.abs(self.gvar - self.gvar_) - self.delta_var)))

        ## Note from Bauke: not sure if this can go or what it was used for.
        # Prefix Origin
        # self.g_loss =  self.g_loss_o

        # OI Prefix in test_IDs
        self.g_loss = self.alpha * (self.g_loss_o) + self.beta * self.info_loss

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # Classifier: COI Prefix in test_IDs
        if self.y_dim:
            self.g_loss = self.alpha * (0.5 * self.g_loss_c + self.g_loss_o) + self.beta * self.info_loss
            self.c_loss_sum = scalar_summary("c_loss", self.c_loss)
            self.c_vars = [var for var in t_vars if 'c_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config, experiment):
        print("Start Training...\n")

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)

        # Classifier
        if self.y_dim:
            c_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                .minimize(self.c_loss, var_list=self.c_vars)

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.g_loss_sum])

        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_sum])

        # Classifier
        if self.y_dim:
            self.c_sum = merge_summary([self.z_sum, self.c_sum, self.c_loss_sum])

        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample = self.data_X[0:self.sample_num]

        if self.y_dim:
            sample_labels = self.data_y[0:self.sample_num]
            sample_labels_normal = self.data_y_normal[0:self.sample_num]

        if (self.grayscale):
            sample_inputs = np.array(sample).astype(
                np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        feature_size = self.feature_size

        gmean = np.zeros((1, feature_size), dtype=np.float32)
        gmean_ = np.zeros((1, feature_size), dtype=np.float32)
        gvar = np.zeros((1, feature_size), dtype=np.float32)
        gvar_ = np.zeros((1, feature_size), dtype=np.float32)

        for epoch in xrange(config.epoch):

            batch_idxs = min(len(self.data_X),
                             config.train_size) // config.batch_size  # train_size= np.inf

            seed = np.random.randint(100000000)
            np.random.seed(seed)
            np.random.shuffle(self.data_X)

            if self.y_dim:
                np.random.seed(seed)
                np.random.shuffle(self.data_y)

                np.random.seed(seed)
                np.random.shuffle(self.data_y_normal)

            for idx in xrange(0, batch_idxs - 1):

                batch = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]

                if self.y_dim:
                    batch_labels = self.data_y[
                                   idx * config.batch_size: (idx + 1) * config.batch_size]

                    batch_labels_normal = self.data_y_normal[
                                          idx * config.batch_size: (idx + 1) * config.batch_size]

                if self.grayscale:
                    batch_images = np.array(batch).astype(
                        np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                if self.y_dim:
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                       self.y_normal: batch_labels_normal
                                                   })
                    self.writer.add_summary(summary_str, counter)

                    # Classifier  Update C network
                    if self.y_dim:
                        _, summary_str = self.sess.run([c_optim, self.c_sum],
                                                       feed_dict={
                                                           self.inputs: batch_images,
                                                           self.z: batch_z,
                                                           self.y: batch_labels,
                                                           self.y_normal: batch_labels_normal
                                                       })
                        self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str, gmean, gmean_, gvar, gvar_ = \
                        self.sess.run([g_optim, self.g_sum, self.gmean, self.gmean_, self.gvar, self.gvar_],
                                      feed_dict={
                                          self.z: batch_z,
                                          self.y: batch_labels,
                                          self.inputs: batch_images,
                                          self.y_normal: batch_labels_normal,
                                          self.prev_gmean: gmean,
                                          self.prev_gmean_: gmean_,
                                          self.prev_gvar: gvar,
                                          self.prev_gvar_: gvar_
                                      })

                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str, gmean, gmean_, gvar, gvar_ = \
                        self.sess.run([g_optim, self.g_sum, self.gmean, self.gmean_, self.gvar, self.gvar_],
                                      feed_dict={self.z: batch_z,
                                                 self.y: batch_labels,
                                                 self.inputs: batch_images,
                                                 self.y_normal: batch_labels_normal,
                                                 self.prev_gmean: gmean,
                                                 self.prev_gmean_: gmean_,
                                                 self.prev_gvar: gvar,
                                                 self.prev_gvar_: gvar_
                                                 })
                    self.writer.add_summary(summary_str, counter)

                    # Classifier
                    errC = self.c_loss.eval({
                        self.inputs: batch_images,
                        self.z: batch_z,
                        self.y: batch_labels,
                        self.y_normal: batch_labels_normal
                    })

                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels,
                        self.y_normal: batch_labels_normal,
                        self.inputs: batch_images,
                        self.prev_gmean: gmean,
                        self.prev_gmean_: gmean_,
                        self.prev_gvar: gvar,
                        self.prev_gvar_: gvar_
                    })

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })

                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                        self.y: batch_labels
                    })
                else:
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                   })
                    self.writer.add_summary(summary_str, counter)

                    # Classifier  Update C network
                    if self.y_dim:
                        _, summary_str = self.sess.run([c_optim, self.c_sum],
                                                       feed_dict={
                                                           self.inputs: batch_images,
                                                           self.z: batch_z,
                                                           self.y: batch_labels,
                                                       })
                        self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str, gmean, gmean_, gvar, gvar_ = \
                        self.sess.run([g_optim, self.g_sum, self.gmean, self.gmean_, self.gvar, self.gvar_],
                                      feed_dict={
                                          self.z: batch_z,
                                          self.inputs: batch_images,
                                          self.prev_gmean: gmean,
                                          self.prev_gmean_: gmean_,
                                          self.prev_gvar: gvar,
                                          self.prev_gvar_: gvar_
                                      })

                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str, gmean, gmean_, gvar, gvar_ = \
                        self.sess.run([g_optim, self.g_sum, self.gmean, self.gmean_, self.gvar, self.gvar_],
                                      feed_dict={self.z: batch_z,
                                                 self.inputs: batch_images,
                                                 self.prev_gmean: gmean,
                                                 self.prev_gmean_: gmean_,
                                                 self.prev_gvar: gvar,
                                                 self.prev_gvar_: gvar_
                                                 })

                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.inputs: batch_images,
                        self.prev_gmean: gmean,
                        self.prev_gmean_: gmean_,
                        self.prev_gvar: gvar,
                        self.prev_gvar_: gvar_
                    })
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                    })
                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                    })

                counter += 1
                experiment.log_metric("d_loss", errD_fake + errD_real, step=idx)
                experiment.log_metric("g_loss", errG, step=idx)
                if self.y_dim:
                    experiment.log_metric("c_loss", errC, step=idx)
                    print("Dataset: [%s] -> [%s] -> Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, "
                          "c_loss: %.8f" % (config.dataset, config.test_id, epoch, idx, batch_idxs,
                                            time.time() - start_time, errD_fake + errD_real, errG, errC))
                else:
                    print("Dataset: [%s] -> [%s] -> Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, "
                          % (config.dataset, config.test_id, epoch, idx, batch_idxs,
                             time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:

                    # Classifier
                    if self.y_dim:
                        samples, d_loss, c_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.c_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                                self.y_normal: sample_labels_normal,
                                self.prev_gmean: gmean,
                                self.prev_gmean_: gmean_,
                                self.prev_gvar: gvar,
                                self.prev_gvar_: gvar_

                            }
                        )
                        print("[Sample] d_loss: %.8f, g_loss: %.8f, c_loss: %.8f" % (d_loss, g_loss, c_loss))

                    else:
                        # Without Classifier
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.prev_gmean: gmean,
                                self.prev_gmean_: gmean_,
                                self.prev_gvar: gvar,
                                self.prev_gvar_: gvar_
                            }
                        )

                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 1000) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            print(not self.y_dim)
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d_h3_conv')))

                h3_f = tf.reshape(h3, [self.batch_size, -1])
                # h4 = linear(tf.reshape(
                #     h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4 = linear(h3_f, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h4), h4, h3_f
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(
                    conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))

                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))

                h1 = tf.reshape(h1, [self.batch_size, -1])

                h1 = concat([h1, y], 1)

                # print( "D Shape h1: " + str(h1.get_shape())) 

                # h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'))) #new D remove

                # h2 = concat([h2, y], 1) #new D remove

                h3 = linear(h1, 1, 'd_h3_lin')

                print("D Shape h3: " + str(h3.get_shape()))

                # return tf.nn.sigmoid(h3), h3, h2
                return tf.nn.sigmoid(h3), h3, h1  # new D

    def sampler_discriminator(self, input, y=None):
        with tf.variable_scope("discriminator") as scope:

            scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(input, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d_h3_conv')))

                h3_f = tf.reshape(h3, [self.batch_size, -1])
                # h4 = linear(tf.reshape(
                #     h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4 = linear(h3_f, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h4)
            else:

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

                x = conv_cond_concat(input, yb)

                h0 = lrelu(
                    conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))

                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))

                h1 = tf.reshape(h1, [self.batch_size, -1])

                h1 = concat([h1, y], 1)

                h3 = linear(h1, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3)

    # Classifier
    def classification(self, image, y, reuse=False):

        with tf.variable_scope("classification") as scope:
            if reuse:
                scope.reuse_variables()
            assert (y is not None)

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(
                conv2d(x, self.c_dim + self.y_dim, name='c_h0_conv'))

            h0 = conv_cond_concat(h0, yb)

            # Classifier c_bn1()
            h1 = lrelu(self.c_bn1(
                conv2d(h0, self.df_dim + self.y_dim, name='c_h1_conv')))

            h1 = tf.reshape(h1, [self.batch_size, -1])  # h1 is 2-d
            h1 = concat([h1, y], 1)

            h3 = linear(h1, 1, 'c_h3_lin')

            return tf.nn.sigmoid(h3), h3, h1

    def generator(self, z, y=None):
        # Add
        with tf.variable_scope("generator") as scope:

            s_h, s_w = self.output_height, self.output_width

            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            # input_height >= 16
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            if self.y_dim:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

            # input_height >= 16 , gf_dim = 64
            # self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            # input_height < 16
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 4 * s_h8 * s_w8, 'g_h0_lin', with_w=True)

            print(" G Shape z : " + str(self.z_.get_shape()))

            # input_height >= 16
            # self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])

            # input_height < 16
            self.h0 = tf.reshape(self.z_, [-1, s_h8, s_w8, self.gf_dim * 4])

            h0 = tf.nn.relu(self.g_bn0(self.h0))
            if self.y_dim:
                h0 = conv_cond_concat(h0, yb)

            # input_height < 16
            h2, self.h2_w, self.h2_b = deconv2d(
                h0, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)

            h2 = tf.nn.relu(self.g_bn2(h2))
            if self.y_dim:
                h2 = conv_cond_concat(h2, yb)

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)

            h3 = tf.nn.relu(self.g_bn3(h3))
            if self.y_dim:
                h3 = conv_cond_concat(h3, yb)

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:

            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width

            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            # input_height >= 16
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            if self.y_dim:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

            # input_height < 16
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 4 * s_h8 * s_w8, 'g_h0_lin',
                                                   with_w=True)  # 4*64=256

            # input_height >= 16
            # self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])

            # input_height < 16
            self.h0 = tf.reshape(self.z_, [-1, s_h8, s_w8, self.gf_dim * 4])

            h0 = tf.nn.relu(self.g_bn0(self.h0))
            if self.y_dim:
                h0 = conv_cond_concat(h0, yb)

            # input_height >= 16
            # self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1',
            #                                         with_w=True) #2*2*256

            # h1 = tf.nn.relu(self.g_bn1(self.h1))
            # h1 = conv_cond_concat(h1, yb)

            # h2, self.h2_w, self.h2_b = deconv2d(
            #     h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True) # 4*4*128

            # input_height < 16
            h2, self.h2_w, self.h2_b = deconv2d(
                h0, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)  # 2*2*128

            h2 = tf.nn.relu(self.g_bn2(h2))
            if self.y_dim:
                h2 = conv_cond_concat(h2, yb)

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)  # 4*4*64 , 8*8*64

            h3 = tf.nn.relu(self.g_bn3(h3))
            if self.y_dim:
                h3 = conv_cond_concat(h3, yb)

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def load_dataset(self, load_fake_data=False):

        return self.load_tabular_data(self.dataset_name, self.input_height, self.y_dim, self.test_id, load_fake_data)

    def load_tabular_data(self, dataset_name, dim, classes=2, test_id='', load_fake_data=False):

        # self.train_data_path = f"./data/{dataset_name}/{dataset_name}"
        self.train_data_path = f'data/{dataset_name}/{dataset_name}'
        self.train_label_path = f'data/{dataset_name}/{dataset_name}_labels'

        if os.path.exists(self.train_data_path + ".csv"):

            X = pd.read_csv(self.train_data_path + ".csv", sep=';')
            print("Loading CSV input file : %s" % (self.train_data_path + ".csv"))

            self.attrib_num = X.shape[1]

            if self.y_dim:
                y = np.genfromtxt(open(self.train_label_path + ".csv", 'r'), delimiter=',')

                print("Loading CSV input file : %s" % (self.train_label_path + ".csv"))

                self.zero_one_ratio = 1.0 - (np.sum(y) / len(y))

        elif os.path.exists(self.train_data_path + ".pickle"):
            with open(self.train_data_path + '.pickle', 'rb') as handle:
                X = pickle.load(handle)

            with open(self.train_label_path + '.pickle', 'rb') as handle:
                y = pickle.load(handle)

            print("Loading pickle file ....")
        else:
            print("Error Loading Dataset !!")
            exit(1)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        # Normalizing Initial Data
        X = pd.DataFrame(min_max_scaler.fit_transform(X))
        # X is [rows * config.attrib_num] 15000 * 23

        padded_ar = padding_duplicating(X, dim * dim)

        X = reshape(padded_ar, dim)

        print("Final Real Data shape = " + str(X.shape))  # 15000 * 8 * 8

        if self.y_dim:
            y = y.reshape(y.shape[0], -1).astype(np.int16)
            y_onehot = np.zeros((len(y), classes), dtype=np.float)
            for i, lbl in enumerate(y):
                y_onehot[i, y[i]] = 1.0
            return X, y_onehot, y

        return X, None, None

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "tableGAN_model"
        if os.path.exists(f'{checkpoint_dir}/{self.model_dir}'):
            highest_num = 0
            for f in os.listdir(f'{checkpoint_dir}'):
                if f.startswith(f'{self.test_id}'):
                    file_idx = os.path.splitext(f)[0][-1]
                    try:
                        file_num = int(file_idx)
                        if file_num > highest_num:
                            highest_num = file_num
                    except ValueError:
                        print(f'The file name {f} is not an integer. Skipping')
            checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}_{highest_num + 1}'
            print(checkpoint_dir)
        else:
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print(" [Saving checkpoints in " + checkpoint_dir + " ...")
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints from " + checkpoint_dir + " ...")

        if os.path.exists(f'{checkpoint_dir}/{self.model_dir}'):
            highest_num = 0
            for f in os.listdir(f'{checkpoint_dir}'):
                print(f)
                if f.startswith(f'{self.model_dir}') and f.replace(self.model_dir, '') != '':
                    print(f)
                    file_name = os.path.splitext(f)[0][-1]
                    try:
                        file_num = int(file_name)
                        if file_num > highest_num:
                            highest_num = file_num
                    except ValueError:
                        print(f'The file name {file_name} is not an integer. Skipping')
            if highest_num == 0:
                checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}'
            else:
                checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}_{highest_num}'
        print(f'checkpoint dir: {checkpoint_dir}')
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))

            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            print(" [*] Success to read {}".format(ckpt_name))

            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
