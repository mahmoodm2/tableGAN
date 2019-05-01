"""
Paper: http://www.vldb.org/pvldb/vol11/p1071-park.pdf
Authors: Mahmoud Mohammadi, Noseong Park Adopted from https://github.com/carpedm20/DCGAN-tensorflow
Created : 07/20/2017
Modified: 10/15/2018
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import matplotlib.pyplot as plt

from sklearn import preprocessing
import pickle
import pandas as pd

import gc

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])

DATASETS = ('LACity', 'Health', 'Adult', 'Ticket')


def padding_duplicating(data, row_size):
    arr_data = np.array(data.values.tolist())

    col_num = arr_data.shape[1]

    npad = ((0, 0), (0, row_size - col_num))

    # PAdding with zero
    arr_data = np.pad(arr_data, pad_width=npad, mode='constant', constant_values=0.)

    # Duplicating Values 
    for i in range(1, arr_data.shape[1] // col_num):
        arr_data[:, col_num * i: col_num * (i + 1)] = arr_data[:, 0: col_num]

    return arr_data


def reshape(data, dim):
    data = data.reshape(data.shape[0], dim, -1)

    return data


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)

    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def save_data(data, data_file):
    with open(data_file, 'wb') as handle:
        return pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(data_file):
    with open(data_file + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                   W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
    # import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def histogram(data_r, data_f, xlabel, ylabel, save_dir):
    if not os.path.exists(save_dir + '/histo'):
        os.makedirs(save_dir + '/histo')

    fig = plt.figure()
    plt.hist(data_r, bins='auto', label="Real Data")
    plt.hist(data_f, bins='auto', alpha=0.5, label=" Fake Data")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid()

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    plt.savefig(save_dir + "/histo/" + xlabel)

    plt.close(fig)

    plt.close()


def cdf(data_r, data_f, xlabel, ylabel, save_dir):
    if not os.path.exists(save_dir + '/cdf'):
        os.makedirs(save_dir + '/cdf')

    axis_font = {'fontname': 'Arial', 'size': '18'}

    # Cumulative Distribution
    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    fig = plt.figure()

    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

    plt.grid()
    plt.margins(0.02)

    plt.plot(x1, y, marker='o', linestyle='none', label='Real Data', ms=8)
    plt.plot(x2, y, marker='o', linestyle='none', label='Fake Data', alpha=0.5, ms=5)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    plt.savefig(save_dir + "/cdf/" + xlabel)

    plt.close(fig)

    gc.collect()


def nearest_value(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def rounding(fake, real, column_list):
    # max_row = min( fake.shape[0], real.shape[0])

    for i in column_list:
        print("Rounding column: " + str(i))
        fake[:, i] = np.array([nearest_value(real[:, i], x) for x in fake[:, i]])

    return fake


def compare(real, fake, save_dir, col_prefix, CDF=True, Hist=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        # Comparing Based on on mimumum number of columns and rows

    max_col = min(real.shape[1], fake.shape[1])
    max_row = min(fake.shape[0], real.shape[0])

    gap = np.zeros(max_col)

    for i in range(max_col):

        if Hist == True:
            histogram(real[: max_row, i], fake[: max_row, i], xlabel=col_prefix + ' : Column ' + str(i + 1), ylabel='',
                      save_dir=save_dir)

        if CDF == True:
            cdf(real[: max_row, i], fake[: max_row, i], xlabel=col_prefix + ' : Columns ' + str(i + 1),
                ylabel='Percentage', save_dir=save_dir)

        print(col_prefix + " : Cumulative Dist of Col " + str(i + 1))


def generate_data(sess, model, config, option):
    print("Start Generatig Data .... ")
    image_frame_dim = int(math.ceil(config.batch_size ** .5))

    if option == 1:

        input_size = len(model.data_X)

        dim = config.output_width  # 8

        merged_data = np.ndarray([config.batch_size * (input_size // config.batch_size), dim, dim],
                                 dtype=float)  # 64 * 234 * 16 * 16

        save_dir = './{}'.format(config.sample_dir + "/" + config.dataset)  # config.test_id)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

            # save_dir = save_dir + "/" + config.test_id

        # if not os.path.exists(save_dir):
        #       os.makedirs(save_dir)

        # samples_dir = save_dir + '/samples'
        #
        # if not os.path.exists(samples_dir):
        #       os.makedirs(samples_dir)

        for idx in xrange(input_size // config.batch_size):
            print(" [*] %d" % idx)
            z_sample = np.random.uniform(-1, 1, size=(config.batch_size, model.z_dim))

            zero_labeles = model.zero_one_ratio

            # if config.dataset == "LACity":
            #   zero_labeles= 0.48       # Based the ratio of labels in initial dataset
            #
            # elif config.dataset == "Health":
            #   zero_labeles= 0.91          # Based the ratio of labels in initial dataset
            #
            # elif config.dataset == "Adult":
            #   # Total =32561 ,  0s = 22980  = 70.6%
            #   zero_labeles= 0.706       # Based the ratio of labels in initial dataset
            #
            # elif config.dataset == "Ticket":
            #   # Total =80000
            #   zero_labeles= 0.575 # Based the ratio of labels in initial dataset

            y = np.ones((config.batch_size, 1))

            y[: int(zero_labeles * config.batch_size)] = 0
            np.random.shuffle(y)

            print("y shape " + str(y.shape))
            y = y.astype('int16')

            y_one_hot = np.zeros((config.batch_size, model.y_dim))

            # y indicates the index of ones in y_one_hot : in this case y_dim =2 so indexe are 0 or 1
            y_one_hot[np.arange(config.batch_size), y] = 1

            samples = sess.run(model.sampler, feed_dict={model.z: z_sample, model.y: y_one_hot, model.y_normal: y})

            # Merging Data for each batch size
            merged_data[idx * config.batch_size: (idx + 1) * config.batch_size] = samples.reshape(samples.shape[0],
                                                                                                  samples.shape[1],
                                                                                                  samples.shape[
                                                                                                      2])  # 234 * 64 * 16 *16

        # All generated data is ready in merged_data , now reshape it to a tabular marix

        fake_data = merged_data.reshape(merged_data.shape[0], merged_data.shape[1] * merged_data.shape[2])

        # Selecting the correct number of atributes (used in training)
        fake_data = fake_data[:, : model.attrib_num]

        print(" Fake Data shape= " + str(fake_data.shape))

        origin_data_path = model.train_data_path  # './data/'+ config.dataset+ '/train_'+ config.dataset + '_cleaned'

        if os.path.exists(origin_data_path + ".csv"):
            origin_data = pd.read_csv(origin_data_path + ".csv", sep=';')

        elif os.path.exists(origin_data_path + ".pickle"):
            with open(origin_data_path + '.pickle', 'rb') as handle:
                origin_data = pickle.load(handle)
        else:
            print("Error Loading Dataset !!")
            exit(1)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        min_max_scaler.fit(origin_data)

        # Fake Gen --> Scaling --> Rounding --> 1) Classification , 2)-->Normalizaing --> ( Euclidian Distance, CDF)
        # transforming data back to original scale
        scaled_fake = min_max_scaler.inverse_transform(fake_data)

        # Rounding Data
        round_columns = range(scaled_fake.shape[1])

        round_scaled_fake = rounding(scaled_fake, origin_data.as_matrix(), round_columns)

        # Required for Classification NN evaluation only
        # save_data(round_scaled_fake , save_dir +'/' + config.test_id + "_scaled_fake_tabular.pickle" )

        rsf_out = pd.DataFrame(round_scaled_fake)

        rsf_out.to_csv(f'{save_dir}/{config.dataset}_{config.test_id}_fake.csv' , index=False, sep=';')

        print("Generated Data shape = " + str(round_scaled_fake.shape))

    elif option == 5:  # Results for ShadowGAN (memberhsip attack).

        # input is data_x which is the fake data/test data/train data

        save_dir = './{}'.format(config.sample_dir + "/" + config.dataset)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

            # Applying Discriminator to Fake Data
        if config.shgan_input_type == 1:

            with open(
                    './samples/' + config.dataset + '/' + config.test_id + '/' + config.test_id + '_scaled_fake_tabular.pickle',
                    'rb') as handle:
                data_x = pickle.load(handle)

            output_file = os.path.join(save_dir, config.dataset + '_' + config.test_id + '_atk_fake_data.csv')

            discriminator_sampling(data_x, [], output_file, 'In', config, model, sess)

        elif config.shgan_input_type == 2:
            # Applying Test Data to Shadow GAN

            with open('./data/' + config.dataset + '/test_' + config.dataset + '_cleaned.pickle', 'rb') as handle:
                data_x = pickle.load(handle)

            with open('./data/' + config.dataset + '/test_' + config.dataset + '_labels.pickle', 'rb') as handle:
                data_y = pickle.load(handle)

            data_y = data_y.reshape(-1, 1)

            output_file = os.path.join(save_dir, config.dataset + '_' + config.test_id + '_atk_test_data.csv')

            discriminator_sampling(data_x, data_y, output_file, 'Out', config, model, sess)

        elif config.shgan_input_type == 3:
            # Applying Original Train Data to Shadow GAN

            with open('./data/' + config.dataset + '/train_' + config.dataset + '_cleaned.pickle', 'rb') as handle:
                data_x = pickle.load(handle)

            with open('./data/' + config.dataset + '/train_' + config.dataset + '_labels.pickle', 'rb') as handle:
                data_y = pickle.load(handle)

            data_y = data_y.reshape(-1, 1)

            output_file = os.path.join(save_dir, config.dataset + '_' + config.test_id + '_atk_train_data.csv')

            discriminator_sampling(data_x, data_y, output_file, '', config, model, sess)


def discriminator_sampling(input, lables, output_file, title, config, dcgan, sess):
    dim = config.output_width  # 8
    chunk = config.batch_size

    X = pd.DataFrame(input)

    padded_ar = padding_duplicating(X, dim * dim)

    X = reshape(padded_ar, dim)

    print("Final Real Data shape = " + str(input.shape))  # 15000 * 8 * 8

    # we need to generate lables from fake date to feed teh Discriminator Sampler
    input_size = len(input)
    print("input shape = " + str(input.shape))

    merged_data = np.ndarray([chunk * (input_size // chunk), 2], dtype=float)

    print(" Chunk Size = " + str(chunk))

    for idx in xrange(input_size // chunk):

        print(" [*] %d" % idx)
        # z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
        if len(lables) == 0:
            if config.dataset == "LACity":

                CLASSIFY_COL = 8  # ( 'Total Payments' = 8 starting from 0)
                CLASSIFY_VAL = 77636.37


            elif config.dataset == "Health":  # Correct
                CLASSIFY_COL = 31  # 'DIQ010' = 31 ( starting from 0)
                CLASSIFY_VAL = 1.0

            elif config.dataset == "Adult":  # Correct

                CLASSIFY_COL = 12  # 'hours-per-week' = 12 ( starting from 0)
                CLASSIFY_VAL = 40.43  # the mean value

            elif config.dataset == "Ticket":
                # Total =80000
                CLASSIFY_COL = 18  # 'MktFare' = 18( starting from 0)
                CLASSIFY_VAL = 204.49

            # Generating Labels
            y = []

            c = 0
            # print((idx+1) * chunk)
            # print((idx) * chunk)
            # print(len( data_x[idx * chunk : (idx+1) * chunk] ) )
            for rec in input[idx * chunk: (idx + 1) * chunk]:
                # print(c)
                c += 1
                if rec[CLASSIFY_COL] > CLASSIFY_VAL:
                    y.append(1)
                else:
                    y.append(0)

            y = np.asarray(y)

        else:
            y = lables[idx * chunk: (idx + 1) * chunk]

        y = y.reshape(-1, 1)

        y = y.astype('int16')
        y_one_hot = np.zeros((chunk, dcgan.y_dim))

        sample_input = X[idx * chunk: (idx + 1) * chunk]

        sample_input = sample_input.reshape(chunk, dim, dim, 1)

        # y indicates the index of ones in y_one_hot : in this case y_dim =2 so indexe are 0 or 1
        y_one_hot[np.arange(chunk), y] = 1

        samples = sess.run(dcgan.sampler_disc,
                           feed_dict={dcgan.inputs: sample_input, dcgan.y: y_one_hot, dcgan.y_normal: y})
        # Samples are Probability of input data (result of Sigmoid Activation in Discriminator)

        # Merging Data for each batch size
        merged_data[idx * chunk: (idx + 1) * chunk, 0] = samples[:, 0]
        merged_data[idx * chunk: (idx + 1) * chunk, 1] = y[:, 0]

    # End For

    print("hstack output  shape = " + str(merged_data.shape))

    f = open(output_file, "w+")

    f.write("Prob, Label , In/Out \n")

    for rec in merged_data:
        f.write("%.3f, %d, %s \n" % (rec[0], rec[1], title))
