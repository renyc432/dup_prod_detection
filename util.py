import os
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import math

import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

lemma = WordNetLemmatizer()

def text_preprocessing(text):
    text = re.sub('[^a-zA-Z0-9 @ . , : - _]', '', str(text))
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = text.lower().split()
    tokens_stop = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemma.lemmatize(word) for word in tokens_stop]
    text_processed = ' '.join(tokens)
    return text_processed

def get_all_img_path(directory):
    img_path = []
    for dirname,_,filenames in os.walk(directory):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            img_path.append(fullpath)
    return img_path


def load_img_RGB(img_dir, img_filenames, width, height):
    '''
    Read an image into python and resize to the ideal shape
    
    img_dir : directory of the image
    img_filenames : name of the image
    width : width of image
    height : height of image

    Returns
    -------
    image array

    '''
    img_dataset = []
    for name in img_filenames:
        path = os.path.join(img_dir,name)
        img = image.load_img(path,target_size=(width, height))
        img = image.img_to_array(img)

        img_dataset.append(img)
    return np.array(img_dataset)


def load_img_BGR(img_dir, img_filenames, width, height):
    img_dataset = []
    for name in img_filenames:
        path = os.path.join(img_dir,name)
        image = cv2.imread(path)
        image = cv2.resize(image,(width, height))
        image = np.array(image).astype('float32')
        img_dataset.append(image)
    return np.array(img_dataset)


def split(X, y, test_size):
    '''
    Split X,y to train and validation
    
    Returns
    -------
    (X_train,y_train),(X_val,y_val)

    '''
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_ind, val_ind in split.split(X,y):
        X_val,y_val = X[val_ind], y[val_ind]
        X_train,y_train = X[train_ind], y[train_ind]
    
    return X_train,y_train,X_val,y_val



def prepare_labels(y):
    '''
    one hot encoding
    '''
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder


# Arcmarginproduct
class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# Function to decode images
def decode_image(image_data, img_size):
    image = tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# evaluation metric for models
def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1