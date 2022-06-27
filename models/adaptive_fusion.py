from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Merge, Permute, Flatten, Dropout, TimeDistributedDense, Reshape, Layer, \
    ActivityRegularization, RepeatVector, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.callbacks import History
from keras.layers import Input, Dense, Embedding, merge, Dropout, BatchNormalization
from keras.optimizers import SGD,Adagrad,Adam,RMSprop 
from keras.utils import np_utils
from keras.layers import ChainCRF
from keras import backend as K
import theano.tensor as T 
import cPickle
import h5py
import numpy as np

from load_data_multimodal import load_data
from ner_evaluate import evaluate,evaluate_each_class


def Adaptive_fusion():
    # word level word representation
    w_tweet = Input(shape=(sent_maxlen,), dtype='int32')
    w_emb = Embedding(input_dim=word_vocab_size, output_dim=w_emb_dim,weights=[word_matrix], input_length=sent_maxlen, mask_zero=False)(
        w_tweet)
    w_feature = Bidirectional(LSTM(w_emb_dim, return_sequences=True, input_shape=(sent_maxlen, w_emb_dim)))(w_emb)

    # char level word representation
    c_tweet = Input(shape=(sent_maxlen*word_maxlen,), dtype='int32')
    c_emb = Embedding(input_dim=char_vocab_size, output_dim=c_emb_dim, input_length=sent_maxlen*word_maxlen, mask_zero=False)(
        c_tweet)
    c_reshape = Reshape((sent_maxlen, word_maxlen, c_emb_dim))(c_emb)
    c_conv1 = TimeDistributed(Convolution1D(nb_filter = 32, filter_length=2, border_mode='same', activation='relu'))(c_reshape)
    c_pool1 = TimeDistributed(MaxPooling1D(pool_length=2))(c_conv1)
    c_dropout1 = TimeDistributed(Dropout(0.25))(c_pool1)
    c_conv2 = TimeDistributed(Convolution1D(nb_filter =32, filter_length=3, border_mode ='same', activation = 'relu'))(c_dropout1)
    c_pool2 = TimeDistributed(MaxPooling1D(pool_length = 2))(c_conv2)
    c_dropout2 = TimeDistributed(Dropout(0.25))(c_pool2)
    c_conv3 = TimeDistributed(Convolution1D(nb_filter = 32, filter_length=4, border_mode='same', activation='relu'))(c_dropout2)
    c_pool3 = TimeDistributed(MaxPooling1D(pool_length=2))(c_conv3)
    c_dropout3 = TimeDistributed(Dropout(0.25))(c_pool3)
    c_batchNorm = BatchNormalization()(c_dropout3)
    c_flatten = TimeDistributed(Flatten())(c_batchNorm)
    c_fullConnect = TimeDistributed(Dense(100))(c_flatten)
    c_activate = TimeDistributed(Activation('relu'))(c_fullConnect)
    c_emb2 = TimeDistributed(Dropout(0.25))(c_activate)
    c_feature = TimeDistributed(Dense(w_emb_dim_char_level))(c_emb2)

    # merge the feature of word level and char level
    merge_w_c_emb = merge([w_feature,c_feature], mode = 'concat', concat_axis = 2)
    w_c_feature = Bidirectional(LSTM(output_dim=final_w_emb_dim, return_sequences = True))(merge_w_c_emb) 
    
    # reshape the image representation
    img = Input(shape=(1,feat_dim, w, w))
    img_reshape = Reshape((feat_dim, w * w))(img)
    img_permute = Permute((2, 1))(img_reshape)

    # word-guided visual attention 
    img_permute_reshape = TimeDistributed(RepeatVector(sent_maxlen))(img_permute) 
    img_permute_reshape = Permute((2, 1, 3))(img_permute_reshape) 
    w_repeat = TimeDistributed(RepeatVector(w*w))(w_c_feature) 
    w_repeat = TimeDistributed(TimeDistributed(Dense(final_w_emb_dim)))(w_repeat)
    img_permute_reshape = TimeDistributed(TimeDistributed(Dense(final_w_emb_dim)))(img_permute_reshape)
    img_w_merge = merge([img_permute_reshape, w_repeat], mode='concat') 

    att_w = TimeDistributed(Activation('tanh'))(img_w_merge)
    att_w = TimeDistributed(TimeDistributed(Dense(1)))(att_w) 
    att_w = TimeDistributed(Flatten())(att_w) 
    att_w_probability = Activation('softmax')(att_w) 

    img_permute_r = TimeDistributed(Dense(final_w_emb_dim))(img_permute)
    img_new = merge([att_w_probability, img_permute_r], mode='dot', dot_axes=(2,1)) 


    # image-guided textual attention
    img_new_dense = TimeDistributed(Dense(final_w_emb_dim))(img_new)  
    img_new_rep = TimeDistributed(RepeatVector(sent_maxlen))(img_new_dense) 

    tweet_dense = TimeDistributed(Dense(final_w_emb_dim))(w_c_feature) 
    tweet_dense1 = Flatten()(tweet_dense)
    tweet_rep = RepeatVector(sent_maxlen)(tweet_dense1) 
    tweet_rep = Reshape((sent_maxlen, sent_maxlen, final_w_emb_dim))(tweet_rep)

    att_img = merge([img_new_rep, tweet_rep], mode='concat') 
    att_img = TimeDistributed(Activation('tanh')) (att_img) 
    att_img = TimeDistributed(TimeDistributed(Dense(1)))(att_img) 
    att_img = TimeDistributed(Flatten())(att_img) 
    att_img_probability = Activation('softmax')(att_img)

    tweet_new = merge([att_img_probability, tweet_dense], mode='dot', dot_axes=(2, 1)) 

    img_new_resize = TimeDistributed(Dense(final_w_emb_dim, activation='tanh'))(img_new) 
    tweet_new_resize = TimeDistributed(Dense(final_w_emb_dim, activation='tanh'))(tweet_new) 


    # gate -> dependecy new
    merge_img_w = merge([img_new_resize, tweet_new_resize], mode='sum')
    gate_img = TimeDistributed(Dense(1, activation='sigmoid'))(merge_img_w)
    gate_img = TimeDistributed(RepeatVector(final_w_emb_dim))(gate_img)  
    gate_img = TimeDistributed(Flatten())(gate_img) 
    part_new_img = merge([gate_img, img_new_resize], mode='mul') 


    #gate -> semantic new
    gate_tweet = Lambda(lambda_rev_gate, output_shape=(sent_maxlen, final_w_emb_dim))(gate_img)
    part_new_tweet = merge([gate_tweet, tweet_new_resize], mode='mul')
    
    part_img_w = merge([part_new_img, part_new_tweet], mode='concat')
    part_img_w = TimeDistributed(Dense(final_w_emb_dim))(part_img_w)


    #gate -> fusion gate feature
    gate_merg = TimeDistributed(Dense(1, activation='sigmoid'))(part_img_w)
    gate_merg = TimeDistributed(RepeatVector(final_w_emb_dim))(gate_merg)  
    gate_merg = TimeDistributed(Flatten())(gate_merg) 
    part_sample = merge([gate_merg, part_img_w], mode='mul')

    w_c_emb = TimeDistributed(Dense(final_w_emb_dim))(w_c_feature) 

    merge_multimodal_w = merge([part_sample, w_c_emb], mode='concat') 
    multimodal_w_feature = TimeDistributed(Dense(num_classes))(merge_multimodal_w)

