#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 16:44:45 2017

@author: lily
"""
import tensorflow as tf
import cv2
import os
import sys
import numpy as np
from PIL import Image   
import csv
reload(sys)
sys.setdefaultencoding('utf8')

setroot = 'ADL-Rundle-6'
root = './train_tfrecord/'  #root of TFRecords
No = 1

#读取tfrecords文件
def decode_from_tfrecords(filename_queue):
    
    #filename_queue = tf.train.string_input_producer([filename],num_epochs=1)
    reader=tf.TFRecordReader()
    _,serialized=reader.read(filename_queue)  
    features=tf.parse_single_example(serialized,features={ 'frame': tf.FixedLenFeature([], tf.int64),
                                                       'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string)})
    frame = tf.cast(features["frame"], tf.int64)
    #label = tf.cast(features["label"], tf.int64)
    img = tf.decode_raw(features["img_raw"], tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    
    label=features['label']
    return img,label,frame

def get_neg_couple(fileroot):
    filenames = os.listdir(fileroot)  
    Filequeue = [] 
    for f in filenames:
        fq = fileroot + f
        Filequeue.append(fq)
    neg_list = []  
    lab_list = []
    f_queue1 = tf.train.string_input_producer(Filequeue,num_epochs=1)   #put the file in to a queue
    Img1,Lab1,Frm1 = decode_from_tfrecords(f_queue1)
    batch_img1,batch_lab1, batch_frm1 = tf.train.shuffle_batch([Img1,Lab1,Frm1], batch_size=1, 
                                        capacity=10, min_after_dequeue=6, num_threads=1)   
    
    f_queue2 = tf.train.string_input_producer(Filequeue,num_epochs=1)   #put the file in to a queue
    Img2,Lab2,Frm2 = decode_from_tfrecords(f_queue2)
    batch_img2,batch_lab2,batch_frm2 = tf.train.shuffle_batch([Img2,Lab2,Frm2], batch_size=1, 
                                        capacity=10, min_after_dequeue=6, num_threads=1) 
    init = tf.global_variables_initializer()  
    local_init = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(local_init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        try:
            while not coord.should_stop():
                img1,lab1,frm1 = sess.run([batch_img1,batch_lab1,batch_frm1])
                img2,lab2,frm2 = sess.run([batch_img2,batch_lab2,batch_frm2])
                if lab1!=lab2:
                    couple = np.concatenate((img1,img2), axis = 3)
                    if len(neg_list)==0:
                        neg_list=couple
                    else:
                        neg_list = np.concatenate((neg_list,couple), axis = 0)
                    if ([lab1,lab2]or[lab2,lab1]) not in lab_list:
                        lab_list.append([lab1,lab2])

        except tf.errors.OutOfRangeError:
            print '***************Negative couple read done*****************'
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(threads)
    	coord.join()
    return neg_list,lab_list
    
def get_pos_couple(fileroot):
    filenames = os.listdir(fileroot)  
    Filequeue = [] 
    for f in filenames:
        fq = fileroot + f
        Filequeue.append(fq)
    pos_list = []  
    lab_list = []
    f_queue = tf.train.string_input_producer(Filequeue,num_epochs=1)   #put the file in to a queue
    Img,Lab,Frm = decode_from_tfrecords(f_queue)
    batch_img,batch_lab, batch_frm = tf.train.shuffle_batch([Img,Lab,Frm], batch_size=2, 
                                        capacity=10, min_after_dequeue=6, num_threads=1)   
    
    init = tf.global_variables_initializer()  
    local_init = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(local_init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        try:
            while not coord.should_stop():
                img,lab,frm = sess.run([batch_img,batch_lab,batch_frm])
                if lab[0]==lab[1]:
                    couple = img.reshape(1,img.shape[1],img.shape[2],6)
                    if len(pos_list)==0:
                        pos_list=couple
                    else:
                        pos_list =  np.concatenate((pos_list,couple), axis = 0)
                    if ([lab[0],lab[1]]or [lab[1],lab[0]])not in lab_list:
                        lab_list.append([lab[0],lab[1]])

        except tf.errors.OutOfRangeError:
            print '***************Positive couple read done*****************'
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(threads)
    	coord.join()
    return pos_list,lab_list
def train_data(fileroot):    
    neg_couple_list,neg_lab = get_neg_couple(fileroot) #read examples
    pos_couple_list,pos_lab = get_pos_couple(fileroot) 
    Xneg = np.random.permutation(neg_couple_list)  #shaffle the ordered examples
    Yneg = [0]*len(Xneg)
    Xpos = np.random.permutation(pos_couple_list)  
    Ypos = [1]*len(Xpos)
    
    X = np.vstack((Xneg,Xpos))  #congeal pos and neg examples as a whole list
    y = Yneg + Ypos
    Y = tf.one_hot(y,2)
    Lab = tf.cast(Y, tf.float32)
    with tf.Session() as sess:
        L = sess.run(Lab)
    return X,L

