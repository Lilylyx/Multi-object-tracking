#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 12:20:31 2017

@author: lily
"""

import scipy.io as sio
import numpy as np
import cv2
import tensorflow as tf



matfile = '/home/lily/Documents/MOT_matlab/cache/ETH-Bahnhof_graph_res2.mat'
imgpath = '/home/lily/Documents/MOT_matlab/data/ETH-Bahnhof/'


def readmat(matfile):
    mdata=sio.loadmat(matfile)
    struct = mdata['dres']
    dres = struct[0,0]
    x = dres['x']
    y = dres['y']
    w = dres['w']
    h = dres['h']
    r = dres['r']
    fr = dres['fr']
    nei = dres['nei']
    c = dres['nei']
    return x, y, w, h, r, fr, nei, c


def crop_image(img_path, x, y, w, h):
    
    image = cv2.imread(img_path)
    crop_img = image[y:y+h,x:x+w ,:]
    crop_image = cv2.resize(crop_img, (32,32)) 
    re_img = crop_image.reshape(1,32,32,3)
    return re_img

 
def calc_simila(img1, img2):
    couple = np.concatenate((img1,img2), axis = 3)
    with tf.Session() as session:
        saver = tf.train.import_meta_graph('save/model.meta')
        saver.restore(session, 'save/model')
        graph = tf.get_default_graph()
        teX= graph.get_tensor_by_name('teX:0')
        test_inf = tf.get_collection("test_inf")[0]
        #test_inf = graph.get_tensor_by_name("test_inf:0")
        simila=session.run(test_inf, feed_dict={teX: couple})  
        print '~~~~~similarity score %s ~~~~~' % simila[0][1]

#    
#    saver = tf.train.Saver()
#    
#    with tf.Session() as session:
#        ckpt = tf.train.get_checkpoint_state("./save") 
#        if ckpt and ckpt.model_checkpoint_path: 
#            saver.restore(session, "save/model.ckpt")  
#            print "Model restored"
#            graph = tf.get_default_graph()
#            teX= graph.get_tensor_by_name('teX')
#            test_inf = graph.get_tensor_by_name("test_inf")
#            simila=session.run(test_inf, feed_dict={teX: couple})  
#            print '~~~~~similarity score %s ~~~~~' % simila
#        else:
#            print 'ERROR'
    return simila[0][1]
        
    

def give_score(x, y, w, h, fr, nei, c):

    for i in range(len(nei)):
        
        if len(nei[i]['inds'][0][0]) != 0:
            print 'i=', i
            Impath1 = imgpath + 'image_' + str('%08d' % fr[i][0]) + '_0.png' 
            x1 = int(x[i][0])
            y1 = int(y[i][0])
            w1 = int(w[i][0])
            h1 = int(h[i][0])
            imgcut_1 = crop_image(Impath1, x1, y1, w1, h1)
            
            for j in range(len(nei[i]['inds'][0][0])):
                tmp = nei[i]['inds'][0][0][j]-1    
                Impath_tmp = imgpath + 'image_' + str('%08d' % fr[tmp][0]) + '_0.png'
                xtmp = int(x[tmp][0])
                ytmp = int(y[tmp][0])
                wtmp = int(w[tmp][0])
                htmp = int(h[tmp][0])
                imgcut_tmp = crop_image(Impath_tmp, xtmp, ytmp, wtmp, htmp)
                scr = calc_simila(imgcut_1, imgcut_tmp)
                
    #==============================================================================
    #           function to change the images' form
    #         using tensorflow to give the score : scr    
                
    #==============================================================================
                c[i]['inds'][0][0][j] = scr
    
        else: 
            continue
    return c

x, y, w, h, r, fr, nei, c = readmat(matfile)
c = give_score(x, y, w, h, fr, nei, c)
