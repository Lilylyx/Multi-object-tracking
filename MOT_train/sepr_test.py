# -*- coding: utf-8 -*-
import  tensorflow as tf  
import input_data1
import numpy as np


trainroot = './train_tfrecord_new/train/' 
testroot = './train_tfrecord_new/test/' 
record = 'ETH-Bahnhof/' 
#record = 'KITTI-17/'



print '============reading testset %s =============' % record
testX_neg, testL_neg, testX_pos, testL_pos = input_data1.train_data_test(testroot+record)
        

testX = np.vstack((testX_neg,testX_pos))  #congeal pos and neg examples as a whole list
testL = np.vstack((testL_neg,testL_pos))    
index = np.random.permutation(len(testX))  
Xte = testX[index,:,:,:]
Yte = testL[index,:]


print 'total number of testset', len(Xte)
print 'number of negative', len(testX_neg)
print 'number of postive', len(testX_pos)
 
init_all = tf.global_variables_initializer()  

#%%   

with tf.Session() as session:  
    session.run(init_all)  
  
    saver = tf.train.import_meta_graph('save1/model.meta')
    saver.restore(session, 'save1/model')
    graph = tf.get_default_graph()
    teX= graph.get_tensor_by_name('teX:0')
    teY= graph.get_tensor_by_name('teY:0')
    test_inf = tf.get_collection("test_inf")[0]
    accuracy = tf.get_collection("accuracy")[0]
    test_inf = session.run(test_inf, feed_dict={teX: Xte})               
    accuracy_np = session.run(accuracy, feed_dict={teX: Xte, teY:Yte})      
    accuracy_neg = session.run(accuracy, feed_dict={teX: testX_neg, teY:testL_neg})  
    accuracy_pos = session.run(accuracy, feed_dict={teX: testX_pos, teY:testL_pos})
    print '~~~~~test accuracy %s ~~~~~' % accuracy_np
    print '~~~~~pos accuracy %s ~~~~~' % accuracy_neg
    print '~~~~~neg accuracy %s ~~~~~' % accuracy_pos
       
