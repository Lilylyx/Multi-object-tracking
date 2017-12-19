#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:11:09 2017

@author: lily
"""

# -*- coding: utf-8 -*-
import  tensorflow as tf  
import input_data1
import numpy as np
import os

trainroot = './train_tfrecord_new/train/' 
testroot = './train_tfrecord_new/test/' 
 
class network(object):  
    def __init__(self):  
        with tf.variable_scope("weights"):  
            self.weights={  
                #39*39*3->36*36*20->18*18*20  
                'conv1':tf.get_variable('conv1',[4,4,6,20],initializer=tf.contrib.layers.xavier_initializer_conv2d()),  
                #18*18*20->16*16*40->8*8*40  
                'conv2':tf.get_variable('conv2',[3,3,20,40],initializer=tf.contrib.layers.xavier_initializer_conv2d()),  
                #8*8*40->6*6*60->3*3*60  
                'conv3':tf.get_variable('conv3',[3,3,40,60],initializer=tf.contrib.layers.xavier_initializer_conv2d()),  
                #3*3*60->120  
                'fc1':tf.get_variable('fc1',[4*4*60,60],initializer=tf.contrib.layers.xavier_initializer()),  
                #120->6  
                'fc2':tf.get_variable('fc2',[60,2],initializer=tf.contrib.layers.xavier_initializer()),  
                }  
        with tf.variable_scope("biases"):  
            self.biases={  
                'conv1':tf.get_variable('conv1',[20,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),  
                'conv2':tf.get_variable('conv2',[40,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),  
                'conv3':tf.get_variable('conv3',[60,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),  
                'fc1':tf.get_variable('fc1',[60,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),  
                'fc2':tf.get_variable('fc2',[2,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))  
  
            }  
  
    def inference(self,images):  
        # 向量转为矩阵  
        images = tf.reshape(images, shape=[-1, 32,32, 6])# [batch, in_height, in_width, in_channels]  
        images=(tf.cast(images,tf.float32)/255.-0.5)*2#归一化处理  
  
  
  
        #第一层  
        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='SAME'),  
                             self.biases['conv1'])  
  
        relu1= tf.nn.relu(conv1)  
        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  
  
  
        #第二层  
        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'),  
                             self.biases['conv2'])  
        relu2= tf.nn.relu(conv2)  
        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  
  
  
        # 第三层  
        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'),  
                             self.biases['conv3'])  
        relu3= tf.nn.relu(conv3)  
        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  
  
  
        # 全连接层1，先把特征图转为向量  
        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])  
  
        drop1=tf.nn.dropout(flatten,0.5)  
        fc1=tf.matmul(drop1, self.weights['fc1'])+self.biases['fc1']  
  
        fc_relu1=tf.nn.relu(fc1)  
  
        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']  
  
        return  fc2  
  
    #计算softmax交叉熵损失函数  
    def sorfmax_loss(self,predicts,labels):  
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicts, labels=labels))
        self.cost= loss  
        return self.cost  
      
    
    def inference_test(self,images):  
 
        images = tf.reshape(images, shape=[-1, 32,32, 6])# [batch, in_height, in_width, in_channels]  
        images=(tf.cast(images,tf.float32)/255.-0.5)*2#归一化处理  
  
  
  
        #第一层  
        conv1=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='SAME'),  
                             self.biases['conv1'])  
  
        relu1= tf.nn.relu(conv1)  
        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  
  
  
        #第二层  
        conv2=tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'),  
                             self.biases['conv2'])  
        relu2= tf.nn.relu(conv2)  
        pool2=tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  
  
  
        # 第三层  
        conv3=tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'),  
                             self.biases['conv3'])  
        relu3= tf.nn.relu(conv3)  
        pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')  
  
  
        # 全连接层1，先把特征图转为向量  
        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])  
  
        
        fc1=tf.matmul(flatten, self.weights['fc1'])+self.biases['fc1']  
  
        fc_relu1=tf.nn.relu(fc1)  
  
        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']  
  
        return  fc2  
  
   
    def optimer(self,loss,lr):  
        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)  
  
        return train_optimizer  
  



trX = tf.placeholder("uint8", [None, 32, 32, 6], name='trX')
trY = tf.placeholder("float32",[None,2], name='trY')
teX = tf.placeholder("uint8", [None, 32, 32, 6], name='teX')
teY = tf.placeholder("float32",[None,2], name='teY')


global_step = tf.Variable(0, trainable=False)
add_global = global_step.assign_add(1)

initial_learning_rate = 0.01 

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=100,decay_rate=0.4)

net = network()    
inf=net.inference(trX)  
loss=net.sorfmax_loss(inf,trY)  
opti=net.optimer(loss,learning_rate)  


#验证集所用  
test_inf=net.inference_test(teX)  
correct_prediction = tf.equal(tf.cast(tf.argmax(test_inf,1),tf.float32), tf.cast(tf.argmax(teY,1),tf.float32))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



init_all = tf.global_variables_initializer()  

saver = tf.train.Saver()
tf.add_to_collection('test_inf',test_inf)
tf.add_to_collection('accuracy',accuracy)
batchsize = 1000
te_batchsize = 10000

testX_neg = []
testX_pos = []
testL_neg = []
testL_pos = []
testname = os.listdir(testroot)
for record in testname:
    filename = testroot+record + '/'
    print '============reading testset %s =============' % record
    Xneg, Lneg, Xpos, Lpos = input_data1.train_data_test(filename)
if len(testX_neg)==0:
    testX_neg=Xneg
    testL_neg=Lneg
    testX_pos=Xpos
    testL_pos=Lpos
    
else:
    testX_neg = np.concatenate((testX_neg,Xneg), axis = 0)
    testL_neg = np.concatenate((testL_neg,Lneg), axis = 0)
    testX_pos = np.concatenate((testX_pos,Xpos), axis = 0)
    testL_pos = np.concatenate((testL_pos,Lpos), axis = 0)
    

testX = np.vstack((Xneg,Xpos))  #congeal pos and neg examples as a whole list
testL = np.vstack((Lneg,Lpos))    
index = np.random.permutation(len(testX))  
Xte = testX[index,:,:,:]
Yte = testL[index,:]


print 'total number of testset', len(Xte)  #testset total Xte,Yte
print 'number of negative', len(testX_neg) #testset negative testX_neg, testL_neg
print 'number of postive', len(testX_pos)  #testset positive testX_pos, testL_pos

#%%    
setnames = os.listdir(trainroot)
for record in setnames:
    filename = trainroot+record + '/'
    print '============reading trainset %s =============' % record
    
    for k in range(1):
        Xneg, Lneg, Xpos, Lpos = input_data1.train_data_posneg(filename)
       
        trainX = np.vstack((Xneg,Xpos))
        trainL = np.vstack((Lneg,Lpos))
        index = np.random.permutation(len(trainX))  
        Xtr = trainX[index,:,:,:]
        Ytr = trainL[index,:]
        
        print 'trainset number ', len(Xtr) #trainingset positive Xtr, Ytr
        #%%
        with tf.Session() as session:  
            session.run(init_all)              
            ckpt = tf.train.get_checkpoint_state("./save1") 
            if ckpt and ckpt.model_checkpoint_path: 
                
                new_saver = tf.train.import_meta_graph('save1/model.meta')
                new_saver.restore(session, "save1/model")  
                print "Model restored"
            
            
                itertimes = int(float(len(Xtr))/batchsize)  #divide training data by batchsize
                accuracy_np=session.run(accuracy, feed_dict={teX: Xte, teY:Yte})  
                accuracy_neg=session.run(accuracy, feed_dict={teX: testX_neg, teY:testL_neg})  
                accuracy_pos=session.run(accuracy, feed_dict={teX: testX_pos, teY:testL_pos})  
                print '~~~~~~~~test accuracy %s ~~~~~~~~' % accuracy_np
                print '~~~~~~~~pos accuracy %s ~~~~~~~~' % accuracy_neg
                print '~~~~~~~~neg accuracy %s ~~~~~~~~' % accuracy_pos
                for iter in range(itertimes):  
                    start = iter*batchsize
                    if iter>=len(Xtr)/batchsize:               
                        end = len(Xtr)
                    else:
                        end = (iter+1)*batchsize
                              
                    
                    loss_np,_,inf_np,_ =session.run([loss,opti,inf,add_global], feed_dict={trX: Xtr[start:end], trY:Ytr[start:end]})  
                    
                    if (iter+1)%10==0: 
                        rate=session.run(learning_rate)
                        print '*****train loss: %s , learning rate: %s *****' % (loss_np,rate)  
                        if (iter+1)%50==0:
                            accuracy_np=session.run(accuracy, feed_dict={teX: Xte, teY:Yte})  
                            accuracy_neg=session.run(accuracy, feed_dict={teX: testX_neg, teY:testL_neg})  
                            accuracy_pos=session.run(accuracy, feed_dict={teX: testX_pos, teY:testL_pos})
                            print '~~~~~~~~test accuracy %s ~~~~~~~~' % accuracy_np
                            print '~~~~~~~~pos accuracy %s ~~~~~~~~' % accuracy_neg
                            print '~~~~~~~~neg accuracy %s ~~~~~~~~' % accuracy_pos

            print 'done'
            saver.save(session, "save1/model")           
            print "Model saved"
          
print '-------------------------------------------------------------'
print '----------------------train done-----------------------------'      
print '-------------------------------------------------------------'
