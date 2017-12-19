# -*- coding: UTF-8 -*-
import os  
import tensorflow as tf  
import cv2
import csv

cwd = os.getcwd()  #返回当前工作目录

#%%

def read_gtfile(filename_queue):

    reader = tf.TextLineReader()  
    key, value = reader.read(filename_queue)  
    record_defaults = [[1], [1], [1.], [.1], [1.], [1.], [1.], [1.], [1.], [1.]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(records = value, record_defaults = record_defaults)
    col3 = tf.cast(col3,tf.int32)
    col4 = tf.cast(col4,tf.int32)
    col5 = tf.cast(col5,tf.int32)
    col6 = tf.cast(col6,tf.int32)
            
    return col1, col2, col3, col4, col5, col6


def inputPipeLine(fileNames, numEpochs = None):
    fileNameQueue = tf.train.string_input_producer(fileNames, num_epochs = numEpochs)  
    frame, label, offset_x, offset_y, bound_w, bound_h = read_gtfile(fileNameQueue)  
    return frame, label, offset_x, offset_y, bound_w, bound_h


def crop_image(img_path, x, y, w, h):
    if (x<0):
        x=0
    if (y<0):
        y=0
    if (w<0):
        w=0
    if (h<0):
        h=0
    image = cv2.imread(img_path)
    crop_img = image[y:y+h,x:x+w ,:]
    crop_image = cv2.resize(crop_img, (32,32)) 
    return crop_image


def write_file(setroot):
    filenames = [cwd+'/train/'+setroot+"/gt/gt.csv"]
    
    fileNameQueue = tf.train.string_input_producer(filenames, 1)
    frame, label, offset_x, offset_y, bound_w, bound_h = read_gtfile(fileNameQueue)
    classes = os.listdir(cwd+'/train/'+setroot+"/img1")  #返回目录下的文件列表   
    pic_list=[]
    
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()  
    
    sess = tf.Session()
    sess.run(init_op)
    sess.run(local_init_op)
    
    print filenames
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)  
    
    #%%
    try:
        while not coord.should_stop():
            
            f,l,x,y,w,h = sess.run([frame, label, offset_x, offset_y, bound_w, bound_h])
            pic_list.append([f,l,x,y,w,h])
    except tf.errors.OutOfRangeError:
            print '---------Read Done----------'
    finally:
            coord.request_stop()
    
    # Wait for threads to finish  
    coord.join()  
    
    pic_list = sorted(pic_list, key=lambda x: x[1])
    #%%
    k = []
    number_list = []
    a = pic_list[0][1] #label
    number = 0
    #m = [pic_list[0][1]]
    m=1
    for i in range(len(pic_list)):
        b = pic_list[i][1] #next of the same label
        if a == b:
            number += 1
            
        else:   
            m+=1 #label number
            k.append(i) #index of change
            number_list.append(['train_object'+str(a)+'.tfrecords', number])
            a = b
            number = 1
    number_list.append(['train_object'+str(a)+'.tfrecords', number])
    k.append(i)

    #%%
    if os.path.exists('./object_number')==False:
    	os.makedirs('./object_number')
        
    csvfile = open('./object_number/'+setroot+'.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(number_list)
    
    csvfile.close()
    #%%
    if os.path.exists("./train_tfrecord/"+setroot)==False:
    	os.makedirs("./train_tfrecord/"+setroot)
    path = "./train_tfrecord/"+setroot
    count = 0 
    for cut in range(m): 
        writer = tf.python_io.TFRecordWriter(path +"/train_object" + str(cut+1) + ".tfrecords")  
        while (count<k[cut]):
            #print pic_list[count]
            pic_name = cwd + '/train/'+setroot + '/img1/' + str('%06d' % pic_list[count][0]) + '.jpg'
            image = crop_image(pic_name, pic_list[count][2],pic_list[count][3],pic_list[count][4],pic_list[count][5])
        	#img = sess.run(image)
            img_raw = image.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "frame": tf.train.Feature(int64_list=tf.train.Int64List(value=[pic_list[count][0]])),     
            	   "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[pic_list[count][1]])),  
            	   "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
            count += 1
            writer.write(example.SerializeToString()) 
        #print count     
        print ('object',cut+1)
        writer.close()
        
    sess.close()
    print '---------Write Done---------'



setroot = 'TUD-Campus'
print "start", setroot
write_file(setroot)
print "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
print "************* %s finished **************" % setroot
print "+++++++++++++++++++++++++++++++++++++++++++++++++++++"