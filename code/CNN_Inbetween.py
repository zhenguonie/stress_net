# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:24:34 2018

@author: ZHENGUO
"""
#768*6
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import time

def main(argv = None):
    #f_name = '../data/rectangle_m.npy'
    #save_prefix = '../result/trained_models_rectangle_multiple/'
    f_name = '../data/all_data_m.npy'
    save_prefix = '../result/trained_models_multiple/'

    batch_size = 256
    decay_steps = 2000
    decay_rate = 0.99
    starter_learning_rate = 1e-3
    train_ratio = 0.95
    n_epochs = 5001
    sample_rate =1.0

    #define the size of graph
    height = 24
    width = 32
    resolution = height*width
    np_input = np.array(np.load(f_name)).astype(np.float32)
    
    stress_mean = np.mean(np_input[:,resolution*5:resolution*6])
    stress_min = np.min(np_input[:,resolution*5:resolution*6])
    stress_max = np.max(np_input[:,resolution*5:resolution*6])
    stree_statistic = [['mean','max','min'],[stress_mean,stress_min,stress_max]]
    
    '''
    pickup_row = 6000
    plt_row_shape = np_input[pickup_row,0:resolution].reshape(24,32)
    plt_row_load_x = np_input[pickup_row,resolution:resolution*2].reshape(24,32)
    plt_row_load_y = np_input[pickup_row,resolution*2:resolution*3].reshape(24,32)
    plt_row_boundry_x = np_input[pickup_row,resolution*3:resolution*4].reshape(24,32)
    plt_row_boundry_y = np_input[pickup_row,resolution*4:resolution*5].reshape(24,32)
    plt_row_stress = np_input[pickup_row,resolution*5:resolution*6].reshape(24,32)
    
    im_node = plt.imshow(plt_row_shape,cmap='jet',interpolation='nearest')
    plt.title(r'$q(x)=$'+str(round(plt_row_load_x[11,31]))+r'$N/mm^2,$'+r'$ q(y)=$'+str(round(plt_row_load_y[11,31]))+r'$N/mm^2$', fontsize=15)
    plt.savefig('../data/shape_'+str(pickup_row)+'.png',dpi=300)
    plt.show()
    
    im_node = plt.imshow(plt_row_load_x,cmap='jet',interpolation='nearest')
    plt.title(r'$q(x)=$'+str(round(plt_row_load_x[11,31]))+r'$N/mm^2,$'+r'$ q(y)=$'+str(round(plt_row_load_y[11,31]))+r'$N/mm^2$', fontsize=15)
    plt.savefig('../data/loadx_'+str(pickup_row)+'.png',dpi=300)
    plt.show()
    
    im_node = plt.imshow(plt_row_load_y,cmap='jet',interpolation='nearest')
    plt.title(r'$q(x)=$'+str(round(plt_row_load_x[11,31]))+r'$N/mm^2,$'+r'$ q(y)=$'+str(round(plt_row_load_y[11,31]))+r'$N/mm^2$', fontsize=15)
    plt.savefig('../data/loady_'+str(pickup_row)+'.png',dpi=300)
    plt.show()
    
    im_node = plt.imshow(plt_row_boundry_x,cmap='jet',interpolation='nearest')
    plt.title(r'$q(x)=$'+str(round(plt_row_load_x[11,31]))+r'$N/mm^2,$'+r'$ q(y)=$'+str(round(plt_row_load_y[11,31]))+r'$N/mm^2$', fontsize=15)
    plt.savefig('../data/bx_'+str(pickup_row)+'.png',dpi=300)
    plt.show()
    
    im_node = plt.imshow(plt_row_boundry_y,cmap='jet',interpolation='nearest')
    plt.title(r'$q(x)=$'+str(round(plt_row_load_x[11,31]))+r'$N/mm^2,$'+r'$ q(y)=$'+str(round(plt_row_load_y[11,31]))+r'$N/mm^2$', fontsize=15)
    plt.savefig('../data/by_'+str(pickup_row)+'.png',dpi=300)
    plt.show()
    
    im_stress = plt.imshow(plt_row_stress,cmap='jet',interpolation='nearest')
    plt.title(r'$q(x)=$'+str(round(plt_row_load_x[11,31]))+r'$N/mm^2,$'+r'$ q(y)=$'+str(round(plt_row_load_y[11,31]))+r'$N/mm^2$', fontsize=15)
    plt.colorbar(im_stress)
    plt.savefig('../data/stress_'+str(pickup_row)+'.png',dpi=300)
    plt.show()
    '''
    
    np.random.shuffle(np_input)
    np_input=np_input[:int(sample_rate * np.shape(np_input)[0]),:]
    
    num_train = int(train_ratio * np.shape(np_input)[0])
    #num_test = np.shape(np_input)[0]-num_train
    
    np_data_train = np_input[:num_train, :]
    np_data_test = np_input[num_train:, :]
    
    #CNN
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return(tf.Variable(initial))
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return(tf.Variable(initial))
        
    def conv2d(x,W):
        return(tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME'))
        
    def max_pool_2x2(x):
        return(tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))
        
    #shape
    xs_shape = tf.placeholder(tf.float32, shape=[None,resolution],name='xs_node')
    xs_shape_reshape = tf.reshape(xs_shape, shape=[-1,height,width,1])
    #load 2 chaeels
    xs_load = tf.placeholder(tf.float32, shape=[None,resolution,2],name='xs_load')
    xs_load_reshape = tf.reshape(xs_load, shape=[-1,height,width,2])
    
    #boundry 2 channels
    xs_boundry = tf.placeholder(tf.float32, shape=[None,resolution,2],name='xs_boundary')
    xs_boundry_reshape = tf.reshape(xs_boundry, shape=[-1,height,width,2])
    #output stress
    ys = tf.placeholder(tf.float32, shape=[None,resolution],name='ys_stress')
    keep_prob = tf.placeholder(tf.float32)
    
    #Define the 1st CNN layer for Eecoding and Pooling
    #shape
    W_conv1_shape = weight_variable([3,3,1,32])
    b_conv1_shape = bias_variable([32])
    h_conv1_shape = tf.nn.relu(conv2d(xs_shape_reshape,W_conv1_shape)+b_conv1_shape) #Nonex24x32x32
    h_pool1_shape = max_pool_2x2(h_conv1_shape) #Nonex12x16x32
    #load
    W_conv1_load = weight_variable([3,3,2,8])
    b_conv1_load = bias_variable([8])
    h_conv1_load = tf.nn.relu(conv2d(xs_load_reshape,W_conv1_load)+b_conv1_load) #Nonex24x32x8
    h_pool1_load = max_pool_2x2(h_conv1_load) #Nonex12x16x8
    #boundry
    W_conv1_boundry = weight_variable([3,3,2,8])
    b_conv1_boundry = bias_variable([8])
    h_conv1_boundry = tf.nn.relu(conv2d(xs_boundry_reshape,W_conv1_boundry)+b_conv1_boundry) #Nonex24x32x8
    h_pool1_boundry = max_pool_2x2(h_conv1_boundry) #Nonex12x16x8
    
    
    #Define the 2nd CNN layer for Eecoding and Pooling
    #shape
    W_conv2 = weight_variable([3,3,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1_shape,W_conv2)+b_conv2) #Nonex12x16x64
    h_pool2 = max_pool_2x2(h_conv2) #6x8x64
    #load
    W_conv2_load = weight_variable([3,3,8,16])
    b_conv2_load = bias_variable([16])
    h_conv2_load = tf.nn.relu(conv2d(h_pool1_load,W_conv2_load)+b_conv2_load) #Nonex12x16x16
    h_pool2_load = max_pool_2x2(h_conv2_load) #Nonex6x8x16
    #boundry
    W_conv2_boundry = weight_variable([3,3,8,16])
    b_conv2_boundry = bias_variable([16])
    h_conv2_boundry = tf.nn.relu(conv2d(h_pool1_boundry,W_conv2_boundry)+b_conv2_boundry) #Nonex12x16x16
    h_pool2_boundry = max_pool_2x2(h_conv2_boundry) #Nonex6x8x16
    
    #define FC1
    #shape
    W_fc1 = weight_variable([6*8*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1,6*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    #load
    W_fc1_load = weight_variable([6*8*16, 128])
    b_fc1_load = bias_variable([128])
    h_pool2_flat_load = tf.reshape(h_pool2_load, [-1,6*8*16])
    h_fc1_load = tf.nn.relu(tf.matmul(h_pool2_flat_load,W_fc1_load)+b_fc1_load) #Nonex128
    h_fc1_drop_load = tf.nn.dropout(h_fc1_load,keep_prob)#Nonex128
    #boundry
    W_fc1_boundry = weight_variable([6*8*16, 128])
    b_fc1_boundry = bias_variable([128])
    h_pool2_flat_boundry = tf.reshape(h_pool2_boundry, [-1,6*8*16])
    h_fc1_boundry = tf.nn.relu(tf.matmul(h_pool2_flat_boundry,W_fc1_boundry)+b_fc1_boundry) #Nonex128
    h_fc1_drop_boundry = tf.nn.dropout(h_fc1_boundry,keep_prob)#Nonex128
    
    #define FC2
    #shape
    W_fc2 = weight_variable([1024, 32]) 
    b_fc2 = bias_variable([32])
    h_fc2 = tf.nn.softplus(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)  #Nonex32
    #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)  #Nonex32
    #load
    W_fc2_load = weight_variable([128, 8]) 
    b_fc2_load = bias_variable([8])
    h_fc2_load = tf.nn.softplus(tf.matmul(h_fc1_drop_load,W_fc2_load)+b_fc2_load)  #Nonex8
    #h_fc2_load = tf.nn.relu(tf.matmul(h_fc1_drop_load,W_fc2_load)+b_fc2_load)  #Nonex8
    #boundry
    W_fc2_boundry = weight_variable([128, 8]) 
    b_fc2_boundry = bias_variable([8])
    h_fc2_boundry = tf.nn.softplus(tf.matmul(h_fc1_drop_boundry,W_fc2_boundry)+b_fc2_boundry)  #Nonex8
    #h_fc2_boundry = tf.nn.relu(tf.matmul(h_fc1_drop_boundry,W_fc2_boundry)+b_fc2_boundry)  #Nonex8
    
    
    #Feature representation fully connected
    dna_full = tf.concat([h_fc2,h_fc2_load,h_fc2_boundry],1) #Nonex48
    
    #Define UFC3
    W_fc3 = weight_variable([48, 1024]) 
    b_fc3 = bias_variable([1024])
    h_fc3 = tf.nn.softplus(tf.matmul(dna_full,W_fc3)+b_fc3) #Nonex1024
    #h_fc3 = tf.nn.relu(tf.matmul(dna_full,W_fc3)+b_fc3) #Nonex1024
    
    #Define UFC4
    W_fc4 = weight_variable([1024, 6*8*64])
    b_fc4 = bias_variable([6*8*64])
    h_fc4 = tf.nn.relu(tf.matmul(h_fc3,W_fc4)+b_fc4) #Nonex3072
    h_fc4_drop = tf.nn.dropout(h_fc4,keep_prob)
    
    #Define Upsampling and the 3rd CNN layer for Decoding
    #reshape
    h_fc4_drop_flat = tf.reshape(h_fc4_drop, [-1,6,8,64]) #Nonex6x8x64 的向量
    h_fc4_up = tf.keras.layers.UpSampling2D(size=(2,2))(h_fc4_drop_flat) #Nonex12x16x64
    W_conv3 = weight_variable([3,3,64,32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_fc4_up,W_conv3)+b_conv3) #Nonex12x16x32
    
    #Define Upsampling and the 4th CNN layer for Decoding
    h_conv3_up = tf.keras.layers.UpSampling2D(size=(2,2))(h_conv3) #Nonex24x32x32
    W_conv4 = weight_variable([3,3,32,16])
    b_conv4 = bias_variable([16])
    h_conv4 = tf.nn.relu(conv2d(h_conv3_up,W_conv4)+b_conv4) #Nonex24x32x16
    
    #Define the 5th CNN layer for Decoding
    W_conv5 = weight_variable([3,3,16,1])
    b_conv5 = bias_variable([1])
    prediction_matrix = tf.nn.relu(conv2d(h_conv4,W_conv5)+b_conv5) #Nonex24x32x1
    prediction = tf.reshape(prediction_matrix, shape=[-1,resolution])#Nonex768
    
    #metrics
    mse = tf.losses.mean_squared_error(ys,prediction)
    mae = tf.reduce_mean(tf.abs(tf.subtract(ys,prediction)))
    #loss
    loss = mse
    
    #train rate    
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_steps,decay_rate=decay_rate)
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
    
    #set GPU
    config = tf.ConfigProto() 
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    
    
    '''
    #set CPU
    config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage  
                    inter_op_parallelism_threads = 1,   
                    intra_op_parallelism_threads = 1,  
                    log_device_placement=True)
    '''
    
    
    saver = tf.train.Saver()
    
    start_time = time.localtime()
    print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
    
    #session
    with tf.Session(config = config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        test_num = 300
        mse_train = np.zeros(n_epochs)
        mse_test = np.zeros(n_epochs)
        mae_train = np.zeros(n_epochs)
        mae_test = np.zeros(n_epochs)
        prediction_num = 50
        prediction_input = np_data_test[0:prediction_num,resolution*0:resolution*6]
        prediction_history = np.zeros((prediction_num,resolution,n_epochs+1))
        prediction_history[:,:,0] = np_data_test[0:prediction_num,resolution*5:resolution*6]
        print('Training...')
        for epoch in range(n_epochs):
            print(str(epoch)+' epoch of '+str(n_epochs))
    
            for iteration in range(num_train // batch_size):
                unit_iter = (num_train // batch_size)//6
                if iteration % unit_iter == 0:
                    print('   '+str(iteration)+' iteration of '+str(num_train // batch_size))
                x_batch_shape = np_data_train[iteration*batch_size:iteration*batch_size+batch_size,0:resolution]
                x_batch_load_x = np_data_train[iteration*batch_size:iteration*batch_size+batch_size,resolution:resolution*2]
                x_batch_load_y = np_data_train[iteration*batch_size:iteration*batch_size+batch_size,resolution*2:resolution*3]     
                x_batch_load = np.stack((x_batch_load_x,x_batch_load_y),axis=-1)
                x_batch_boundry_x = np_data_train[iteration*batch_size:iteration*batch_size+batch_size,resolution*3:resolution*4]
                x_batch_boundry_y = np_data_train[iteration*batch_size:iteration*batch_size+batch_size,resolution*4:resolution*5]     
                x_batch_boundry = np.stack((x_batch_boundry_x,x_batch_boundry_y),axis=-1)
                y_batch = np_data_train[iteration*batch_size:iteration*batch_size+batch_size,resolution*5:resolution*6]

                _, l_rate = sess.run([add_global,learning_rate], feed_dict={xs_shape:x_batch_shape, xs_load:x_batch_load, xs_boundry:x_batch_boundry, ys:y_batch, keep_prob:0.5})            
                sess.run(train_step, feed_dict={xs_shape:x_batch_shape, xs_load:x_batch_load, xs_boundry:x_batch_boundry, ys:y_batch, keep_prob:0.5})
                
            print('Epoch:',epoch,', Learning rate:',l_rate)
        
            [mse_train[epoch], mae_train[epoch]] = sess.run([mse,mae], feed_dict={xs_shape:x_batch_shape, xs_load:x_batch_load, xs_boundry:x_batch_boundry, ys:y_batch, keep_prob:0.5})
            print('Epoch:',epoch, 'mse_train:', mse_train[epoch])  
            print('Epoch:',epoch, 'MAE_train:', mae_train[epoch])  
            
            [mse_test[epoch], mae_test[epoch],prediction_temp] =  sess.run([mse, mae,prediction], feed_dict={xs_shape:np_data_test[0:test_num,0:resolution], xs_load:np.stack((np_data_test[0:test_num,resolution:resolution*2],np_data_test[0:test_num,resolution*2:resolution*3]),axis=-1),xs_boundry:np.stack((np_data_test[0:test_num,resolution*3:resolution*4],np_data_test[0:test_num,resolution*4:resolution*5]),axis=-1), ys:np_data_test[0:test_num,resolution*5:resolution*6], keep_prob:0.5})
            prediction_history[:,:,epoch+1] = prediction_temp[0:prediction_num,:]
            print('Epoch:',epoch, 'mse_test:', mse_test[epoch])
            print('Epoch:',epoch, 'MAE_test:', mae_test[epoch])
            
            if epoch%1000 == 0:
                save_path = save_prefix
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_file = save_prefix + 'cnn_multiple'
                saver.save(sess,save_file,global_step=epoch,write_meta_graph=True)
                print('Session is saved: {}'.format(save_path))
    
            current_time = time.localtime()
            print('Time current: ', time.strftime('%Y-%m-%d %H:%M:%S', current_time))
    
        print('Training is finished!')
           
    np.save(save_prefix+'mse_train',mse_train)
    np.save(save_prefix+'mse_test',mse_test)
    np.save(save_prefix+'mae_train',mae_train)
    np.save(save_prefix+'mae_test',mae_test)
    np.save(save_prefix+'prediction_input',prediction_input)
    np.save(save_prefix+'prediction_history',prediction_history)
    np.save(save_prefix+'stree_statistic',stree_statistic)
    
    print('Mean stress is:',stress_mean)
    end_time = time.localtime()
    print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
    print('Time end: ', time.strftime('%Y-%m-%d %H:%M:%S', end_time))

if __name__ == '__main__':
    main()