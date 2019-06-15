# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:24:34 2018

@author: Zhenguo Nie, Haoliang Jiang
"""
#768*6
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import conv2d_transpose

# The best results: MSE  MAE
np.random.seed(0)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return(tf.Variable(initial))

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return(tf.Variable(initial))
	
def conv2d(x, W, s=[1,1,1,1], padding='SAME'):
	if (padding.upper() == 'VALID'):
		return (tf.nn.conv2d(x,W,strides=s,padding='VALID'))
	# SAME
	return (tf.nn.conv2d(x,W,strides=s,padding='SAME'))
	
	
def max_pool_2x2(x):
	return(tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))

def Global_Average_Pooling(x):
	return global_avg_pool(x)

def relu(x):
	return tf.nn.relu(x)

def bn(x):
	return tf.layers.batch_normalization(x)

def sigmoid(x):
	return tf.nn.sigmoid(x)

def dropout(x, keep_prob=0.5):
	return tf.nn.dropout(x, keep_prob)

def fc(x, in_cn, out_cn, name_scope, drop_out=False):
	with tf.variable_scope(name_scope):
		w = weight_variable([in_cn, out_cn])
		b = bias_variable([out_cn])
		h = relu(tf.matmul(x, w) + b)
	if drop_out:
		return dropout(h)
	else:
		return h

def seblock(x, in_cn):

	squeeze = Global_Average_Pooling(x)  # None*128

	with tf.variable_scope('sq'):
		w = weight_variable([in_cn, in_cn//16])
		b = bias_variable([in_cn//16])
		h = tf.matmul(squeeze, w) + b  # None*128/16
		excitation = relu(h)  # None*128/16

	with tf.variable_scope('ex'):
		w = weight_variable([in_cn//16, in_cn])  # None1*128
		b = bias_variable([in_cn])
		h = tf.matmul(excitation, w) + b
		excitation = sigmoid(h)  # None*128
		excitation = tf.reshape(excitation, [-1, 1, 1, in_cn])  # None*1*1*128

	return x * excitation

	# return xs_shape_reshape, xs_load_reshape, xs_boundry_reshape, ys

def training_loss_writer(path, i, model_name, cur_mse, cur_mae):
	if not os.path.exists(path):
		os.makedirs(path)
	
	if i == 0:
		with open(path + '/train_record_' + model_name + '.txt', 'w+') as text_file:
			text_file.write('epoch:{:5d}   '.format(i))
			text_file.write('train_mse: {:6.4f}   '.format(cur_mse))
			text_file.write('train_mae: {:6.4f}   '.format(cur_mae))
		text_file.close()
	else:
		with open(path + '/train_record_' + model_name + '.txt', 'a+') as text_file:
			text_file.write('epoch:{:5d}   '.format(i))
			text_file.write('train_mse: {:6.4f}   '.format(cur_mse))
			text_file.write('train_mae: {:6.4f}   '.format(cur_mae) )
		text_file.close()

def testing_loss_writer(path, model_name, cur_mse, cur_mae):
	with open(path + '/train_record_' + model_name + '.txt', 'a+') as text_file:
		text_file.write('test_mse: {:6.4f}   test_mae: {:6.4f}'.format(cur_mse,cur_mae)+'\n')
	text_file.close()

def residual_block(x, cn, scope_name):
	with tf.variable_scope(scope_name):
		shortcut = x # None*6*8*128
		w1 = weight_variable([3, 3, cn, cn])
		b1 = bias_variable([cn])
		x1 = bn(relu(conv2d(x, w1) + b1)) # None*6*8*128
		w2 = weight_variable([3, 3, cn, cn])
		b2 = bias_variable([cn])
		x2 = bn(conv2d(x1, w2) + b2) # None*6*8*128

		x3 = seblock(x2, cn) # None*6*8*128

	return x3 + shortcut


def model(x):
	w1 = weight_variable([9, 9, 5, 32])
	b1 = bias_variable([32])
	x1 = relu(bn(conv2d(x, w1) + b1)) # None*24*32*32
	'''
	w2 = weight_variable([4, 4, 32, 64])
	b2 = bias_variable([64])
	pad_x1 = tf.pad(x1, [[0 ,0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
	x2 = relu(bn(conv2d(pad_x1, w2, s=[1, 2, 2, 1], padding='VALID') + b2))
	'''
	w2 = weight_variable([3, 3, 32, 64])
	b2 = bias_variable([64])
	x2 = relu(bn(conv2d(x1, w2, s=[1, 2, 2, 1], padding='SAME') + b2)) # None*12*16*64
	
	'''
	w3 = weight_variable([4, 4, 64, 128])
	b3 = bias_variable([128])
	pad_x2 = tf.pad(x2, [[0,0 ], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
	x3 = relu(bn(conv2d(pad_x2, w3, s=[1, 2, 2, 1], padding='VALID') + b3))
	'''

	w3 = weight_variable([3, 3, 64, 128])
	b3 = bias_variable([128])
	x3 = relu(bn(conv2d(x2, w3, s=[1, 2, 2, 1], padding='SAME') + b3)) # None*6*8*128

	x4 = residual_block(x3, 128, 'res1') # None*6*8*128
	x5 = residual_block(x4, 128, 'res2') # None*6*8*128
	x6 = residual_block(x5, 128, 'res3') # None*6*8*128
	x7 = residual_block(x6, 128, 'res4') # None*6*8*128
	x8 = residual_block(x7, 128, 'res5') # None*6*8*128

	# pad_x8 = tf.pad(x8, [[0 ,0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
	x9 =  bn(relu(conv2d_transpose(x8, 64, kernel_size=(3, 3), stride=(2, 2), padding='SAME', activation_fn=None))) # None*12*16*64
	# pad_x9 = tf.pad(x9, [[0 ,0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
	x10 =  bn(relu(conv2d_transpose(x9, 32, kernel_size=(3, 3), stride=(2, 2), padding='SAME', activation_fn=None))) # None*24*32*32
	
	w4 = weight_variable([9, 9, 32, 1])
	#b4 = bias_variable([1])
	x11 = relu(conv2d(x10, w4)) # None*24*32*1

	return x11, x10, x9, x3, x2, x1

def main(argv = None):
	#f_name = '../data/rectangle_m.npy'
	#save_prefix = '../result/trained_models_rectangle_multiple_stressnet/'
	f_name = '../data/all_data_m.npy'
	save_prefix = '../result/trained_models_all_multiple_stressnet/'
	
	batch_size = 256
	decay_steps = 4000
	decay_rate = 0.995
	starter_learning_rate = 1e-4
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
	
	np.random.shuffle(np_input)
	np_input=np_input[:int(sample_rate * np.shape(np_input)[0]),:]
	total_num = np.shape(np_input)[0]
	num_train = int(train_ratio * np.shape(np_input)[0])
	#num_test = np.shape(np_input)[0]-num_train
	test_num = total_num - num_train
	np_data_train = np_input[:num_train, :]
	np_data_test = np_input[num_train:, :]

	print(num_train)

	xs = tf.placeholder(tf.float32, shape=[None, resolution, 5],name='xs_node')
	xs_reshape = tf.reshape(xs, shape=[-1, height, width, 5])
	ys = tf.placeholder(tf.float32, shape=[None, resolution])
	prediction_matrix, x10, x9, x3, x2, x1 = model(xs_reshape)
	prediction = tf.reshape(prediction_matrix, shape=[-1,resolution])
	
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
		decay_steps=decay_steps,
		decay_rate=decay_rate,
		staircase=False)
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
	model_name = 'CNN_M_RENET'
	path = save_prefix
	#session
	with tf.Session(config = config) as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		batch_test = 256
		mse_train = np.zeros(n_epochs)
		mse_test = np.zeros(n_epochs)
		mae_train = np.zeros(n_epochs)
		mae_test = np.zeros(n_epochs)
		prediction_num = 50
		prediction_history = np.zeros((prediction_num,resolution,n_epochs+1))		
		iter_num = num_train // batch_size
		test_iter_num = test_num//batch_test
		order = np.arange(num_train)
		print('Training...')
		for epoch in range(n_epochs):
			total_mse = 0
			total_mae = 0
			test_total_mse = 0
			test_total_mae = 0
			np.random.shuffle(order)
			np_data_train =  np_data_train[order]
			for iter_train in range(iter_num):
				x_batch_shape = np_data_train[iter_train*batch_size:iter_train*batch_size+batch_size,0:resolution]
				x_batch_load_x = np_data_train[iter_train*batch_size:iter_train*batch_size+batch_size,resolution:resolution*2]
				x_batch_load_y = np_data_train[iter_train*batch_size:iter_train*batch_size+batch_size,resolution*2:resolution*3]
				x_batch_boundry_x = np_data_train[iter_train*batch_size:iter_train*batch_size+batch_size,resolution*3:resolution*4]
				x_batch_boundry_y = np_data_train[iter_train*batch_size:iter_train*batch_size+batch_size,resolution*4:resolution*5]
				x_batch = np.stack((x_batch_shape, 
									x_batch_load_x,
									x_batch_load_y,
									x_batch_boundry_x,
									x_batch_boundry_y), axis=-1)
				y_batch = np_data_train[iter_train*batch_size:iter_train*batch_size+batch_size,resolution*5:resolution*6]
				_, l_rate= sess.run([add_global, learning_rate,], feed_dict={xs: x_batch,ys:y_batch})
				_, batch_loss, batch_mae = sess.run([train_step, mse, mae], feed_dict={xs:x_batch,ys:y_batch})
				total_mse += batch_loss
				total_mae += batch_mae

			print('Epoch:',epoch,', Learning rate:',l_rate)

			mse_train[epoch] = total_mse/iter_num
			mae_train[epoch] = total_mae/iter_num
			print('MSE_train:', mse_train[epoch], end = ' ')
			print('MAE_train:', mae_train[epoch])
			training_loss_writer(path, epoch, model_name, mse_train[epoch], mae_train[epoch])

			for iter_test in range(test_iter_num):
				x_test_shape = np_data_test[iter_test*batch_test:iter_test*batch_test+batch_test,0:resolution]
				x_test_load_x = np_data_test[iter_test*batch_test:iter_test*batch_test+batch_test,resolution:resolution*2]
				x_test_load_y = np_data_test[iter_test*batch_test:iter_test*batch_test+batch_test,resolution*2:resolution*3]
				x_test_boundry_x = np_data_test[iter_test*batch_test:iter_test*batch_test+batch_test,resolution*3:resolution*4]
				x_test_boundry_y = np_data_test[iter_test*batch_test:iter_test*batch_test+batch_test,resolution*4:resolution*5]
				x_test = np.stack((x_test_shape, 
								x_test_load_x,
								x_test_load_y,
								x_test_boundry_x,
								x_test_boundry_y), axis=-1)
				y_test = np_data_test[iter_test*batch_test:iter_test*batch_test+batch_test,resolution*5:resolution*6]
				test_mse,test_mae,test_prediction = sess.run([mse, mae, prediction], feed_dict={xs:x_test,ys:y_test})
				test_total_mse += test_mse
				test_total_mae += test_mae
			prediction_input = x_test[0:prediction_num,:]
			prediction_history[:,:,0] = y_test[0:prediction_num,:]			
			prediction_history[:,:,epoch+1] = test_prediction[0:prediction_num,:]
			mse_test[epoch] = test_total_mse/test_iter_num
			mae_test[epoch] = test_total_mae/test_iter_num
			print('MSE_test:', mse_test[epoch], end = '   ')
			print('MAE_test:', mae_test[epoch])
			testing_loss_writer(path, model_name, mse_test[epoch], mae_test[epoch])


			if epoch%1000 == 0:
				if not os.path.exists(path):
					os.makedirs(path)
				save_file = path + 'cnn_resnet'
				saver.save(sess,save_file,global_step=epoch,write_meta_graph=True)
				print('Session is saved: {}'.format(save_file))
	
			current_time = time.localtime()
			print('Time current: ', time.strftime('%Y-%m-%d %H:%M:%S', current_time))
	
		print('Training is finished!')

	np.save(save_prefix+'mse_train',mse_train)
	np.save(save_prefix+'mae_train',mae_train)
	np.save(save_prefix+'mse_test',mse_test)
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