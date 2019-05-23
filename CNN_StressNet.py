# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:24:34 2018

@author: Haoliang Jiang

In this version, the bug of bn is fixed.

"""
#768*6
import os
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import conv2d_transpose
import argparse

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

def bn(x, is_training=True):
	return tf.layers.batch_normalization(x, training=is_training)

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

def training_loss_writer(path, i, model_name, cur_loss, cur_mae):
	if not os.path.exists(path):
		os.makedirs(path)
	
	if i == 0:
		with open(path + '/train_record_' + model_name + '.txt', 'a+') as text_file:
			text_file.write('epoch {:02d}:'.format(i) + '\n')
			text_file.write('train mse: {:04f} '.format(cur_loss) + '\n')
			text_file.write('train mae: {:04f} '.format(cur_mae) + '\n')
		text_file.close()
	else:
		with open(path + '/train_record_' + model_name + '.txt', 'a+') as text_file:
			text_file.write('epoch {:02d}:'.format(i) + '\n')
			text_file.write('train mse: {:04f} '.format(cur_loss) + '\n')
			text_file.write('train mae: {:04f} '.format(cur_mae) + '\n')
		text_file.close()

def testing_loss_writer(path, model_name, cur_loss, cur_mae):
	with open(path + '/train_record_' + model_name + '.txt', 'a+') as text_file:
		text_file.write('test mse: {:04f}'.format(cur_loss) + '\n')
		text_file.write('test mae: {:04f}'.format(cur_mae) + '\n')
	text_file.close()

def residual_block(x, cn, scope_name, is_training=True):
	with tf.variable_scope(scope_name):
		shortcut = x # None*6*8*128
		w1 = weight_variable([3, 3, cn, cn])
		b1 = bias_variable([cn])
		x1 = bn(relu(conv2d(x, w1) + b1), is_training=is_training) # None*6*8*128
		w2 = weight_variable([3, 3, cn, cn])
		b2 = bias_variable([cn])
		x2 = bn(conv2d(x1, w2) + b2, is_training=is_training) # None*6*8*128

		x3 = seblock(x2, cn) # None*6*8*128

	return x3 + shortcut


def model(x, reuse=False, is_training=True):
	with tf.variable_scope("autoencoder") as scope:

		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse == False

		w1 = weight_variable([9, 9, 5, 32])
		b1 = bias_variable([32])
		x1 = relu(bn(conv2d(x, w1) + b1), is_training=is_training) # None*24*32*32
		'''
		w2 = weight_variable([4, 4, 32, 64])
		b2 = bias_variable([64])
		pad_x1 = tf.pad(x1, [[0 ,0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
		x2 = relu(bn(conv2d(pad_x1, w2, s=[1, 2, 2, 1], padding='VALID') + b2))
		'''
		w2 = weight_variable([3, 3, 32, 64])
		b2 = bias_variable([64])
		x2 = relu(bn(conv2d(x1, w2, s=[1, 2, 2, 1], padding='SAME') + b2, is_training=is_training)) # None*12*16*64
		
		'''
		w3 = weight_variable([4, 4, 64, 128])
		b3 = bias_variable([128])
		pad_x2 = tf.pad(x2, [[0,0 ], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
		x3 = relu(bn(conv2d(pad_x2, w3, s=[1, 2, 2, 1], padding='VALID') + b3))
		'''

		w3 = weight_variable([3, 3, 64, 128])
		b3 = bias_variable([128])
		x3 = relu(bn(conv2d(x2, w3, s=[1, 2, 2, 1], padding='SAME') + b3, is_training=is_training)) # None*6*8*128

		x4 = residual_block(x3, 128, scope_name='res1', is_training=is_training) # None*6*8*128
		x5 = residual_block(x4, 128, scope_name='res2', is_training=is_training) # None*6*8*128
		x6 = residual_block(x5, 128, scope_name='res3', is_training=is_training) # None*6*8*128
		x7 = residual_block(x6, 128, scope_name='res4', is_training=is_training) # None*6*8*128
		x8 = residual_block(x7, 128, scope_name='res5', is_training=is_training) # None*6*8*128

		# pad_x8 = tf.pad(x8, [[0 ,0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
		x9 =  bn(relu(conv2d_transpose(x8, 64, kernel_size=(3, 3), stride=(2, 2), padding='SAME', activation_fn=None)), is_training=is_training) # None*12*16*64
		# pad_x9 = tf.pad(x9, [[0 ,0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
		x10 =  bn(relu(conv2d_transpose(x9, 32, kernel_size=(3, 3), stride=(2, 2), padding='SAME', activation_fn=None)), is_training=is_training) # None*24*32*32
		
		w4 = weight_variable([9, 9, 32, 1])
		b4 = bias_variable([1])
		x11 = relu(conv2d(x10, w4)) # None*24*32*1

	return x11, x10, x9, x3, x2, x1


def main(argv = None):
	f_name = argv.sample_dir #'../data/all_data_m.npy'
	save_prefix = argv.checkpoint_dir #'../result/trained_models_multiple_resnet_v5/'
	if not os.path.exists(save_prefix):
		os.makedirs(save_prefix)

	batch_size = argv.batch_size
	decay_steps = argv.decay_steps
	decay_rate = argv.decay_rate
	starter_learning_rate = argv.lr
	# train_ratio = argv.train_ratio
	n_epochs = argv.epoch
	sample_number = argv.sample_number
	print(sample_number)
	batch_test = 256
	# validation_ratio = argv.validation_ratio
	lr = args.lr
	#define the size of graph
	height = argv.height
	width = argv.width
	resolution = height*width
	np_input = np.array(np.load(f_name)).astype(np.float32)
	start_epoch = 0

	if args.restore != 'none':
		print('restore model:')
		parameter, start_epoch = restore.split('-')
		print('epoch start from:', start_epoch + 1)
		parameter = parameter.split('_')
		sample_number = parameter[-1]
		print(sample_number)

	# if not os.path.exists('.'.join(f_name.split('.')[:-1])+'_train.npy') or not os.path.exists('.'.join(f_name.split('.')[:-1])+'_test.npy') or args.restart == True:
	# np.random.shuffle(np_input)
	# if the seperated data no exist(all data-> train and test)
	order = np.arange(np.shape(np_input)[0])
	np.random.shuffle(order)
	np_input = np_input[order]

	print('data_amount:', np.shape(np_input)[0])

	stress_mean = np.mean(np_input[:,resolution*5:resolution*6])
	stress_min = np.min(np_input[:,resolution*5:resolution*6])
	stress_max = np.max(np_input[:,resolution*5:resolution*6])
	stree_statistic = [['mean','max','min'],[stress_mean,stress_min,stress_max]]

	print(stree_statistic)

	# total_num = np.shape(np_input)[0]
	# train_num = int(sample_number * 10000)
	# #num_test = np.shape(np_input)[0]-train_num
	# test_num = total_num - train_num
	# np_data_train = np_input[:train_num, :]
	# np_data_test_final = np_input[train_num:, :]

	# order = np.arange(np.shape(np_data_test_final)[0])
	# np.random.shuffle(order)
	# np_data_test_final = np_data_test_final[order]
	# np_data_test =  np_data_test_final[:10000, :]

	# np.save('.'.join(f_name.split('.')[:-1])+str(sample_number)+'_train.npy', np_data_train)
	# np.save('.'.join(f_name.split('.')[:-1])+str(sample_number)+'_test_total.npy', np_data_test_final)
	# np.save('.'.join(f_name.split('.')[:-1])+str(sample_number)+'_test.npy', np_data_test)

	np_data_train = np.load('../data/all_data_m2.0_train_padded.npy')
	np_data_test = np.load('../data/all_data_m2.0_test_padded.npy')


	# np_data_train = np.load('.'.join(f_name.split('.')[:-1])+'_train.npy')
	# np_data_test = np.load('.'.join(f_name.split('.')[:-1])+'_test.npy')
	# if int(sample_number) != 12:
	# 	print("sample number: ", sample_number)
	# 	total_num = int(sample_number*10000)
	# 	train_num = int(total_num*train_ratio)
	# 	test_num = total_num - train_num
	# 	np_data_train = np_data_train[:train_num, :]
	# 	np_data_test = np_data_test[:test_num, :]

	train_num = np.shape(np_data_train)[0]
	test_num = np.shape(np_data_test)[0]
	print('train: ', train_num // 256 * 256)
	print('test', test_num // 256 * 256)

	# np.random.shuffle(np_data_train)
	# validation_num = int(validation_ratio * train_num)
	# train_num = train_num - validation_num
	# np_data_validation = np_data_train[:validation_num, :]
	# np_data_train = np_data_train[validation_num:, :]


	xs = tf.placeholder(tf.float32, shape=[None, resolution, 5],name='xs_node')
	xs_reshape = tf.reshape(xs, shape=[-1, height, width, 5])
	ys = tf.placeholder(tf.float32, shape=[None, resolution])

	prediction_matrix, x10, x9, x3, x2, x1 = model(xs_reshape)
	prediction = tf.reshape(prediction_matrix, shape=[-1,resolution])
	mse = tf.losses.mean_squared_error(ys,prediction)
	mae = tf.reduce_mean(tf.abs(tf.subtract(ys,prediction)))

	prediction_matrix_test, _, _, _, _, _ = model(xs_reshape, reuse=True, is_training=False)
	prediction_test = tf.reshape(prediction_matrix_test, shape=[-1,resolution])
	mse_test = tf.losses.mean_squared_error(ys,prediction_test)
	mae_test = tf.reduce_mean(tf.abs(tf.subtract(ys,prediction_test)))

	#train rate    
	global_step = tf.Variable(0, trainable=False)
	add_global = global_step.assign_add(1)
	learning_rate = tf.train.exponential_decay(starter_learning_rate,
		global_step=global_step,
		decay_steps=decay_steps,
		decay_rate=decay_rate,
		staircase=False)
	train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(mse)

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

	start_time = time.localtime()
	print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
	model_name = 'CNN_M_resnet_v6_gan_comparison'
	path = save_prefix
	saver = tf.train.Saver(max_to_keep=1000)
	#session
	with tf.Session(config = config) as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		if args.restore != 'none':
			saver.restore(sess, checkpoint_dir + args.restore)
			print('load ', args.resotre)
		mse_train = np.zeros(n_epochs)
		mse_test = np.zeros(n_epochs)
		mae_train = np.zeros(n_epochs)
		mae_test = np.zeros(n_epochs)
		mse_validation = np.zeros(n_epochs)
		mae_validation = np.zeros(n_epochs)
		prediction_num = 50
		prediction_history = np.zeros((prediction_num,resolution,n_epochs+1))		
		iter_num = train_num // batch_size
		test_iter_num = test_num // batch_test
		# validation_iter_num = validation_num // batch_test
		batch_validation = batch_test
		order = np.arange(train_num)
		print('Training...')
		for epoch in range(start_epoch, n_epochs):
			print('epoch:' + str(epoch))
			total_mse = 0
			total_mae = 0
			test_total_mse = 0
			test_total_mae = 0
			validation_total_mse = 0
			validation_total_mae = 0
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

			print('Learning rate:',l_rate, end='	')

			mse_train[epoch] = total_mse/iter_num
			mae_train[epoch] = total_mae/iter_num
			print('MSE_train:', mse_train[epoch], end ='	')
			print('MAE_train:', mae_train[epoch])
			training_loss_writer(path, epoch, model_name + str(sample_number), mse_train[epoch], mae_train[epoch])

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
				test_mse, test_mae, test_prediction = sess.run([mse_test, mae_test, prediction_test], feed_dict={xs:x_test,ys:y_test})
				test_total_mse += test_mse
				test_total_mae += test_mae
				if iter_test == 0:
					x_test_cat = x_test
					y_test_cat = y_test
					test_prediction_cat = test_prediction
				else:
					x_test_cat = np.concatenate((x_test_cat, x_test), axis=0)
					y_test_cat = np.concatenate((y_test_cat, y_test), axis=0)
					test_prediction_cat = np.concatenate((test_prediction_cat, test_prediction), axis=0)

			mse_test[epoch] = test_total_mse/test_iter_num
			mae_test[epoch] = test_total_mae/test_iter_num
			print('MSE_test:', mse_test[epoch], end = '   ')
			print('MAE_test:', mae_test[epoch])
			testing_loss_writer(path, model_name + str(sample_number), mse_test[epoch], mae_test[epoch])
			
			if (epoch%500 == 0) or (epoch >= 4980):
				if not os.path.exists(path):
					os.makedirs(path)
				save_file = path + '_'.join((model_name, str(sample_number)))
				saver.save(sess,save_file,global_step=epoch,write_meta_graph=True)

		prediction_input = x_test_cat
		prediction_gt = y_test_cat
		prediction_history = test_prediction_cat

		np.save(save_prefix+'prediction_input_' + str(sample_number) + '_' + model_name, prediction_input)
		np.save(save_prefix+'prediction_gt_' + str(sample_number) + '_' + model_name, prediction_gt)
		np.save(save_prefix+'prediction_history_' + str(sample_number) + '_' + model_name, prediction_history)
		np.save(save_prefix+'mse_train_' + str(sample_number) + '_' + model_name, mse_train)
		np.save(save_prefix+'mae_train_' + str(sample_number) + '_' + model_name, mae_train)
		np.save(save_prefix+'mse_test_' + str(sample_number) + '_' + model_name, mse_validation)
		np.save(save_prefix+'mae_test_' + str(sample_number) + '_' + model_name, mae_validation)

		# 	for iter_validation in range(validation_iter_num):
		# 		x_validation_shape = np_data_validation[iter_validation*batch_validation:iter_validation*batch_validation+batch_validation,0:resolution]
		# 		x_validation_load_x = np_data_validation[iter_validation*batch_validation:iter_validation*batch_validation+batch_validation,resolution:resolution*2]
		# 		x_validation_load_y = np_data_validation[iter_validation*batch_validation:iter_validation*batch_validation+batch_validation,resolution*2:resolution*3]
		# 		x_validation_boundry_x = np_data_validation[iter_validation*batch_validation:iter_validation*batch_validation+batch_validation,resolution*3:resolution*4]
		# 		x_validation_boundry_y = np_data_validation[iter_validation*batch_validation:iter_validation*batch_validation+batch_validation,resolution*4:resolution*5]
		# 		x_validation = np.stack((x_validation_shape, 
		# 						x_validation_load_x,
		# 						x_validation_load_y,
		# 						x_validation_boundry_x,
		# 						x_validation_boundry_y), axis=-1)
		# 		y_validation = np_data_validation[iter_validation*batch_validation:iter_validation*batch_validation+batch_validation,resolution*5:resolution*6]
		# 		validation_mse,validation_mae = sess.run([mse, mae], feed_dict={xs:x_validation,ys:y_validation})
		# 		validation_total_mse += validation_mse
		# 		validation_total_mae += validation_mae
		# 	mse_validation[epoch] = validation_total_mse/validation_iter_num
		# 	mae_validation[epoch] = validation_total_mae/validation_iter_num
		# 	print('MSE_validation:', mse_validation[epoch], end = '   ')
		# 	print('MAE_validation:', mae_validation[epoch])
		# 	testing_loss_writer(path, model_name + str(sample_number), mse_validation[epoch], mae_validation[epoch])

				# print('Session is saved: {}'.format(save_file))
	
		# 	# current_time = time.localtime()
		# 	# print('Time current: ', time.strftime('%Y-%m-%d %H:%M:%S', current_time))

		print('Training is finished!')

		# np.save(save_prefix+'mse_train_' + str(sample_number),mse_train)
		# np.save(save_prefix+'mae_train_' + str(sample_number),mae_train)
		# np.save(save_prefix+'mse_validation_' + str(sample_number),mse_validation)
		# np.save(save_prefix+'mae_validation_' + str(sample_number),mae_validation)
		# # np.save(save_prefix+'prediction_input',prediction_input)
		# # np.save(save_prefix+'prediction_history',prediction_history)
		# np.save(save_prefix+'stree_statistic_' +  str(sample_number),stree_statistic)

	# print('Mean stress is:',stress_mean)
	end_time = time.localtime()
	print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
	print('Time end: ', time.strftime('%Y-%m-%d %H:%M:%S', end_time))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--epoch', dest='epoch', type=int, default=5021, help='# of epoch')
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help='# images in batch')
	parser.add_argument('--test_size', dest='test_size', type=int, default=1024, help='# images in test batch')
	parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='../result/trained_models_multiple_resnet_v4/', help='models are saved here')
	parser.add_argument('--sample_dir', dest='sample_dir', default='../data/all_data_m.npy', help='sample are saved here')
	parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
	parser.add_argument('--restore', dest='restore', default='none', help='the name of the model you restore')
	parser.add_argument('--decay_steps', dest='decay_steps', type=float, default=6000, help='lr decay_steps')
	parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.995, help='lr decay_rate')
	parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='starter_learning_rate')
	parser.add_argument('--train_ratio', dest='train_ratio', type=float, default=0.8, help='not useful in this case')
	parser.add_argument('--validation_ratio', dest='validation_ratio', type=float, default=0, help='validation data/training data')
	parser.add_argument('--sample_number', dest='sample_number', type=float, default=10, help='the total number of training data * 10k')
	parser.add_argument('--height', dest='height', type=int, default=24, help='image height')
	parser.add_argument('--width', dest='width', type=int, default=32, help='image width')
	parser.add_argument('--restart', dest='restart', type=bool, default=False, help='restart to shuffle the data and spit it into training data and test data')

	args = parser.parse_args()
	main(args)