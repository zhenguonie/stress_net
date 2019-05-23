# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:27:52 2018

@author: ZHENGUO
"""
#import pprint
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
#save_prefix = '../result/trained_models_all_s/'
#save_prefix = '../result/trained_models_rectangle_s/'
#save_prefix = '../result/trained_models_all_m/'
#save_prefix = '../result/trained_models_rectangle_m/'
save_prefix = '../result/trained_models_all_multiple_new_v2/'

stree_statistic = np.load(save_prefix+'stree_statistic.npy')

End = 500
mse_train_total = np.load(save_prefix+'mse_train.npy')
mse_train = mse_train_total[0:End]
mse_test_total = np.load(save_prefix+'mse_test.npy')
mse_test = mse_test_total[0:End]
mae_train_total = np.load(save_prefix+'mae_train.npy')
mae_train = mae_train_total[0:End]
mae_test_total = np.load(save_prefix+'mae_test.npy')
mae_test = mae_test_total[0:End]
#pprint.pprint(mse_train)
#pprint.pprint(mse_test)
#pprint.pprint(mae_train)
#pprint.pprint(mae_test)


plt.figure(1)
plt.plot(np.arange(np.size(mse_train)), mse_train, 'r--')
plt.xlabel('Epochs')
plt.ylabel('MSE of training data')
plt.savefig(save_prefix+'MSE_train.png',dpi=100) 
plt.show()
    
plt.figure(2)
plt.plot(np.arange(np.size(mse_test)), mse_test, 'g--')
plt.xlabel('Epochs')
plt.ylabel('MSE of test data')        
plt.savefig(save_prefix+'MSE_test.png',dpi=100)
plt.show()
    
plt.figure(3)
plt.plot(np.arange(np.size(mse_train)), mse_train, 'r--')
plt.plot(np.arange(np.size(mse_test)), mse_test, 'g--')
plt.xlabel('Epochs')
plt.ylabel('MSE')
label = ["Training data", "Test data"]  
plt.legend(label, loc = 0, ncol = 1)  
plt.savefig(save_prefix+'MSE.png',dpi=300)
plt.show()

plt.figure(4)
plt.plot(np.arange(np.size(mae_train)), mae_train, 'r--')
plt.plot(np.arange(np.size(mae_test)), mae_test, 'g--')
plt.xlabel('Epochs')
plt.ylabel('MAE')
label = ["Training data", "Test data"]  
plt.legend(label, loc = 0, ncol = 1)  
plt.savefig(save_prefix+'MAE.png',dpi=200)
plt.show()



prediction_history = np.load(save_prefix+'prediction_history.npy')
for i in range(np.shape(prediction_history)[0]): 
    plt.figure(figsize=(50,10))    
    epoch =   [0,1,10,100,2000] 
    k=0
    for j in epoch:
        k += 1
        ax = plt.subplot(1,np.size(epoch),k)
        data_stress = prediction_history[i,:,j].reshape(24,32)        
        im_stress =ax.imshow(data_stress,cmap='jet',interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.20)
        cb = plt.colorbar(im_stress, cax=cax)
        #cb.set_label('Stress(MPa)',fontdict=font)
        cb.ax.tick_params(labelsize='30')
    
    fname = save_prefix+'Test_'+str(i)
    plt.savefig(fname,dpi=300, bbox_inches='tight')
    plt.show()