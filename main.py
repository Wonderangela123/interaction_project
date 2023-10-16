import os
import numpy as np

from train_test import train_test
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

all_m1s = []
all_m2s = []
all_m3s = []

for rep in range(5):
    if __name__ == "__main__":    
        data_folder = 'ROSMAP'
        view_list = [1,2,3]
        num_epoch_pretrain = 500
        num_epoch = 2500
        lr_e_pretrain = 1e-3
        lr_e = 5e-4
        lr_c = 1e-3

        if data_folder == 'ROSMAP':
            num_class = 2
        if data_folder == 'BRCA':
            num_class = 5

        matrices = train_test(data_folder, view_list, num_class,
                              lr_e_pretrain, lr_e, lr_c, 
                              num_epoch_pretrain, num_epoch)    

        all_m1s.append(matrices['m1'])
        all_m2s.append(matrices['m2'])
        all_m3s.append(matrices['m3'])

all_m1s = np.array(all_m1s)
all_m2s = np.array(all_m2s)
all_m3s = np.array(all_m3s)

mean_m1 = np.mean(all_m1s, axis=0)
mean_m2 = np.mean(all_m2s, axis=0)
mean_m3 = np.mean(all_m3s, axis=0)

std_m1 = np.std(all_m1s, axis=0)
std_m2 = np.std(all_m2s, axis=0)
std_m3 = np.std(all_m3s, axis=0)


for i in range(len(mean_m1)):
    print("\nTest: Epoch {:d}".format(50*i))
    print("Mean m1: {:.3f}, Stddev: {:.3f}".format(mean_m1[i], std_m1[i]))
    print("Mean m2: {:.3f}, Stddev: {:.3f}".format(mean_m2[i], std_m2[i]))
    print("Mean m3: {:.3f}, Stddev: {:.3f}".format(mean_m3[i], std_m3[i]))
