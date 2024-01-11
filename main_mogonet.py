import os
import numpy as np
import argparse

from train_test import train_test
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

all_m1s = []
all_m2s = []
all_m3s = []

for rep in range(5):
    if __name__ == "__main__":    
        # Set up argument parsing
        parser = argparse.ArgumentParser(description='Main Mogonet Script')
        parser.add_argument('--data_folder', type=str, required=True, help='Data folder path')

        # Parse arguments
        args = parser.parse_args()

        data_folder = args.data_folder
        view_list = [1,2,3]
        num_epoch_pretrain = 500
        num_epoch = 5000
        lr_e_pretrain = 1e-3
        lr_e = 5e-4
        lr_c = 1e-3

        if data_folder == 'ROSMAP':
            num_class = 2
        if data_folder == 'BRCA':
            num_class = 5
        if data_folder == 'KIPAN':
            num_class = 3
        if data_folder == 'LGG':
            num_class = 2
        if data_folder == 'BRCA/processed':
            num_class = 4
        if data_folder == 'OV/processed':
            num_class = 4
        if data_folder == 'KIPAN/processed':
            num_class = 3
        if data_folder == 'GBMLGG/processed':
            num_class = 2
            view_list = [1,2]
        if data_folder == 'BMD/processed':
            num_class = 2
            view_list = [1,2,3,4]
            
        matrices = train_test(os.path.join(data_folder, str(rep+1)), view_list, num_class,
                            lr_e_pretrain, lr_e, lr_c, 
                            num_epoch_pretrain, num_epoch)   
    # train_test(data_folder, view_list, num_class,
    #            lr_e_pretrain, lr_e, lr_c, 
    #            num_epoch_pretrain, num_epoch)                         

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
    print("Mean ACC: {:.3f}, Stddev: {:.3f}".format(mean_m1[i], std_m1[i]))
    print("Mean F1: {:.3f}, Stddev: {:.3f}".format(mean_m2[i], std_m2[i]))
    print("Mean AUC: {:.3f}, Stddev: {:.3f}".format(mean_m3[i], std_m3[i]))
