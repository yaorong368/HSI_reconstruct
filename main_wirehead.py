import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# from IPython.display import display, clear_output  # For updating plots in Jupyter
# from itertools import permutations 
# from sklearn.metrics import multilabel_confusion_matrix
# import seaborn as sns
# from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_recall_fscore_support, hamming_loss, jaccard_score
# import numpy as np
import matplotlib.pyplot as plt
# import os
# import pandas as pd
from moduler import *

import sys
sys.path.append('/data/users2/yxiao11/mangoDB/wirehead')
from wirehead import WireheadGenerator
from wirehead import MongoTupleheadDataset, MongoheadDataset


import argparse



# torch.set_printoptions(sci_mode=False)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train a model with different data types")
#     parser.add_argument("-n", "--name", type=str, default="mixed",
#                         help="Specify the data type (e.g., 'mixed', 'Pristine', 'Irradiated')")
#     return parser.parse_args()



 
if __name__ == "__main__":
    # args = parse_args()
    # print(f'Using data type: {args.name}')
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CubeModel(52, 19).to(device)
    model = RNNFeatureExtractor(19, 0.1).to(device)
    # model = AlexNet(7).to(device)

    # Define Loss Function and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.011, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=0.000001)



    #set dataset pulled from mangoDB
    dataset = MongoTupleheadDataset(config_path = "/data/users2/yxiao11/mangoDB/wirehead/examples/satellite/config.yaml")

    
    
    # Initialize lists for tracking loss
    my_train_loss = []
    my_test_loss = []


    num_samples = dataset.__len__()
    print('dataset is: ', num_samples)
    model.train()
    my_loss = []
    iteration = 0
    step_size=10
    lr_decay_factor = 0.9

    # batch_size = 100

    # num_epoch = int(num_samples/batch_size)
    while True:  # Modify with a proper stopping condition (e.g., fixed number of iterations)
        
        running_loss = 0
        index = np.random.permutation(np.arange(num_samples))

        batch_size = np.random.randint(50,300)

        num_epoch = int(num_samples/batch_size)

        for i in range(num_epoch):   
            
            batch = dataset.__getitem__(index[i*batch_size: (i+1)*batch_size])
            cube = torch.cat([t.unsqueeze(0) for t in [b[0] for b in batch]], dim=0)
            labels = torch.cat([t.unsqueeze(0) for t in [b[1] for b in batch]], dim=0)
            cube, labels = cube.to(device, non_blocking=True).float(), labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            outputs = model(cube)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # scheduler.step()
            running_loss += loss.item()
            
        # learning rate update
        if iteration % step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor
            print(f"Iteration {iteration}: Learning rate updated to {optimizer.param_groups[0]['lr']:.6f}")

        # Append the new loss
        my_loss.append(running_loss/num_epoch)

        # Print loss after each iteration
    #     print(f"Iteration {iteration+1}, Loss: {loss.item():.4f}")

        # ✅ Update the plot dynamically in Jupyter Notebook

        plt.clf()  # Clears the current figure
        plt.plot(my_loss, label="Training Loss", color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Live Training Loss")
        # plt.legend()


        iteration += 1
        plt.savefig('/data/users2/yxiao11/model/satellite_project/resluts_n_model/loss.png')
        torch.save(model, f"/data/users2/yxiao11/model/satellite_project/resluts_n_model/model.pth")
    
