import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from piq import ssim  # pip install piq
import os
from time import time
import time


def safe_load_numpy(file_path, retries=100, wait=0.1):
    for attempt in range(retries):
        try:
            data = np.load(file_path)   # Attempt to load
            return data                 # Success! Immediately returns
        except (FileNotFoundError, OSError, ValueError) as e:
            if attempt < retries - 1:   # Not the last attempt? Wait, then try again
                time.sleep(wait)
            else:                       # Last attempt and still fails
                raise e

class get_dataset(Dataset):
    def __init__(self, data_dir,):
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        ipt = safe_load_numpy(self.data_dir[idx])
       
        images_np = ipt[:, None, :, :].astype(np.float32)
        images = torch.tensor(images_np)

        return images[0:1], images[1:]
    
def build_branch():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 1, 3, padding=1),
        nn.Sigmoid()
    )
    
    
if __name__ == "__main__":
    # data_type = 'mixed'
    data_type = 'Pristine'
    # data_type = 'Irradiated'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blur_dir = '/data/users2/yxiao11/model/satellite_project/database/' +data_type + '/blur_cube/'
    data_file = []
    while len(os.listdir(blur_dir))<1000:
        print('insuffient data:', len(os.listdir(blur_dir)))
        time.sleep(5)

    for i in range(len(os.listdir(blur_dir))):
        data_file.append(blur_dir + f"{i}.npy")

    my_dataset = get_dataset(data_file)
    branches = nn.ModuleList([build_branch() for _ in range(29)]).to(device)
    optimizer = optim.Adam([p for net in branches for p in net.parameters()], lr=1e-3)

    num_epochs = 100

    for epoch in range(num_epochs):
        total_loss = 0
        count = 0
        for ref_image, input_images in my_dataset:
            ref_image, input_images = ref_image.to(device), input_images.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = [branches[i](input_images[i:i+1]) for i in range(29)]  # each: [1, 1, 25, 25]

            # SSIM loss with clamping
            loss = sum(
                1 - ssim(
                    torch.clamp(output, 0.0, 1.0),
                    torch.clamp(ref_image, 0.0, 1.0),
                    data_range=1.0
                )
                for output in outputs
            ) / len(outputs)

            loss.backward()
            optimizer.step()
            total_loss += loss
            count+=1

            print(f"count {count}, Avg SSIM-structural loss: {loss.item():.6f}")
        print('------------------------')
        print(f"Epoch {epoch+1}, Avg SSIM-structural loss: {total_loss/1000}")