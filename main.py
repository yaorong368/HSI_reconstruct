import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import ica


from moduler import *


torch.set_printoptions(sci_mode=False)

def multiclass_dice_loss(preds, targets, num_classes=72, smooth=1e-5, weight=None):
    preds = F.softmax(preds, dim=1)              # [B, C, H, W]
    targets_onehot = F.one_hot(targets, num_classes)  # [B, H, W, C]
    targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

    intersection = (preds * targets_onehot).sum(dim=(0, 2, 3))  # per class
    union = preds.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)

    if weight is not None:
        dice = dice * weight.to(dice.device)  # apply per-class weights

    return 1 - dice.mean()

def hybrid_loss(preds, targets, alpha=0.7):
    dice = multiclass_dice_loss(preds, targets)
    ce = F.cross_entropy(preds, targets)
    return alpha * dice + (1 - alpha) * ce

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] logits
        targets: [B, H, W] integer class labels (0 to C-1)
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')  # [B, H, W]
        pt = torch.exp(-ce_loss)  # pt: probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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

def compute_pos_weight_from_two_hot(labels_2hot):
    """
    Compute pos_weight for BCEWithLogitsLoss from two-hot label format.

    Args:
        labels_2hot (torch.Tensor): shape [num_samples, 19, 2]
            - [1, 0] means exist
            - [0, 1] means not exist

    Returns:
        pos_weight (torch.Tensor): shape [19], positive class weights
    """

    # Convert two-hot to binary labels [num_samples, 19]
    # We take the first column: 1 if exists, 0 otherwise
    binary_labels = labels_2hot[:, :, 0]

    # Count positives and negatives per class
    num_pos = binary_labels.sum(dim=0)                  # [19]
    num_neg = binary_labels.size(0) - num_pos           # [19]

    # Compute pos_weight (avoid division by zero)
    pos_weight = num_neg / (num_pos + 1e-5)

    return pos_weight

class get_dataset(Dataset):
    def __init__(self, data_dir, labels_dir):
        self.data_dir = data_dir
        self.labels_dir = labels_dir

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        ipt = safe_load_numpy(self.data_dir[idx])
        # ipt = (ipt - ipt.mean(axis=(1,2), keepdims=True)) / (ipt.std(axis=(1,2), keepdims=True)+ 1e-8)
        ipt = torch.from_numpy(ipt)
        # ipt,_,_ = ica.pca_whiten(ipt.reshape(100,-1), 10)
        # ipt = torch.from_numpy(ipt.reshape(10,32,32))

        tgt = torch.from_numpy(safe_load_numpy(self.labels_dir[idx]))
        # one_hot = F.one_hot(tgt, num_classes=72)
        # one_hot = one_hot.permute(0, 3, 1, 2).float()


        # tgt = torch.from_numpy(safe_load_numpy(self.labels_dir[idx])[None,:])
        # tgt = (tgt - tgt.mean(axis=(1,2), keepdims=True)) / (tgt.std(axis=(1,2), keepdims=True)+ 1e-8)


        # label_index = torch.from_numpy(safe_load_numpy(self.labels_dir[idx])).long()
        # label = torch.zeros(71)
        # label[label_index] = 1

        # label = torch.tensor([0, 1]).repeat(71, 1)
        # label[label_index] = torch.tensor([1, 0])


        
        return ipt.float(), tgt.long()
    
if __name__ == "__main__":
#--------set data
    # data_type = 'mixed'
    data_type = 'Pristine'
    # data_type = 'Irradiated'

    blur_dir = '/data/users2/yxiao11/model/satellite_project/database/' +data_type + '/blur_cube/'
    label_dir = '/data/users2/yxiao11/model/satellite_project/database/' +data_type + '/spectral_cube/'
    # label_dir = '/data/users2/yxiao11/model/satellite_project/database/' +data_type + '/label/'

    data_file = []
    label_file = []
    # spectral_file = []

    while len(os.listdir(blur_dir))<1000:
        print('insuffient data:', len(os.listdir(blur_dir)))
        time.sleep(5)


    for i in range(len(os.listdir(blur_dir))):
        data_file.append(blur_dir + f"{i}.npy")
        label_file.append(label_dir + f"{i}.npy")
    #     spectral_file.append(spectral_dir + f"{i}.npy")
    

    my_dataset = get_dataset(data_file, label_file)
    print(f"Total samples: {len(my_dataset)}")
    

#----define training pararmeter
    # batch_size = np.random.randint(50,300)
    batch_size = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # model = model = DeepLSTMTemporalEncoder(
    #     input_dim=20,          # e.g. 8
    #     hidden_dim=128,
    #     num_classes=71,
    #     num_layers=5,
    #     dropout=0.5
    #     ).to(device)
    # model = SpectralCubeNetV2(in_channels=20, num_classes=71).to(device)
    # model = CubeModel(50,71,input_size=16).to(device)
    # model = HSIReconstructor(in_channels=20, num_classes=24).to(device)
    # model = HSIReconstructor(in_channels=20, num_classes=72).to(device)
    # model = HSIReconstructor(in_channels=40, num_classes=72).to(device)
    model = SpectralSpatialHSIModel(in_channels=50, num_classes=72).to(device)
    

    loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    # criterion = torch.nn.CrossEntropyLoss()

    # lb_list = []
    # for file in label_file:
    #     label_index = torch.from_numpy(safe_load_numpy(file)).long()
    #     label = torch.tensor([0, 1]).repeat(71, 1)
    #     label[label_index] = torch.tensor([1, 0])
    #     lb_list.append(label.unsqueeze(0))

    # criterion = torch.nn.BCEWithLogitsLoss(
    #     pos_weight = compute_pos_weight_from_two_hot(torch.cat(lb_list,0)).to(device)
    # )

    # criterion = nn.BCEWithLogitsLoss()
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([20.0]).to(device))  
    # criterion = nn.MSELoss()
      
    # weights = torch.ones(24)
    # weights[0] = 0.5

    # # Move to correct device (GPU or CPU)
    # weights = weights.to(device)

    # # Use with CrossEntropyLoss
    # criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = FocalLoss(gamma=2.0)

    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


#----training
    my_loss = []
    iteration = 0
    step_size=4
    lr_decay_factor = 0.98
    count_to_stop = 0

    # while True:
    # for i in range(10000):
    while count_to_stop < 200:
        # Training Phase
        model.train()
        running_loss = 0.0

        for cube, labels in loader:
            cube, labels = cube.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(cube)
            # loss = criterion(outputs, labels)
            loss = hybrid_loss(outputs, labels)

            # Encourage at most 3 strong activations
            # prob = torch.sigmoid(outputs)  # [B, 71]
            # mask = (prob > 0.5).float()    # 1 for values > 0.5, else 0
            # activation_penalty = (prob * mask).sum(dim=1).clamp(min=0)  # total "active units" per sample
            # activation_penalty = ((activation_penalty - 3).clamp(min=0)) ** 2  # penalize > 3 active units
            # activation_penalty = activation_penalty.mean()

            # loss = bce_loss + 0.05 * activation_penalty

            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # scheduler.step()

            
            running_loss += loss.item()
        # # learning rate update
        # if iteration % step_size == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= lr_decay_factor
        #     print(f"Iteration {iteration}: Learning rate updated to {optimizer.param_groups[0]['lr']:.6f}")

        avg_train_loss = running_loss / len(loader)
        my_loss.append(avg_train_loss)
        count_to_stop += 1

        plt.clf()  # Clears the current figure
        plt.plot(my_loss, label="Training Loss", color="blue")
        # plt.ylim(0,1.5)
        plt.grid()
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"Live Training Loss_iter({iteration})_{min(my_loss)}")
        # plt.legend()


        iteration += 1
        plt.savefig('/data/users2/yxiao11/model/satellite_project/resluts_n_model/loss.png')
        if avg_train_loss <= min(my_loss):
            count_to_stop = 0
            # torch.save(model, f"/data/users2/yxiao11/model/satellite_project/resluts_n_model/model.pth")
            traced_model = torch.jit.trace(model, cube)
            traced_model.save("/data/users2/yxiao11/model/satellite_project/resluts_n_model/model.pt")
            print(f"Model saved at iteration {iteration} with loss {avg_train_loss:.6f}")
        
        
        # Print results for this epoch
        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

