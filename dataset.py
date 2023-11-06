import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import glob
from PIL import Image

class LowHighDataSet(Dataset):
    def __init__(self, transform, folder_path):
        self.transform = transform
        self.folder_path = folder_path
        self.input_images = glob.glob(folder_path+"low_4/*")
        self.gt_images = glob.glob(folder_path+"high/*")
    
    def __getitem__(self, index):
        name = self.input_images[index].split(self.folder_path+"low_4/")[1]
        input_image = Image.open(self.input_images[index]).convert("RGB")
        img_size = 572
        input_image = input_image.resize((img_size, img_size))
        gt_image = Image.open(self.folder_path+"high/"+name).convert("RGB")
        gt_image = gt_image.resize((img_size, img_size))

        input_image = self.transform(input_image)
        gt_image = self.transform(gt_image)

        return input_image, gt_image
    
    def __len__(self):
        return len(self.input_images)

