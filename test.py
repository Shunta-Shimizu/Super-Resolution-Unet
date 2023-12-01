import torch
import torch.nn as nn
import torchvision
from model import SuperResolution
from PIL import Image
import numpy as np
import glob

test_datas = glob.glob("./dataset/test/low_11/*")
model_path = "./model/572_4_epoch50/572_4_super_resolution.pth"
img_size = 572

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = SuperResolution().to(device)
model.load_state_dict(torch.load(model_path))

model.eval()

for t_data in test_datas:
    name = t_data.split("./dataset/test/low_4/")[1]
    low_img = Image.open(t_data).convert("RGB")
    low_img = low_img.resize((img_size, img_size))
    low_img.save("./dataset/test/572_4_inputs/"+name)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    low_img = transform(low_img)
    low_img = low_img.unsqueeze(0).to(device)

    output = model(low_img)
    torchvision.utils.save_image(output, fp="./dataset/test/572_4_epoch50/"+name)