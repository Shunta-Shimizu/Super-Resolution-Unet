import torch 
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from model import SuperResolution
from dataset import LowHighDataSet
from tqdm import tqdm

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_folder_path = "/home/shimizu/CV_project/dataset/train/"
train_dataset = LowHighDataSet(transform, train_folder_path)

test_folder_path = "/home/shimizu/CV_project/dataset/test/"
test_dataset = LowHighDataSet(transform, test_folder_path)

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

epochs = 50
# learning_rate=1e-5, weight_decay=1e-5 ./model/572_4_epoch50.pth
learning_rate = 1e-5
weight_decay = 1e-5

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

model = SuperResolution().to(device)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if device == "cuda":
    model = torch.nn.DataParallel(model)

test_loss = []
for i in range(epochs):
    print("epoch{}:".format(str(i+1)))
    model.train()
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    sum_loss = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.data

    print("Test mean loss={}".format(sum_loss * batch_size / len(test_loader.dataset)))
    test_loss.append(sum_loss * batch_size / len(test_loader.dataset))

    if i == 0:
        min_loss = sum_loss * batch_size / len(test_loader.dataset)
    else:
        loss_i = sum_loss * batch_size / len(test_loader.dataset)
        if loss_i < min_loss:
            min_loss = loss_i
            model = model.to("cpu")
            torch.save(model.state_dict(), "/home/shimizu/CV_project/model/572_4_epoch50/572_4_super_resolution.pth")
            print("save model")
            model = model.to(device)
        elif (i+1)%10 == 0:
            model = model.to("cpu")
            torch.save(model.state_dict(), "/home/shimizu/CV_project/model/572_4_epoch50/572_4_super_resolution{}.pth".format(str(i+1)))
            print("save model")
            model = model.to(device)
