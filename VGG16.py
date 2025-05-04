# 라이브러리 import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import os
import pandas as pd
import cv2 # 이미지 처리를 도와주는 라이브러리
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json
import h5py
from torch.utils.data import DataLoader

# 픽셀값들을 정규화 해줌
# 픽셀들은 0~255사이의 값을 가지기 때문에 값들간의 편차가 크다
# 따라서 값들간의 차이를 줄여 가중치의 영향을 줄여주기 위함
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터셋의 경로
PATH = 'Datasets/butterfly'
image_folder = []
csv_file = []

# image파일과 csv파일 분리
for path in os.listdir(PATH):
    if ".csv" in path:
        csv_file.append(path)
    else:
        image_folder.append(path)

file_path = dict()
print("image, csv파일 분리")

# train, test dataset 분리
# zip -> 반복문에서 인자로 두가지 변수를 한번에 사용하기 위해.
# join -> 인자들을 붙여준다.
for csv, folder in zip(csv_file, image_folder):
    if "Training" in csv:
        file_path["train_path"] = os.path.join(PATH, csv)
    else:
        file_path["test_path"] = os.path.join(PATH, csv)
    
    if folder == "train":
        file_path["train_img"] = os.path.join(PATH, folder)
    else:
        file_path["test_img"] = os.path.join(PATH, folder)

print("train, test dataset 분리")

# csv 파일 로드
train_csv = pd.read_csv(file_path['train_path'], encoding='utf-8')
test_csv = pd.read_csv(file_path['test_path'], encoding='utf-8')

train_img_path = []
test_img_path = []

for train_img in train_csv['filename']:
    train_img_path.append(os.path.join(file_path['train_img'], train_img))
    
for test_img in test_csv['filename']:
    test_img_path.append(os.path.join(file_path['test_img'], test_img))

print("train, test image 파일경로 저장")

train_data = []
test_data = []

# cv2(open-cv) 라이브러리를 통해 image파일을 읽어서 저장
for train_img in train_img_path:
    img = cv2.imread(train_img, cv2.IMREAD_COLOR_RGB)
    train_data.append(img)

for test_img in test_img_path:
    img = cv2.imread(test_img, cv2.IMREAD_COLOR_RGB)
    test_data.append(img)
    
train_data = np.array(train_data)
test_data = np.array(test_data)

print("image 로드?")


train_label = train_csv.iloc[:, -1].to_numpy()
le = LabelEncoder()
le_fit = le.fit(train_label)
train_classes = {num: name for num, name in enumerate(le_fit.classes_)}
train_label = le_fit.transform(train_label)

print("train label encoding")

# with h5py.File('butterfly_original.hdf5', 'w') as hf:
#     hf.create_dataset("train_data", data=train_data)
#     hf.create_dataset("train_label", data=train_label)
#     hf.create_dataset("test_data", data=test_data)
    
# with open('butterfly_original_classes.json', 'w') as j:
#     json.dump(train_classes, j)

with h5py.File('butterfly_original.hdf5', 'r') as hf:
    train_data = hf['train_data'][:]
    train_label = hf['train_label'][:]
    test_data = hf['test_data'][:]

with open("butterfly_original_classes.json", 'r') as j:
    train_classes = json.load(j)

print("저장된 데이터셋 로드")

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16,self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 224 -> 112

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 112 -> 56

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 56 -> 28
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 28 -> 14

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 14 -> 7

            nn.Flatten(), # Flatten은 layer로 안쳐준다.
            nn.Dropout(.5),
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000)
        )
    def forward(self, x):
        return self.convnet(x)

model = VGG_16()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=1e-2, momentum=.9, weight_decay=5e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader) # 사실 없어도 됨 마지막에 가독성을 좋게하기위해 사용
    model.train()   # torch안의 module안에 train이라는 메소드가 있다.
    
    for batch, data in enumerate(dataloader):
        image, label = data

        # Compute prediction error 
        pred = model(image) # model.forward(image)
        # softmax = nn.Softmax(dim=1)
        # pred_probab = softmax(pred)
        loss = loss_fn(pred, label)

        # backpropagation
        loss.backward() # 쓸모없는 노드를 지운다. # 매개변수를 안정해주면 랜덤으로 잡고 경사하강을 한다.
        optimizer.step() #learning rate 만큼 내려간다.
        optimizer.zero_grad() #초기화해주는데 해주는 이유는 이전 스텝을 기억하고있으면 다음 스텝에 영향을 줄 수 있기때문에 초기화한다.

        if not(batch % 100):
            loss, current = loss.item(), (batch + 1) * len(image)
            print(f"loss : {loss:>7f}   [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #평가를 해주겠다 명시
    test_loss, correct = 0, 0
    
    with torch.no_grad(): #no_grad -> 최적화를 수행하지 않겠다.
        for image, label in dataloader:
            pred = model(image)
            # softmax = nn.Softmax(dim=1)
            # pred_probab = softmax(pred)
            test_loss += loss_fn(pred, label).item() #.item() -> loss_fn에 있는 데이터값을 불러준다.
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
                            #argmax --> 예측값중에 가장 큰 값을 뽑아오는 것
    
    test_loss /= num_batches
    correct /=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")\
    
epochs = 30
for i in range(epochs):
    print(f"Epoch {i + 1}\n------------------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done")