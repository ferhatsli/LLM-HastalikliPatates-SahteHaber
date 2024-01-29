import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import timm
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

root = "archive/"
a = os.listdir(root)
dic = {}
csv_file = pd.DataFrame()
for disease in a:
    sub_path =  os.path.join(root,disease)
    for img in os.listdir(sub_path):
        dict_a = {'img': ['0'], 'label': ['0']}
        temp = pd.DataFrame(dict_a)

        temp.iloc[0,0] = os.path.join(sub_path,img)
        temp.iloc[0,1] = disease
        csv_file = pd.concat([csv_file,temp])

csv_file

dic = {}
for item in tqdm(set(csv_file["label"])):
    dic[item] = len(dic)
dic

class PatchDataset(Dataset):
    def __init__(self, dic_map,file, mode = "train"):
        self.mode = mode
#         self.root = root
        if self.mode == "train":
            print("Train")
            self.file = file
            self.transformer = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        else:
            print("Test")
            self.file = file
            self.transformer = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

        self.dic_map = dic_map


    def __getitem__(self, item):
        path = self.file.iloc[item,0]
#         path = os.path.join(self.root+path)
        label = self.dic_map[self.file.iloc[item,1]]
        img = Image.open(path).convert('RGB')
        img = self.transformer(img)
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.file)

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(csv_file, test_size=0.2, random_state=42)
dataset1 = PatchDataset(dic, train_df)
dataset2 = PatchDataset(dic, test_df, mode="test")

trainset = torch.utils.data.DataLoader(dataset1, batch_size=16, shuffle=True)
testset = torch.utils.data.DataLoader(dataset2, batch_size=16, shuffle=True)

# len(testset)
len(trainset)

if torch.cuda.is_available():
    print("GPU is available!")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
timm.list_models()

net = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=7).to(device)
optimizer = optim.AdamW(net.parameters(), lr=1e-5)

for epoch in range(10):
    net.train()
    loss_batch = 0
    for data, target in tqdm(trainset):
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = criterion(output, target)
        loss_batch+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("The {} epoch Loss is {}".format(epoch,loss_batch/len(trainset)))
    net.eval()
    correct = 0
    total = 0
    best_accuracy = 0.0
    with torch.no_grad():
        for data,target in tqdm(testset):
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            #print(correct)
    accuracy = correct / len(testset.dataset)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(net.state_dict(),"best_model.pth")
        print("Saved model with accuracy: {:.2f}".format(best_accuracy))
    print("The {} epoch accuracy is {}".format(epoch,correct/len(testset.dataset)))
