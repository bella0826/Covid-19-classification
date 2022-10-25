from torch.autograd import Variable
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms, datasets

root_path = "./Covid/"
EPOCHS = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop((224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder(root_path, transform=data_transform)

train_size = int(0.8 * len(train_data))
valid_size = len(train_data) - train_size

train, valid = torch.utils.data.random_split(dataset = train_data, lengths = [train_size, valid_size])
train_loader = DataLoader(dataset = train, batch_size = 32, shuffle = True, num_workers = 2)
valid_loader = DataLoader(dataset = valid, batch_size = 32, shuffle = True, num_workers = 2)

dataset_sizes = len(train)
val_sizes = len(valid)

model = models.resnet50(pretrained=True)
for par in model.parameters():
    par.requires_grad = False
model_fc = model.fc.in_features
model.fc = nn.Linear(model_fc,4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def fit(model,train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_acc_his = []
    train_loss_his = []
    val_acc_his = []
    val_loss_his = []
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            
            if phase == "train":
                for inputs, labels in train_loader:
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()  
                            optimizer.step() 

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            else:
                for inputs, labels in val_loader:
        
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                epoch_loss = running_loss / dataset_sizes
                epoch_acc = running_corrects.double() / dataset_sizes
                train_acc_his.append(epoch_acc)
                train_loss_his.append(epoch_loss)
            else:
                epoch_loss = running_loss / val_sizes
                epoch_acc = running_corrects.double() / val_sizes
                val_acc_his.append(epoch_acc)
                val_loss_his.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    return train_acc_his, train_loss_his, val_acc_his, val_loss_his

train_acc_his, train_loss_his, val_acc_his, val_loss_his = fit(model, train_loader, valid_loader, criterion, optimizer, EPOCHS)

'''plt.plot(range(EPOCHS), train_acc_his, "b-", label="Training ACC")
plt.plot(range(EPOCHS), val_acc_his, "g-", label="Validation ACC")
plt.title("Training & Validation Accuracy")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./acc.jpg/")
'''



