import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
# import tqdm
from torch.optim import lr_scheduler
import time

# from torchviz import make_dot

class AlexNet(nn.Module):
    def __init__(self, in_channels =1, num_classes=1000):
        super(AlexNet,self).__init__()

        self.c1=nn.Conv2d(in_channels=in_channels,out_channels=96,kernel_size=11,stride=4,padding=2)  
        self.a1=nn.ReLU(inplace=True)
        self.p1=nn.MaxPool2d(kernel_size=3,stride=2)   
        # self.l1 = nn.LocalResponseNorm(size=96, alpha=0.0001, beta=0.75, k=1.0)


        self.c2=nn.Conv2d(96,256,5,stride=1,padding=2)
        self.a2=nn.ReLU(inplace=True)
        self.p2=nn.MaxPool2d(kernel_size=3,stride=2)   
        # self.l2 = nn.LocalResponseNorm(size=256, alpha=0.0001, beta=0.75, k=1.0)


        self.c3=nn.Conv2d(256,384,3,stride=1,padding=1)
        self.a3=nn.ReLU(inplace=True)

        self.c4=nn.Conv2d(384,384,3,stride=1,padding=1)
        self.a4=nn.ReLU(inplace=True)

        self.c5=nn.Conv2d(384,256,3,stride=1,padding=1)  
        self.a5=nn.ReLU(inplace=True)
        self.p5 = nn.MaxPool2d(kernel_size=3, stride=2)  

        self.fc1_d=nn.Dropout(p=0.5)
        self.fc1=nn.Linear(256*6*6,2048)
        self.fc1_a=nn.ReLU(inplace=True)

        self.fc2_d=nn.Dropout(p=0.5)
        self.fc2=nn.Linear(2048,2048)
        self.fc2_a=nn.ReLU(inplace=True)

        self.fc3=nn.Linear(2048,num_classes)

    def forward(self,x):
        x = self.c1(x)
        x = self.a1(x)
        x = self.p1(x)
        # x = self.l1(x)

        x = self.c2(x)
        x = self.a2(x)
        x = self.p2(x)
        # x = self.l2(x)

        x = self.c3(x)
        x = self.a3(x)

        x = self.c4(x)
        x = self.a4(x)

        x = self.c5(x)
        x = self.a5(x)
        x = self.p5(x)

        x = torch.flatten(x,start_dim=1)

        x = self.fc1_d(x)
        x = self.fc1(x)
        x = self.fc1_a(x)

        x = self.fc2_d(x)
        x = self.fc2(x)
        x = self.fc2_a(x)

        x = self.fc3(x)
        return  x




if __name__ == '__main__':
    
    time_start = time.time()

    device = torch.device('mlu:0' if torch.cuda.is_available() else 'cpu')

    batchSize = 64  
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    data_transform =  transforms.Compose([
									     transforms.Resize((224,224)),
									     transforms.ToTensor(),
									     normalize])
									 
    trainset = torchvision.datasets.CIFAR10(root='./Cifar-10',
 										    train=True, download=True, transform=data_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./Cifar-10', 
										    train=False, download=True, transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)

  
    model = AlexNet(in_channels = 3, num_classes = 10).to(device)

    n_epochs = 40
    num_classes = 10
    learning_rate = 0.0001
    momentum = 0.9 

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("-"*10)
       
        running_loss = 0.0
        running_correct = 0
        for data in trainloader:
            X_train, y_train = data
            X_train, y_train = X_train.to(device), y_train.to(device)
            outputs = model(X_train)
            
            # make_dot(outputs, params=dict(list(model.named_parameters()))).render("model", format="png")
            
            loss = criterion(outputs, y_train)
            _,pred = torch.max(outputs.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
            running_correct += torch.sum(pred == y_train.data)


        testing_correct = 0
        for data in testloader:
            X_test, y_test = data
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)

        print("Loss is: {:.4f}, Train Accuracy is: {:.4f}%, Test Accuracy is: {:.4f}%, Elapsed Time is: {:.2f} s".format(torch.true_divide(running_loss, len(trainset)),
                                                                                          torch.true_divide(100*running_correct, len(trainset)),
                                                                                          torch.true_divide(100*testing_correct, len(testset)),
                                                                                          time.time() - time_start))
    torch.save(model.state_dict(), "model_parameter.pkl")