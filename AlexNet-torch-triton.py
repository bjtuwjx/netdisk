import torch
import torch_mlu
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import time

import triton
import triton.language as tl

@triton.jit
def relu_kernel(x):
    """
    ReLU_ activation function

    .. _ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    zero = 0.0
    zero = zero.to(x.dtype)
    return tl.where(x >= 0, x, zero)

@triton.jit
def relu_grad_kernel(x):
    # ReLU is different from other activations
    # in that it does not require the input to retrospectively compute its gradient
    # here the input is the downstream gradient, and we return the upstream gradient directly
    one = 1.0
    one = one.to(x.dtype)
    zero = 0.0*one
    return tl.where(x >= 0, one, zero)

@triton.jit
def load_store_fwd(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # add to tl.int64 for large tensor
    # block_start = block_start.to(tl.int64)
    block_start = block_start.to(tl.int32)
    step_size = tl.num_programs(axis=0) * BLOCK_SIZE
    while 0 <= block_start and block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = relu_kernel(x)
        tl.store(output_ptr + offsets, output, mask=mask)
        block_start += step_size

@triton.jit
def load_store_bwd(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_start = block_start.to(tl.int32)
    step_size = tl.num_programs(axis=0) * BLOCK_SIZE
    while 0 <= block_start and block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = relu_grad_kernel(x)
        tl.store(output_ptr + offsets, output, mask=mask)
        block_start += step_size

class ApplyReLUTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        output = torch.empty_like(x)
        assert x.is_mlu
        assert x.is_contiguous()
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        load_store_fwd[grid](x, output, n_elements, BLOCK_SIZE=512)
        ctx.save_for_backward(x)
        return output
    @staticmethod
    def backward(ctx, dy):
        x = ctx.saved_tensors[0]
        assert x.is_mlu
        assert x.is_contiguous()
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        # load_store_bwd[grid](x, dy, n_elements, BLOCK_SIZE=512)
        # return dy
        grad = torch.randn_like(dy)
        load_store_bwd[grid](x, grad, n_elements, BLOCK_SIZE=512)
        return grad.mul(dy)

def apply_relu_triton(x: torch.Tensor):
    return ApplyReLUTriton.apply(x)

class ReLUTriton(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor):
        return apply_relu_triton(x)
relu_triton = ReLUTriton()


class AlexNet(nn.Module):
    def __init__(self, in_channels =1, num_classes=1000):
        super(AlexNet,self).__init__()

        self.c1=nn.Conv2d(in_channels=in_channels,out_channels=96,kernel_size=11,stride=4,padding=2)  
        # self.a1=nn.ReLU(inplace=True)
        self.a1=ReLUTriton()
        self.p1=nn.MaxPool2d(kernel_size=3,stride=2)   
        # self.l1 = nn.LocalResponseNorm(size=96, alpha=0.0001, beta=0.75, k=1.0)


        self.c2=nn.Conv2d(96,256,5,stride=1,padding=2)
        # self.a2=nn.ReLU(inplace=True)
        self.a2=ReLUTriton()
        self.p2=nn.MaxPool2d(kernel_size=3,stride=2)   
        # self.l2 = nn.LocalResponseNorm(size=256, alpha=0.0001, beta=0.75, k=1.0)


        self.c3=nn.Conv2d(256,384,3,stride=1,padding=1)
        # self.a3=nn.ReLU(inplace=True)
        self.a3=ReLUTriton()

        self.c4=nn.Conv2d(384,384,3,stride=1,padding=1)
        # self.a4=nn.ReLU(inplace=True)
        self.a4=ReLUTriton()

        self.c5=nn.Conv2d(384,256,3,stride=1,padding=1)  
        # self.a5=nn.ReLU(inplace=True)
        self.a5=ReLUTriton()
        self.p5 = nn.MaxPool2d(kernel_size=3, stride=2)  

        self.fc1_d = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*6*6,2048)
        # self.fc1_a = nn.ReLU(inplace=True)
        self.fc1_a=ReLUTriton()

        self.fc2_d=nn.Dropout(p=0.5)
        self.fc2=nn.Linear(2048,2048)
        # self.fc2_a=nn.ReLU(inplace=True)
        self.fc2_a=ReLUTriton()

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

        x = torch.flatten(x, start_dim=1)

        x = self.fc1_d(x)
        x = self.fc1(x)
        x = self.fc1_a(x)

        x=self.fc2_d(x)
        x = self.fc2(x)
        x = self.fc2_a(x)

        x=self.fc3(x)
        return  x

if __name__ == '__main__':

    time_start = time.time()

    device = torch.device('mlu' if torch.mlu.is_available() else 'cpu')

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

    print(model)

    n_epochs = 10
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
            X_train, y_train = X_train.mlu(), y_train.mlu()
            outputs = model(X_train)
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
            X_test, y_test = X_test.mlu(), y_test.mlu()
            outputs = model(X_test)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)

        print("Loss is: {:.4f}, Train Accuracy is: {:.4f}%, Test Accuracy is: {:.4f}%, Elapsed Time is: {:.2f} s".format((running_loss / len(trainset)),
                                                                                          (100*running_correct / len(trainset)),
                                                                                          (100*testing_correct / len(testset)),
                                                                                          time.time() - time_start))
    # torch.save(model.state_dict(), "model_parameter.pkl")
