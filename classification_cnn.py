from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hn_supervise import spectral_norm
import torchvision
from torch.autograd import Variable, grad
import math
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import numpy as np
from delta_orthogonal import conv_delta_orthogonal_
# torch.set_default_dtype(torch.float64)

DATASET = 'mnist'
batch_size_train = 200
batch_size_test = 1000
LEARNING_RATE = 1e-4
MOMENTUM = 0.99
DEPTH = 100
CHANNEL = 128
INIT_METHOD = 'default'
PC_LEVEL = 2

file = open("LOGS/{}_depth{}_channel{}_init{}_batchsize{}_lr{}_momentum{}_withPC{}.txt".format(
    DATASET, DEPTH, CHANNEL, INIT_METHOD, batch_size_train, LEARNING_RATE, MOMENTUM, PC_LEVEL),"w+") 

class CircularPadding(nn.Module):
    def __init__(self):
    	super(CircularPadding, self).__init__()

    def forward(self, x):
    	# creating the circular padding
    	return F.pad(x, (1, 1, 1, 1), mode='circular')


class DeepCNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(DeepCNN, self).__init__()
        self.DEPTH = DEPTH # number of layers.
        self.C_SIZE = 128 # channel size.
        self.K_SIZE = 3 # kernel size
        self.phi = nn.Tanh() # non-linearity

        modules = []
        # modules.append(spectral_norm(nn.Conv2d(in_channels=1, out_channels=self.C_SIZE, kernel_size=3, padding=1),
        #                                             use_adaptivePC=False, pclevel=2))
        modules.append(nn.Conv2d(in_channels=1, out_channels=self.C_SIZE, kernel_size=3, padding=1))
        modules.append(self.phi)

        for _ in range(2):
            modules.append(spectral_norm(nn.Conv2d(in_channels=self.C_SIZE, out_channels=self.C_SIZE, kernel_size=3, padding=1, stride=2),
                                                    use_adaptivePC=False, pclevel=PC_LEVEL))
            modules.append(self.phi)

        for _ in range(self.DEPTH):
            # modules.append(CircularPadding)
            modules.append(spectral_norm(nn.Conv2d(in_channels=self.C_SIZE, out_channels=self.C_SIZE, kernel_size=3, padding=1, padding_mode='zeros'),
                                                    use_adaptivePC=False, pclevel=PC_LEVEL))
            modules.append(self.phi)
        
        self.conv = nn.Sequential(*modules)
        self.fc = spectral_norm(nn.Linear(self.C_SIZE, 10), use_adaptivePC=False, pclevel=PC_LEVEL)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2,3])
        x = self.fc(x)
        return x

# class DeepCNN(nn.Module):
#     """CNN."""

#     def __init__(self):
#         """CNN Builder."""
#         super(DeepCNN, self).__init__()
#         self.DEPTH = DEPTH # number of layers.
#         self.C_SIZE = CHANNEL # channel size.
#         self.K_SIZE = 3 # kernel size
#         self.phi = nn.Tanh() # non-linearity

#         modules = []
#         modules.append(('Conv0', nn.Conv2d(in_channels=1, out_channels=self.C_SIZE, kernel_size=3, padding=1)))
#         modules.append(('acti0', self.phi))

#         for d in range(2):
#             modules.append(('Conv{}'.format(d+1), nn.Conv2d(in_channels=self.C_SIZE, out_channels=self.C_SIZE, kernel_size=3, padding=1, stride=2)))
#             modules.append(('acti{}'.format(d+1), self.phi))

#         for d in range(self.DEPTH):
#             # modules.append(CircularPadding)
#             modules.append(('Conv{}'.format(d+3), nn.Conv2d(in_channels=self.C_SIZE, out_channels=self.C_SIZE, kernel_size=3, padding=1)))
#             modules.append(('acti{}'.format(d+3), self.phi))
        
#         # self.conv = nn.Sequential(*modules)
#         self.conv = nn.Sequential(OrderedDict(modules))
#         self.fc = nn.Linear(self.C_SIZE, 10)

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.mean([2,3])
#         x = self.fc(x)
#         return x

def init_ortho_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.orthogonal_(m.weight) # may need to scale up
        init.normal_(m.bias, std=math.sqrt(1e-5))

W_VAR, B_VAR = 1.05, 2.01e-5
def init_delta_ortho(m):
    if type(m) == nn.Conv2d:
        if m.weight.shape[0] == m.weight.shape[1] and m.stride == 1:
            conv_delta_orthogonal_(m.weight, gain=np.sqrt(W_VAR))

        elif m.weight.shape[0] == m.weight.shape[1]:
            std = np.sqrt(W_VAR / (3**2 * 128.))
            nn.init.normal_(m.weight, std=std)

        else:
            std = np.sqrt(W_VAR / (3**2 * 1.))
            nn.init.normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=np.sqrt(B_VAR))
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=np.sqrt(W_VAR/128.))
        if m.bias is not None:
            nn.init.normal_(m.bias, std=B_VAR)

def calculate_accuracy(loader, network):
    with torch.no_grad():
        for data, target in loader:
            data = data.cuda()
            target = target.cuda()
            output = F.log_softmax(network(data))
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum()
            break
    return 1.* correct / target.shape[0]

def getGradNorm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
    total_norm = total_norm ** (1. / 2)
    return total_norm

def getConditionNumber(model):
    cns = []
    for name, p in model.named_parameters():
        if 'weight_ori' in name:
            weight = p.data
            weight_mat = weight.reshape(weight.shape[0], -1)

            # n, m = weight_mat.shape
            # I = torch.eye(n).cuda()
            # wwt = weight_mat.mm(weight_mat.t())
            # # weight_mat = (2.083 * I + wwt.mm(-1.643 * I + 0.560 * wwt)).mm(weight_mat)
            # weight_mat = (3.625 * I + wwt.mm(-9.261 * I + wwt.mm(14.097 * I + wwt.mm(-10.351 * I + 2.890 * wwt)))).mm(
            #     weight_mat)

            S = torch.svd(weight_mat)[1]
            sin_num = max(1, int(S.shape[0] * 0.1))
            condition_number = S[0] / (S[-sin_num:]).mean()
            cns.append(condition_number.item())
            # break
    return cns, np.cumprod(cns)[-1]


trainloader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/home/illini/rsgan/DATA/mnist/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                            #    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

testloader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/home/illini/rsgan/DATA/mnist/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                            #    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

# trainloader = torch.utils.data.DataLoader(
#   torchvision.datasets.CIFAR10('/home/illini/rsgan/DATA/cifar10/', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                              ])),
#   batch_size=batch_size_train, shuffle=True)

# testloader = torch.utils.data.DataLoader(
#   torchvision.datasets.CIFAR10('/home/illini/rsgan/DATA/cifar10/', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                              ])),
#   batch_size=batch_size_test, shuffle=True)


net = DeepCNN()
# print(net)
if INIT_METHOD == 'deltaortho':
    net.apply(init_delta_ortho)
elif INIT_METHOD == 'ortho':
    net.apply(init_ortho_weights)
elif INIT_METHOD == 'default':
    pass

net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader))

for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs[inputs < 0.5] = 0.
        inputs[inputs >= 0.5] = 1.
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Normalizing the loss by the total number of train batches
    running_loss /= len(trainloader)

    # Calculate training/test set accuracy of the existing model
    train_accuracy = calculate_accuracy(trainloader, net)
    test_accuracy = calculate_accuracy(testloader, net)

    gradSum = getGradNorm(net)

    msg = "Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}% | Gradient Sum: {4}".format(epoch+1, running_loss, train_accuracy*100., test_accuracy*100., gradSum)
    print(msg)
    file.write(msg+'\n')
    cns, cnprod = getConditionNumber(net)
    # msg = "condition numbers {}".format(cns)
    # print(msg)
    # file.write(msg+'\n')
    msg = "accumlative condition numbers {}".format(cnprod)
    print(msg)
    file.write(msg+'\n')

    scheduler.step()



    # # save model
    # if epoch % 50 == 0:
    #     print('==> Saving model ...')
    #     state = {
    #         'net': net.module if opt.is_gpu else net,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, '../checkpoint/ckpt.t7')

print('==> Finished Training ...')
file.close()

# m = DeepCNN().cuda()
# x = torch.randn(3,3,32,32).cuda()
# y = m(x)
# print(y.shape)


# '''
# Creating circular padding
# '''
# def __init__(self):
# 	super(padding_cir, self).__init__()

# def forward(self, x):
# 	# creating the circular padding
# 	return F.pad(x, (1, 1, 1, 1), mode='circular')`
# and then when you create you list of layers you just dont have padding for you conv2d
# conv2d = nn.Conv2d(in_channels, v, kernel_size=3, bias=False)
# layers = [padding_cir(), conv2d, self.activation]
# nn.Sequential(*layers)



# C_SIZE = 10 # channel size.
# K_SIZE = 3
# w = nn.Conv2d(1, C_SIZE, K_SIZE)
# makeDeltaOrthogonal(w.weight.data, 1)
# print(w.weight)