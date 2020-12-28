from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import Higham_norm
import torchvision
from torch.autograd import Variable, grad
import math
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import numpy as np
# torch.set_default_dtype(torch.float64)

def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q

def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows < cols:
        print("In_filters : {}; Out_filters : {}".format(rows, cols))
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    weights[:, :, mid1, mid2] = q[:weights.size(0), :weights.size(1)]
    weights.mul_(gain)

def conv_delta_orthogonal_(tensor, gain=1.):
    r"""Initializer that generates a delta orthogonal kernel for ConvNets.
    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero. See
    algorithm 2 in [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`3 \leq n \leq 5`
        gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
    Examples:
        >>> w = torch.empty(5, 4, 3, 3)
        >>> nn.init.conv_delta_orthogonal_(w)
    """
    if tensor.ndimension() < 3 or tensor.ndimension() > 5:
      raise ValueError("The tensor to initialize must be at least "
                       "three-dimensional and at most five-dimensional")
    
    if tensor.size(1) > tensor.size(0):
      raise ValueError("In_channels cannot be greater than out_channels.")
    
    # Generate a random matrix
    a = tensor.new(tensor.size(0), tensor.size(0)).normal_(0, 1)
    # Compute the qr factorization
    q, r = torch.qr(a)
    # Make Q uniform
    d = torch.diag(r, 0)
    q *= d.sign()
    q = q[:, :tensor.size(1)]
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndimension() == 3:
            tensor[:, :, (tensor.size(2)-1)//2] = q
        elif tensor.ndimension() == 4:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2] = q
        else:
            tensor[:, :, (tensor.size(2)-1)//2, (tensor.size(3)-1)//2, (tensor.size(4)-1)//2] = q
        tensor.mul_(math.sqrt(gain))
    return tensor


class CircularPadding(nn.Module):
    def __init__(self):
    	super(CircularPadding, self).__init__()

    def forward(self, x):
    	# creating the circular padding
    	return F.pad(x, (1, 1, 1, 1), mode='circular')


# class DeepCNN(nn.Module):
#     """CNN."""

#     def __init__(self):
#         """CNN Builder."""
#         super(DeepCNN, self).__init__()
#         self.DEPTH = 500 # number of layers.
#         self.C_SIZE = 128 # channel size.
#         self.K_SIZE = 3 # kernel size
#         self.phi = nn.Tanh() # non-linearity

#         modules = []
#         modules.append(Higham_norm.spectral_norm(nn.Conv2d(in_channels=1, out_channels=self.C_SIZE, kernel_size=3, padding=1),
#                                                     use_adaptivePC=False, pclevel=0))
#         modules.append(self.phi)

#         for _ in range(2):
#             modules.append(Higham_norm.spectral_norm(nn.Conv2d(in_channels=self.C_SIZE, out_channels=self.C_SIZE, kernel_size=3, padding=1, stride=2),
#                                                     use_adaptivePC=False, pclevel=0))
#             modules.append(self.phi)

#         for _ in range(self.DEPTH):
#             # modules.append(CircularPadding)
#             modules.append(Higham_norm.spectral_norm(nn.Conv2d(in_channels=self.C_SIZE, out_channels=self.C_SIZE, kernel_size=3, padding=1, padding_mode='zeros'),
#                                                     use_adaptivePC=False, pclevel=0))
#             modules.append(self.phi)
        
#         self.conv = nn.Sequential(*modules)
#         self.fc = Higham_norm.spectral_norm(nn.Linear(self.C_SIZE, 10), use_adaptivePC=False, pclevel=0)

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.mean([2,3])
#         x = self.fc(x)
#         return x

class DeepCNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(DeepCNN, self).__init__()
        self.DEPTH = 100 # number of layers.
        self.C_SIZE = 128 # channel size.
        self.K_SIZE = 3 # kernel size
        self.phi = nn.Tanh() # non-linearity

        modules = []
        modules.append(('Conv0', nn.Conv2d(in_channels=1, out_channels=self.C_SIZE, kernel_size=3, padding=1)))
        modules.append(('acti0', self.phi))

        for d in range(2):
            modules.append(('Conv{}'.format(d+1), nn.Conv2d(in_channels=self.C_SIZE, out_channels=self.C_SIZE, kernel_size=3, padding=1, stride=2)))
            modules.append(('acti{}'.format(d+1), self.phi))

        for d in range(self.DEPTH):
            # modules.append(CircularPadding)
            modules.append(('Conv{}'.format(d+3), nn.Conv2d(in_channels=self.C_SIZE, out_channels=self.C_SIZE, kernel_size=3, padding=1)))
            modules.append(('acti{}'.format(d+3), self.phi))
        
        # self.conv = nn.Sequential(*modules)
        self.conv = nn.Sequential(OrderedDict(modules))
        self.fc = nn.Linear(self.C_SIZE, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2,3])
        x = self.fc(x)
        return x


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out = F.relu(out)
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])

# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])

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

batch_size_train = 200
batch_size_test = 1000
LEARNING_RATE = 1e-3
MOMENTUM = 0.95

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
# net = ResNet50()
# net = ResNet101()
net.apply(init_delta_ortho)
# net.apply(init_ortho_weights)
net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

for epoch in range(1000):
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

    print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(epoch+1, running_loss, train_accuracy, test_accuracy))

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