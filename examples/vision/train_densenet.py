import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.datasets as datasets


def cifar_loaders(batch_size):
    train_set = datasets.CIFAR10('./data', train=True, download=False, transform=transforms.ToTensor())
    test_set = datasets.CIFAR10('./data', train=False, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


batch_size = 64
test_batch_size = 1000

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

# DenseNet in PyTorch.
# https://github.com/kuangliu/pytorch-cifar
class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # out = self.conv1(F.relu(x))
        # out = self.conv2(F.relu(out))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=True)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        # out_planes = int(math.floor(num_planes*reduction))
        # self.trans3 = Transition(num_planes, out_planes)
        # num_planes = out_planes

        # self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        # num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear1 = nn.Linear(14336, 512)
        self.linear2 = nn.Linear(512, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        # out = self.dense4(out)
        out = F.relu(self.bn(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        return out

# def DenseNet121():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)
#
# def DenseNet169():
#     return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)
#
# def DenseNet201():
#     return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)
#
# def DenseNet161():
#     return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)
#
# def Densenet_cifar():
#     return DenseNet(Bottleneck, [2,4,6], growth_rate=16)

def cifar_densenet(in_ch=3, in_dim=32):
    return DenseNet(Bottleneck, [2,4,4], growth_rate=32)


model = cifar_densenet()
# print(model)
num_epochs = 50


def train(model, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        batch_loss = 0
        total_batches = 0

        for i, (images, labels) in enumerate(train_loader):
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, 3, 32, 32)
            # print(images.shape)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

            total_batches += 1     
            batch_loss += loss.item()

        avg_loss_epoch = batch_loss / total_batches
        print('Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]'.format(epoch + 1, num_epochs, epoch + 1, avg_loss_epoch))

    torch.save(model.state_dict(), './pretrain/cifar_densenet.pth')


def accuracy_test(model_path, test_loader):
    correct = 0
    total = 0

    model = cifar_densenet()
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 3, 32, 32)
            # print(labels)
            outputs_test = model(images)
            _, predicted = torch.max(outputs_test.data, 1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 1000 test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    train(model, train_loader)
    accuracy_test('./pretrain/cifar_densenet', test_loader)
