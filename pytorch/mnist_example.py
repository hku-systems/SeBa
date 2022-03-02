import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import datasets
from torchvision import transforms
# import matplotlib.pyplot as plt
import torch.optim as optim
import argparse
import torch.nn.functional as F
import pickle
import random
import os

torch.manual_seed(0)

def default_loader(path):
    return Image.open(path).convert('RGB')

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        # 1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1conv -> 5x5conv branch
        # we use 2 3x3 conv filters stacked instead
        # of 1 5x5 filters to obtain the same receptive
        # field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3pooling -> 1x1conv
        # same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(1, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # although we only use 1 conv layer as prelayer,
        # we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        # """In general, an Inception network is a network consisting of
        # modules of the above type stacked upon each other, with occasional
        # max-pooling layers with stride 2 to halve the resolution of the
        # grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        # input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    #x: input image
    #attack == retrieve means extracting the activation in the middle
    #attack == infer means attacking
    #co_pre: dictionary, empty for "train" and "retrieve"
    #label for evaluating attack
    #output: original final result, activation: intermediate output, att_out: attack final output

    def forward(self,x):#co_pre: checked out previous layer, for retrieve, it should be empty
        layers = ["self.prelayer(x)", "self.a3(output)", "self.b3(output)",
                  "self.maxpool(output)","self.a4(output)","self.b4(output)",
                  "self.c4(output)","self.d4(output)","self.e4(output)",
                  "self.maxpool(output)","self.a5(output)","self.b5(output)",
                  "self.avgpool(output)","self.dropout(output)","output.view(output.size()[0], -1)",
                  "self.linear(output)"]
        for index, f in enumerate(layers,1):
            output = eval(f)
            # if index == break_layer:
            #     break
        return output


def train(train_loader,device,epoch, num_cli=10,meta=0.0): #num_cli is only used in path naming, the code logic is generic to num_cli
    model = GoogleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimier = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    plot_loss = {}
    iter = 1
    for e in range(epoch):
        print("training the " + str(e) + " epoch")
        for i, data in enumerate(train_loader, 0):#batch size is 40, automatically run one batch
            input, label = data
            input = input.to(device)
            label = label.to(device)
            optimier.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, label)
            loss.backward(retain_graph=False)
            optimier.step()
            iter = iter + 1
            plot_loss[iter] = loss.item()
            break
            # print("loss is " + str(loss.item()))
    # assert False
    paths = ["model/" + str(num_cli) + "_" + str(meta), "obj/"+ str(num_cli) + "_" + str(meta)]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
    torch.save(model.state_dict(), paths[0] + "/mnist")
    with open(paths[1] +"/mnist.pickle", 'wb') as handle: #store the loss values
        pickle.dump(plot_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test(test_loader,device,num_cli=0,meta=0.0):
    plot_loss = {}
    paths = ["model/" + str(num_cli) + "_" + str(meta), "obj/" + str(num_cli) + "_" + str(meta), "fig/"+ str(num_cli) + "_" + str(meta)]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
    # with open(paths[1] +"/mnist.pickle", 'rb') as handle: #retrieve the loss values
    #     plot_loss = pickle.load(handle)
    #     lists = sorted(plot_loss.items())  # sorted by key, return a list of tuples
    #     x, y = zip(*lists)  # unpack a list of pairs into two tuples
    #     plt.plot(x, y)
    #     plt.savefig(paths[2] +"/mnist.pdf")
    model = GoogleNet().to(device)
    iter_count = 0
    model.load_state_dict(torch.load(paths[0] +"/mnist"))
    model.eval()
    final_total = 0
    final_correct = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader,0):
            input,label = data
            input = input.to(device)
            label = label.to(device)
            outputs = model(input)
            predicted = torch.max(outputs.data, 1)
            final_total += label.size(0)
            # print("predicted : " + str(predicted[1]))
            # print("label : " + str(label))
            final_correct += (predicted[1] == label).sum().item()
        acc = final_correct/final_total*100
        print("final model's accracy is %.2f" % acc)

if __name__ =='__main__':
    transform = transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch', help='batch size')
    parser.add_argument('--mode', help='training or testing')
    parser.add_argument('--epoch', help='training or testing')
    args = parser.parse_args()
    batch_size = int(args.batch)
    epoch = int(args.epoch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_mnist = datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
    test_data_mnist = datasets.MNIST('./MNIST_data', train=False, transform=transform)
    train_loader = DataLoader(train_data_mnist, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data_mnist, shuffle=False, batch_size=batch_size)
    device = torch.device("cpu")
    if args.mode == "train":
        train(train_loader,device,epoch)
    elif args.mode == "test":
        test(test_loader,device)
    # f = open('/home/dong/pytorch/study/googlenet_acc_base_conv5.txt','a')
    # f.writelines(str(iteration)+','+str(test_)+'%\n')
    # f.close()
