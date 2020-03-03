import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.weight_norm import WeightNorm, weight_norm
import torch.nn.functional as F
from torch.autograd import Variable, Function
import torch

from torchvision import datasets, models, transforms

class ResBase(nn.Module):

    def __init__(self, option = 'resnet18', pret =True):
        super(ResBase, self).__init__()

        self.dim = 2048

        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)

        mod = list(model_ft.children())
        #print(mod.pop())
        mod.pop() #  제일 마지막 layer 사라짐
        self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        print(x.size)
        x = x.view(x.size(0), self.dim) #features에서 나온 것을 2048 dimension으로 변형, 아마 원래도 2048 일 것
        return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=13, num_layer = 2, num_unit =2048, prob = 0.5, middle = 1000):
        super(ResClassifier,self).__init__()

        layers = []
        # 2048 -> dropout -> FC(1000) -> BN -> Relu : 기본으로 무조건 존재
        #      -> dropout -> FC(1000) -> BN -> Relu : layer 1
        #      -> dropout -> FC(1000) -> BN -> Relu : layer 2
        # -> FC(class수) -> output
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer - 1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle, middle))
            layers.append(nn.BatchNorm1d(middle, affine=True))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle,num_classes))

        self.classifier = nn.Sequential(*layers)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse = False): # False여서 GRL 사용 안함
        if reverse:
            x = grad_reverse(x, self.lambd)

        x = self.classifier(x)
        return x


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self,x):
        return x.view_as(x)

    def backward(self,grad_output):
        return (grad_output*-self.lambd)