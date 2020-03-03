from torch import FloatTensor
from torch.autograd import Variable, Function
import torch.nn as nn



class GradReverse(Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -lambd)

def grad_reverse(x):
    return GradReverse()(x)

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        #eights = [Variable(FloatTensor([i]), requires_grad=True) \
                   #for i in (2, 5)]
        self.w1 = nn.Parameter(Variable(FloatTensor([2]), requires_grad=True))
        self.w2 = nn.Parameter(Variable(FloatTensor([5]), requires_grad=True))

        #self.f1 = w1 * x
        #self.f2 = w2 * x
        #self.L = L = ( 10 - self.f2 ) + 2*self.f1

    def forward(self, x):
        #x = grad_reverse(x)
        f1 = self.w1 * x # 8
        f2 = self.w2 * f1 # 40
        L = ( 10 - f2 ) + 2 * f1  # ( 10 - 40 ) + 2 * 8 = -30+16 = -14
        #x = self.f1(x)
        #x = self.f2(x)
        #x = self.L(x)
        return L

net = network()
#print(net)
params = list(net.parameters())
#print(params)
input = Variable(FloatTensor([4]))
out = net(input)
print(out)
net.zero_grad()
out.backward()

for index, weight in enumerate(params, start=1):
    gradient, *_ = weight.grad.data
    print("Gradient of w{} to L : {}".format(index, gradient))






#x = Variable(FloatTensor([4]))

#model = network()

#model_out = model(x)
#print(model_out)

#model_out.zero_grad()

#model_out.backward()



#for index, weight in enumerate(weights, start=1):
    #gradient, *_ = weight.grad.data
    #print("Gradient of w{} to L : {}".format(index, gradient))

'''
weights = [Variable(FloatTensor([i]), requires_grad=True)\
           for i in (2,5)]

#print(weights)

w1, w2 = weights

f1 = w1 * x
GradReverse()
f2 = w2 * x

L = ( 10 - f2 ) + 2*f1

L.register_hook(lambda grad : print(grad))
f2.register_hook(lambda grad : print(grad))
f1.register_hook(lambda grad : print(grad))

L.backward()

for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print("Gradient of w{} to L : {}".format(index, gradient))

'''