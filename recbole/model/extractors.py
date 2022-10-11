import torch
import torch.nn as nn
import torch.nn.functional as fn
import math
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": fn.relu, "swish": swish}


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# Augmenter
class Extractor(nn.Module):
    def __init__(self,layers, activation='gelu', init_method=None):
        super(Extractor, self).__init__()
        self.dense_1=nn.Linear(layers[0],layers[1])
        self.dense_2=nn.Linear(layers[1],layers[2])
        self.dense_3=nn.Linear(layers[2],layers[3])
        if activation!=None:
            self.actfunction = ACT2FN[activation]
        self.normal=Normalize()
    def forward(self,input): # [B H]
        # layer 1
        output=self.dense_1(input)
        if self.actfunction!=None:
            output=self.actfunction(output)
        # layer 2
        output=self.dense_2(output)
        if self.actfunction != None:
            output = self.actfunction(output)
        # layer 3
        output=self.dense_3(output)
        output=self.normal(output)
        return output
