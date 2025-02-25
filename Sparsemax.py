import torch
from torch import nn
import matplotlib.pyplot as plt
import math
import seaborn as sns
import torch.nn.functional as F
from torch.autograd import Function

class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = dim
        input = input.transpose(0, dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range_tensor = torch.arange(1, number_of_logits + 1, dtype=input.dtype, device=input.device).view(1, -1)
        range_tensor = range_tensor.expand_as(zs)

        bound = 1 + range_tensor * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range_tensor, dim, keepdim=True)[0]

        zs_sparse = is_gt * zs

        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        output = torch.max(torch.zeros_like(input), input - taus)

        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, ctx.dim)

        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = ctx.dim

        nonzeros = torch.ne(output, 0)
        sum_grad = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        grad_input = nonzeros * (grad_output - sum_grad.expand_as(grad_output))

        return grad_input, None

class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        return SparsemaxFunction.apply(input, self.dim)
