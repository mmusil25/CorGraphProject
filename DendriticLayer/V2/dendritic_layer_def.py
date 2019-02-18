"""
Name: Mark Musil
Date: November 21, 2018

Project: Dendritic layer honor's thesis

Description:

This is a module definition for Pytorch which implements the dendritic layer. 

reference: 

https://pytorch.org/docs/stable/notes/extending.html

Notes: This layer definition was not written to be scaled and until that is done
a few practices need to be followed:

1. input_features must be set equal to 1568 = (7*7*32), output_features = 10

"""

import torch
import torch.nn as nn
import numpy


def multi_variate_sigmoid(x):  # Here x is a numpy array
    return (1 + numpy.exp(numpy.sum(x))) ** (-1)


def dendritic_boundary(x):  # Here x is a single valued real number
    alpha_L, alpha_U = 0.5, 0.5
    b_U, b_L = 0, 1
    numerator = (1 + numpy.exp(alpha_L * (x - b_L))) ** (alpha_L ** (-1))
    denominator = (1 + numpy.exp(alpha_U * (x - b_U))) ** (alpha_U ** (-1))
    return numpy.log(numerator / denominator) + b_L


def dendritic_transfer(x):  # Here x is a numpy array
    a_d, c_d, b_d = 1, 0.5, 0.5 
    arg1 = c_d * multi_variate_sigmoid(numpy.multiply(a_d, numpy.subtract(x, b_d)) + numpy.sum(x))
    return dendritic_boundary(arg1)


# class Dendritic(Function):
#     def forward(ctx, activations, weight, dendrites, bias=None):
#         ctx.save_for_backward(input, weight, bias)
#         activations_np = activations.numpy()
#         weight_np = weight.numpy()
#         if bias is not None:
#             bias_np = bias.numpy()
#
#         dendrites_np = dendrites.numpy()
#         soma_input = np.zeros(dendrites)
#         output = np.zeros(output_features)
#
#         for n in range(output_features):
#             for d in range(dendrites):
#                 soma_input[d] = np.dot(activations[d:d+dendrites+1],weight[n:,d:])
#             output[n] = dendritic_transfer(soma_input)
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return torch.tensor(output)
#
#
#
#     def backward(ctx, grad_output):
#         # This is a pattern that is very convenient - at the top of backward
#         # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#         # None. Thanks to the fact that additional trailing Nones are
#         # ignored, the return statement is simple even when the function has
#         # optional inputs.
#         input, weight, bias = ctx.saved_tensors
#         grad_input = grad_weight = grad_bias = None
#
#         # These needs_input_grad checks are optional and there only to
#         # improve efficiency. If you want to make your code simpler, you can
#         # skip them. Returning gradients for inputs that don't require it is
#         # not an error.
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.mm(weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.t().mm(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0).squeeze(0)
#
#         return grad_input, grad_weight, grad_bias
#
# linear = LinearFunction.apply
    
# from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
# input = (torch.randn(20,20,dtype=torch.double,requires_grad=True),
#          torch.randn(30,20,dtype=torch.double,requires_grad=True))
# test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
# print(test)

class Dendritic(nn.Module):
    def __init__(self, input_features, output_features, dendrites, den_view, batch_size, bias=True):
        super(Dendritic, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.dendrites = dendrites
        self.Den_view = den_view # Number if input neurons each dendrite can see.
        self.batch_size = batch_size

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        
        self.weight = nn.Parameter(torch.Tensor(output_features, dendrites, int(input_features/dendrites)))

        # TODO: Add input protection for the final dimension so that
        # this layer can handle any input size

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input_set):
        input_set = input_set.detach().numpy()
        weight_np = self.weight.detach().numpy()
        if self.bias is not None:
            bias_np = self.bias.detach().numpy()
        soma_input = numpy.zeros(self.dendrites)
        output = numpy.zeros( (int(self.batch_size), int(self.output_features)))
        for i in range(self.batch_size):
            for n in range(self.output_features):
                for d in range(self.dendrites):
                    soma_input[d] = numpy.dot(input_set[i,d:d + self.Den_view], weight_np[n,d].reshape(49,1))
                output[i, n] = dendritic_transfer(soma_input)
                if self.bias is not None:
                    output[i] += bias_np
        return torch.tensor(output)
   