import math
import torch as pt
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from groupy.gconv.make_gconv_indices import *


class SplitGConv2d(nn.Module):
    """
    Group equivariant convolution layer.
    
    :parm g_input: One of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
        The parameter value 'Z2' specifies the data being convolved is from the Z^2 plane (discrete mesh).
    :parm g_output: One of ('C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
        The value of g_input of the subsequent layer should match the value of g_output from the previous.
    :parm in_channels: The number of input channels. Based on the input group action the number of channels 
        used is equal to nti*in_channels.
    :parm out_channels: The number of output channels. Based on the output group action the number of channels
        used is equal to nto*out_channels.
    """

    def __init__(self, 
                g_input, 
                g_output, 
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1,
                padding=0, 
                bias=True) -> None:
        
        super(SplitGConv2d, self).__init__()

        # Transform kernel size argument 
        self.ksize = kernel_size
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.g_input = g_input
        self.g_output = g_output

        # Convert g_input, g_output to integer keys
        # sets values for nit, nto paramters
        self.nti, self.nto, self.inds = self.make_filter_indices()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Construct convolution kernel weights 
        self.weight = Parameter(pt.Tensor(out_channels, self.in_channels, self.nti, *kernel_size))
        if bias:
            self.bias = Parameter(pt.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Initialize convolution kernel weights
        init.xavier_normal_(self.weight)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_filter_indices(self):
        indices = None
        nti = 1
        nto = 1
        if self.g_input == 'Z2' and self.g_output == 'C4':
            nti = 1
            nto = 4
            indices = make_c4_z2_indices(self.ksize)
        elif self.g_input == 'C4' and self.g_output == 'C4':
            nti = 4
            nto = 4
            indices = make_c4_p4_indices(self.ksize)
        elif self.g_input == 'Z2' and self.g_output == 'D4':
            nti = 1
            nto = 8
            indices = make_d4_z2_indices(self.ksize)
        elif self.g_input == 'D4' and self.g_output == 'D4':
            nti = 8
            nto = 8
            indices = make_d4_p4m_indices(self.ksize)
        else:
            raise ValueError(f"unsupported g_input g_output pair in make_indices(): {self.g_input, self.g_output}")
        return nti, nto, indices
    
    def transform_filter_2d_nncchw(self, w, inds):
        """
        Transform filter output to be of the form [ksize, ksize, out_channels, nto, input_shape[1], input_shape[0]]
        """
        inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int32)
        w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
        w_indexed = w_indexed.view(w_indexed.size()[0], 
                                   w_indexed.size()[1],
                                   inds.shape[0], 
                                   inds.shape[1], 
                                   inds.shape[2], 
                                   inds.shape[3])
        w_transformed = w_indexed.permute(0, 1, 3, 2, 4, 5) # Previously: w_transformed = w_indexed.permute(0, 1, 3, 2, 4, 5)
        return w_transformed.contiguous()
    
    def transform_filter_2d_nchw(self, y, shape):
        """
        Transform filter output to be of the form [ksize, ksize, out_channels*nto, input_shape[1], input_shape[0]]
        """
        return y.view(shape[0], shape[1]*shape[2], shape[3], shape[4])

    def forward(self, input):
        tw = self.transform_filter_2d_nncchw(self.weight, self.inds)
        tw_shape = (self.out_channels*self.nto,
                    self.in_channels*self.nti,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape)

        input_shape = input.shape
        input = input.reshape(input_shape[0], self.in_channels*self.nti, input_shape[-2], input_shape[-1])

        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)
        
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.nto, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1) # Applies bias to out_channels and not out_channels*nto
            y = y + bias
            
        y = self.transform_filter_2d_nchw(y, [batch_size, self.out_channels, self.nto, ny_out, nx_out])

        return y


def gconv2d(g_input, g_output, *args, **kwargs):
    """
    Wrapper function for SplitConv2D class. Provides group equivariant 2D convolution action on g_input and 
    returning g_output group.

    :parm g_input: One of {'Z2', 'C4', 'D4'}. Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
        The parameter value 'Z2' specifies the data being convolved is from the Z^2 plane (discrete mesh).
    :parm g_output: One of {'C4', 'D4'}. What kind of transformations to use (rotations or roto-reflections).
        The value of g_input of the subsequent layer should match the value of g_output from the previous.
    :parm in_channels: The number of input channels. Based on the input group action the number of channels 
        used is equal to nti*in_channels.
    :parm out_channels: The number of output channels. Based on the output group action the number of channels
        used is equal to nto*out_channels.
    """
    return SplitGConv2d(g_input, g_output, *args, **kwargs)

