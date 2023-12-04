import torch
from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4

# Construct G-Conv layers
C1 = P4ConvZ2(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
C2 = P4ConvP4(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

# Create 10 images with 3 channels and 9x9 pixels:
x = Variable(torch.randn(10, 3, 9, 9))

# fprop
y = C2(C1(x))
print(y.data.shape)  # (10, 64, 4, 9, 9)