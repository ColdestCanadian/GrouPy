import numpy as np


### ---[ Test pytorch implementation ]-----------
import torch as pt
from torch.autograd import Variable
from groupy.gconv.pytorch_gconv.splitgconv2d import GConv2d
from groupy.gfunc import Z2FuncArray, P4FuncArray
import groupy.garray.C4_array as c4a
from PIL import Image

def test_p4_net_equivariance():
    im = np.random.randn(1, 1, 11, 11)
    check_equivariance(
        im=im,
        layers=[
            GConv2d(g_input='Z2', g_output='C4', in_channels=1, out_channels=1, kernel_size=3, padding=1)
        ],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )

def check_equivariance(im, layers, input_array, output_array, point_group):
    # Transform the image
    print("Input: "+str(im), flush=True)
    f = input_array(im)
    print("Network output: "+str(f), flush=True)
    g = point_group.rand()
    gf = g*f # Default g*f
    im1 = gf.v
    # Apply layers to both images
    im = Variable(pt.Tensor(im))
    im1 = Variable(pt.Tensor(im1))

    fmap = im
    fmap1 = im1
    for layer in layers:
        fmap = layer(fmap)
        fmap1 = layer(fmap1)

    # Transform the computed feature maps
    fmap1_garray = output_array(fmap1.data.numpy())
    r_fmap1_data = (g.inv() * fmap1_garray).v

    fmap_data = fmap.data.numpy()
    assert np.allclose(fmap_data, r_fmap1_data, rtol=1e-5, atol=1e-3)


def test_p4_net_pooling_equivariance():
    # im = np.random.randn(1, 1, 11, 11)
    im = pt.randn(1,1,5,5)
    imT = pt.rot90(im, dims=[2,3])
    layers=[
            GConv2d(g_input='Z2', g_output='C4', in_channels=1, out_channels=1, kernel_size=3, padding=1)
        ]
    
    print("Image : "+str(im))
    print("Image.T : "+str(imT))
    
    y = im
    for layer in layers:
        y = layer(y)
        print("y: "+str(y))
    y = pt.mean(y, dim=2)

    yT = imT
    for layer in layers:
        yT = layer(yT)
        print("yT: "+str(yT))
    yT = pt.mean(yT, dim=2)

    print("y_pooled: "+str(y))
    print("yT_pooled: "+str(yT))
    difference = pt.abs(y-yT)
    error = pt.sum(difference)

    print("Error : "+str(error))
    print("Difference: "+str(difference))


### ---[ Test tensforflow implementation ]-------
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util, gconv2d
from groupy.gfunc.z2func_array import Z2FuncArray
from groupy.gfunc.p4func_array import P4FuncArray
from groupy.gfunc.p4mfunc_array import P4MFuncArray
import groupy.garray.C4_array as C4a
import groupy.garray.D4_array as D4a

def check_c4_z2_conv_equivariance():
    im = np.random.randn(2, 5, 5, 1)
    imT = np.rot90(im)
    
    print("Image: "+str(im))
    print("Image.T: "+str(imT))

    x, y = make_graph('Z2', 'C4')

    print("x: "+str(x))
    print("y: "+str(y))

    check_equivariance(im, x, y, Z2FuncArray, P4FuncArray, C4a)


def make_graph(h_input, h_output):
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
        h_input=h_input, h_output=h_output, in_channels=1, out_channels=1, ksize=3)
    nti = gconv_shape_info[-2]
    x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 5, 5, 1 * nti])
    w = tf.Variable(tf.compat.v1.truncated_normal(shape=w_shape, stddev=1.))
    y = gconv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME',
                gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
    return x, y


def check_equivariance(im, input, output, input_array, output_array, point_group):

    # Transform the image
    f = input_array(im.transpose((0, 3, 1, 2)))
    g = point_group.rand()
    gf = g * f
    im1 = gf.v.transpose((0, 2, 3, 1))

    # Compute
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    yx = sess.run(output, feed_dict={input: im})
    yrx = sess.run(output, feed_dict={input: im1})
    sess.close()

    # Transform the computed feature maps
    fmap1_garray = output_array(yrx.transpose((0, 3, 1, 2)))
    r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 1))

    print (np.abs(yx - r_fmap1_data).sum())
    assert np.allclose(yx, r_fmap1_data, rtol=1e-5, atol=1e-3)



### ---[ Main ]----------------------------------

if __name__=="__main__":
    # image_dir = "/Users/spencerszabados/Library/CloudStorage/OneDrive-UniversityofWaterloo/Projects/Research/GrouPy/groupy/gconv/pytorch_gconv/test_image.JPEG"
    # im = np.asarray(Image.open(image_dir))
    # test_p4_net_equivariance()
    # test_p4_net_pooling_equivariance()
    check_c4_z2_conv_equivariance()


