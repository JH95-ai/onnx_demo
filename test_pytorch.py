import torch.nn as nn
import torch.nn.init as init
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import io
import numpy as np
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
class SuperResolutionNet(nn.Module):
    def __init__(self,upscale_factor,inplace=False):
        super(SuperResolutionNet,self).__init__()
        self.relu=nn.ReLU(inplace=inplace)
        self.conv1=nn.Conv2d(1,64,(5,5),(1,1),(2,2))
        self.conv2=nn.Conv2d(64,64,(3,3),(1,1),(1,1))
        self.conv3=nn.Conv2d(64,32,(3,3),(1,1),(1,1))
        self.conv4=nn.Conv2d(32,upscale_factor **2,(3,3),(1,1),(1,1))
        self.pixel_shuffle=nn.PixelShuffle(upscale_factor)
        self._initialize_weights()
    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.relu(self.conv3(x))
        x=self.pixel_shuffle(self.conv4(x))
        return x
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight,init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight,init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight,init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

#Create the super-resolution model by using the above model definition
torch_model=SuperResolutionNet(upscale_factor=3)

#Load pretrained model weights
model_url='https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size=1    #just a random number

#Initialize model with the pretrained weights
map_location=lambda storage,loc:storage
if torch.cuda.is_available():
    map_location=None
torch_model.load_state_dict(model_zoo.load_url(model_url,map_location=map_location))

#set the train mode to false since we will only run the forward pass.
torch_model.train(False)

#Input to the model
x=torch.randn(batch_size,1,224,224,requires_grad=True)

#Export the model
torch_out=torch.onnx._export(torch_model,   #model being run
                             x,             #model input(or a tuple for multiple inputs)
                             'super_resolution.onnx',   #where to save the model(can be a file or file-like object)
                             export_params=True)    #store the trained parameter weights inside the model file
#Load the ONNX ModelProto object.model is a standard Python protobuf object
model=onnx.load('super_resolution.onnx')
#prepare the caffe2 backend for executinf the model this converts the ONNX model into
#a Caffe2 NetDef that can execute it.Other ONNX bakends,like one for CNTK will be
#availiable soon.
prepared_backend=onnx_caffe2_backend.prepare(model)
#run the model in Caffe2
#Construct a map from input names to Tensor data.
#The graph of the model itself contains inputs for all weight parameters,
#after the input image.
W={model.graph.input[0].name:x.data.numpy()}
#Run the caffe2 net:
c2_out=prepared_backend.run(W)[0]
#Verify the numerical correctness upto 3 decimal places
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(),c2_out,decimal=3)
print('Exported model has been executed on Caffe2 backend,and the result looks good!')

# extract the workspace and the model proto from the internal representation
c2_workspace = prepared_backend.workspace
c2_model = prepared_backend.predict_net
