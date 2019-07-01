from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import os
#Preprocessing:load the old model
old_model_path=os.path.join('/tmp/mozilla_jethro0/onnx-master/onnx/examples/resources','single_relu.onnx')
onnx_model=onnx.load(old_model_path)
#Preprocessing :get the path to saved model
new_model_path=os.path.join('/tmp/mozilla_jethro0/onnx-master/onnx/examples/resources','single_relu_new2.onnx')
#Save the ONNX model
onnx.save(onnx_model,new_model_path)
print('The model is saved.')