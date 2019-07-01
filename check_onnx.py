from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import onnx
import os
#preprocessing:load the ONNX model
model_path=os.path.join('/tmp/mozilla_jethro0/onnx-master/onnx/examples/resources','single_relu.onnx')
onnx_model=onnx.load(model_path)
print('The model is:\n{}'.format(onnx_model))
#Check the model
onnx.checker.check_model(onnx_model)
print('The model is checked')