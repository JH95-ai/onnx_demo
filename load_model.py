from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import onnx
import os
#Load the ONNX model
onnx_model=onnx.load(os.path.join('/tmp/mozilla_jethro0/onnx-master/onnx/examples/resources','single_relu.onnx'))
print(onnx_model)