from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import onnx
import os
from onnx import optimizer
#preprocessing:load the model contains two transposes
model_path=os.path.join('/tmp/mozilla_jethro0/onnx-master/onnx/examples/resources/two_transposes.onnx')
original_model=onnx.load(model_path)
print('The model before optimization:\n\n{}'.format(onnx.helper.printable_graph(original_model.graph)))
#A full list of supported optimization passes can be found using get_available_passes()
all_passes=optimizer.get_available_passes()
print('Avaiable optimization passes:')
for p in all_passes:
    print('\t{}'.format(p))
print()
#Pick one pass as example
passes=['fuse_consecutive_transposes']
#Apply the optimization on the original serialized model
optimized_model=optimizer.optimize(original_model,passes)
print('The model after optimization:\n\n{}'.format(onnx.helper.printable_graph(optimized_model.graph)))
