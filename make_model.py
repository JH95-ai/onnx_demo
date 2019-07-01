from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import onnx
from onnx import helper
from onnx import AttributeProto,TensorProto,GraphProto
#The protobuf definition can be found here:
#https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

#Create one input (ValueInfoProto)
X=helper.make_tensor_value_info('X',TensorProto.FLOAT,[1,2])
#Create one output (ValueInfoProto)
Y=helper.make_tensor_value_info('Y',TensorProto.FLOAT,[1,4])
#Create a node(NodeProto)
node_def=helper.make_node(
    'Pad',#node name
    ['X'],#inputs
    ['Y'],#outputs
    mode='constant',#Attributes
    value=1.5,
    pads=[0,1,0,1],
)
#Create the Graph(GraphProto)
graph_def =helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
)
#Create the model(ModelProto)
model_def =helper.make_model(graph_def,producer_name='onnx-example')
print('The ir_vision in model:{}\n'.format(model_def.ir_version))
print('The producer_name in model:{}\n'.format(model_def.producer_name))
print('The graph in model:{}\n'.format(model_def.graph))
onnx.checker.check_model(model_def)
print('The model is checked')