import numpy
import onnx
from onnx import numpy_helper
from distutils.version import LooseVersion
#Preprocessing:create a Numpy array
numpy_array=numpy.array([[1.0,2.0,3.0],[4.0,5.0,6.0]],dtype=float)
if LooseVersion(numpy.version.version) < LooseVersion('1.14'):
    print('Original Numpy array:\n{}\n'.format(numpy.array2string(numpy_array)))
else:
    print('Original Numpy array:\n{}\n'.format(numpy.array2string(numpy_array, legacy='1.13')))
#Convert the Numpy array to a TensorProto
tensor=numpy_helper.from_array(numpy_array)
print('TensorProto:\n{}'.format(tensor))

#Convert the TensorProto to a Numpy arry
new_array=numpy_helper.to_array(tensor)
print('After round trip.Numpy array;\n{}\n'.format(numpy_array))
#Save the TensorProto
with open('tensor_test.pb','wb')as f:
    f.write(tensor.SerializeToString())
#Load a TensorProto
new_tensor=onnx.TensorProto()
with open('tensor_test.pb','rb')as f:
    new_tensor.ParseFromString(f.read())
print('After saving and loading,new TensorProto:\n{}'.format(new_tensor))
