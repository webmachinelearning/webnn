| WebNN | matmul | a | b | output |
|----|-----|-----|----|-----| 
| NNAPI | [ANEURALNETWORKS_FULLY_CONNECTED](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0aaada7a3dbaf4676aba560c933ff610c5) | if N > 3, reshape (ANEURALNETWORKS_RESHAPE) to 3-D tensor. if N == 3, slice (ANEURALNETWORKS_RESHAPE) to 2-D tensors along axis 0. Use inputs[0] of ANEURALNETWORKS_FULLY_CONNECTED | if N > 3, reshape (ANEURALNETWORKS_RESHAPE) to 3-D tensor. if N == 3, slice (ANEURALNETWORKS_RESHAPE) to 2-D tensors along left axis. Transpose the tensor (ANEURALNETWORKS_TRANSPOSE). Use inputs[1] of ANEURALNETWORKS_FULLY_CONNECTED | As ANEURALNETWORKS_FULLY_CONNECTED output is 2-D, need to reshape (ANEURALNETWORKS_RESHAPE) to 3-D tensors, concat (ANEURALNETWORKS_CONCATENATION) along axis 9 and reshape (ANEURALNETWORKS_RESHAPE) to N-D. |
| DirectML | [DML_GEMM_OPERATOR_DESC](https://docs.microsoft.com/en-us/windows/win32/api/directml/ns-directml-dml_gemm_operator_desc) when: DML_GEMM_OPERATOR_DESC::Alpha == 1.0, DML_GEMM_OPERATOR_DESC::Beta == 0, DML_GEMM_OPERATOR_DESC::TransA == DML_MATRIX_TRANSFORM_NONE, DML_GEMM_OPERATOR_DESC::TransB == DML_MATRIX_TRANSFORM_NONE, DML_GEMM_OPERATOR_DESC::CTensor == null | **ATensor** of DML_GEMM_OPERATOR_DESC | **BTensor** of DML_GEMM_OPERATOR_DESC | **OutputTensor** of DML_GEMM_OPERATOR_DESC |
| MPS | [MPSNDArrayMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsndarraymatrixmultiplication?language=objc) or [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication?language=objc) or [MPSCNNFullyConnected](https://developer.apple.com/documentation/metalperformanceshaders/mpscnnfullyconnected?language=objc)? | | |
| BNNS | [BNNSFilterCreateFullyConnectedLayer](https://developer.apple.com/documentation/accelerate/1642286-bnnsfiltercreatefullyconnectedla?language=objc) or [vDSP_mmul](https://developer.apple.com/documentation/accelerate/1449984-vdsp_mmul?language=objc)? | | |
| DNNL | [matmul](https://uxlfoundation.github.io/oneDNN/group_dnnl_api_matmul.html) | f N > 3, reorder to 3-D tensor. Use DNNL_ARG_SRC of matmul primitive. | If N > 3, reorder to 3-D tensor. Use DNNL_ARG_WEIGHTS of matmul primitive. | Use DNNL_ARG_DST of matmul primitive. If N > 3, need to reorder to N-D. |
| ONNX | [MatMul](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul) | A | B | Y | 

References:
  * NNAPI: TF-Lite converter [unroll_batch_matmul.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/graph_transformations/unroll_batch_matmul.cc) and [resolve_tensorflow_matmul.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/graph_transformations/resolve_tensorflow_matmul.cc)

**Opens:**
 * NNAPI: for N-D output when N >2, is it correct to map TF-Lite `Pack` op to NNAPI `ANEURALNETWORKS_RESHAPE` and `ANEURALNETWORKS_CONCATENATION` ops?
 * MPS: which is the right op to map? 
 * BNNS: which is the right op to map?
