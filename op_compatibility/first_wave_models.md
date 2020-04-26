## The first-wave models

| Model | Use Case |
|----------|--------|
| [SqueezeNet](https://arxiv.org/abs/1602.07360) | Image Classification |
| [MobileNet](https://arxiv.org/abs/1801.04381) | Image Classification |
| [ResNet](https://arxiv.org/abs/1603.05027) | Image Classifciation |
| [TinyYOLO](https://arxiv.org/abs/1612.08242) | Object Detection |

## The ops required by the first-wave models
| Op | SqueezeNetV1.1 [ONNX](https://github.com/onnx/models/tree/master/vision/classification/squeezenet) [TFLite](https://www.tensorflow.org/lite/guide/hosted_models) | MobileNetV2 [ONNX](https://github.com/onnx/models/tree/master/vision/classification/mobilenet) [TFLite](https://www.tensorflow.org/lite/guide/hosted_models) | ResNetV2 [ONNX](https://github.com/onnx/models/tree/master/vision/classification/resnet) [TFLite](https://www.tensorflow.org/lite/guide/hosted_models) | TinyYOLOV2 [ONNX](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov2) [TFLite](https://github.com/intel/webml-polyfill/tree/master/examples/object_detection/model#for-tiny-yolo-models) | Remarks |
|----|----|----|----|----|----|
| Add | | :heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: | The existing [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-add) is to be completed. |
| AveragePool |:heavy_check_mark: | | | |
| BatchNormalization | |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: | TFLite models either fuse it into preceding `conv2d` or replace it by `mul` and `add`. |
| Concat |:heavy_check_mark: | | | | |
| Conv |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: | Need to support depthwise conv2d of MobileNetV2. Will we add `groups` attribute into existing [`conv2d`](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-conv2d) or add new ops like `groupedConv2d` or `depthwiseConv2d`? |
| Gemm | | |:heavy_check_mark: | | Can it be covered by existing [`matmul`](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-matmul) and [`add`](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-add)? |
| GlobalAveragePool | |:heavy_check_mark: |:heavy_check_mark: | | Can it be covered by `AveragePool`? |
| LeakyRelu | | | |:heavy_check_mark: | TFLite models substitute it by `mul` and `maximum` as `y = mul(maximum(x, 0), alpha)`. |
| MaxPool |:heavy_check_mark: | |:heavy_check_mark: |:heavy_check_mark: | |
| Mul | | | | :heavy_check_mark: | The existing [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-mul) is to be completed. |
| Relu |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: | | |
| Reshape |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: | | |
| Softmax |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: | | It is for post-processing and not included in ONNX models. |