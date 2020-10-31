# Web Neural Network API Explained

## What’s all this then?

With emerging AI innovations in both software and hardware ecosystem, one of the main challenges for the web is to bridge this software and hardware development and bring together a solution that scales across hardware platforms and works with any framework for web AI experiences. We propose the WebNN API as an abstraction for neural networks in the web browsers.

![WebNN architecture](content/webnn_arch.png)

As illustrated in the architecture diagram of the figure above, web browsers may implement the WebNN API using native machine learning API available in the operating system. This architecture allows JavaScript frameworks to tap into cutting-edge machine learning innovations in the operating system and the hardware platform underneath it without being tied to platform-specific capabilities, bridging the gap between software and hardware through a hardware-agnostic abstraction layer. 

At the heart of neural networks is a computational graph of mathematical operations. These operations are the building blocks of modern AI and machine learning technologies in computer vision, natural language processing, and robotics. 

The WebNN API is a specification for constructing and executing computational graphs of neural networks. It provides web applications with the ability to create, compile, and run machine learning networks on the web browsers. The WebNN API may be implemented in web browsers using the available native operating system machine learning APIs for the best performance and reliability of results. The following code sample illustrates a simple usage of this API.

``` JavaScript
const operandType = {type: 'float32', dimensions: [2, 2]};
const context = navigator.ml.getNeuralNetworkContext();
const builder = context.createModelBuilder();
// 1. Create a model of the computational graph 'C = 0.2 * A + B'.
const constant = builder.constant(0.2);
const A = builder.input('A', operandType);
const B = builder.input('B', operandType);
const C = builder.add(builder.mul(A, constant), B);
const model = builder.createModel({'C': C});
// 2. Compile the model into executable.
const compilation = await model.compile();
// 3. Bind inputs to the model and execute for the result.
const bufferA = new Float32Array(4).fill(1.0);
const bufferB = new Float32Array(4).fill(0.8);
const inputs = {'A': {buffer: bufferA}, 'B': {buffer: bufferB}};
const outputs = await compilation.compute(inputs);
// The computed result of [[1, 1], [1, 1]] is in the buffer associated with
// the output operand.
console.log('Output shape: ' + outputs.C.dimensions);
console.log('Output value: ' + outputs.C.buffer);
```

### Goals

Web applications and frameworks can take advantage of the native operating system services for machine learning and the underlying hardware innovations available on the user's computers to implement consistent, efficient, and reliable AI experiences on the web platform.

### Non-goals

1. We do not define model serialization format. Formats are framework's choices, which may be vendor-specific. The role of the WebNN API is to facilitate solutions that work across the web regardless of the model format by leveraging the underlying support available in the target platform for reliable and efficient results.

2. We do not define the packaging and delivery mechanism of machine learning models, such as the choice of encryption and content protection.

3. The machine learning model's input may be of any media type, such as video streams, images, or audio signals. We do not define new media types but relying on the existing web standards for media types needed for respective scenarios.

### Target hardware

Web applications and frameworks can target typical computing devices on popular operating systems that people use in their daily lives. Initial prototypes demonstrate respectable performance on:

* Smartphones e.g. Google Pixel 3 or similar
* Laptops e.g. 13" MacBook Pro 2015 or similar

The WebNN API is not tied to specific platforms and is implementable by existing major platform APIs, such as:

* Android Neural Networks API
* Windows DirectML API
* macOS/iOS Metal Performance Shaders and Basic Neural Network Subroutines

Depending on the underlying hardware capabilities, these platform APIs may make use of CPU parallelism, general-purpose GPU, or dedicated hardware accelerators for machine learning. The WebNN API provides [performance adaptation](https://webmachinelearning.github.io/webnn/#usecase-perf-adapt) options but remains hardware agnostic.

## Getting started

A core abstraction behind popular neural networks is a computational graph, a directed graph with its nodes corresponding to operations (ops) and input variables. One node's output value is the input to another node. The WebNN API brings this abstraction to the web.

In the WebNN API, the [`Operand`](https://webmachinelearning.github.io/webnn/#operand) objects represent input, output, and constant multi-dimensional arrays known as [tensors](https://mathworld.wolfram.com/Tensor.html). The [`NeuralNetworkContext`](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext) defines a set of operations that facilitate the construction and execution of this computational graph. Such operations may be accelerated with dedicated hardware such as the GPUs, CPUs with extensions for deep learning, or dedicated AI accelerators. These operations defined by the WebNN API are required by [models](https://github.com/webmachinelearning/webnn/blob/master/op_compatibility/first_wave_models.md) that address key application use cases. Additionally, the WebNN API provides affordances to builder a computational graph, compile the graph, execute the graph, and integrate the graph with other Web APIs that provide input data to the graph e.g. media APIs for image or video frames and sensor APIs for sensory data.

This [example](https://webmachinelearning.github.io/webnn/#examples) builds, compiles, and executes a graph comprised of three ops, takes four inputs and returns one output.

## Key scenarios

There are many important [application use cases](https://webmachinelearning.github.io/webnn/#usecases-application) for high-performance neural network inference. One such use cases is deep-learning noise suppression (DNS) in web-based video conferencing. The following sample shows the [NSNet](https://github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline) deep learning model for noise suppression implemented in WebNN.

```JavaScript
// Noise Suppression Net (NSNet) Baseline Model for Deep Noise Suppression Challenge (DNS) 2020.
//      https://github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline
async function nsnet(sequenceLength, batchSize) {
    // Constant shapes and sizes
    const HIDDEN_DIMS = [1,1,257];
    const HIDDEN_SIZE = 257;
    const WEIGHT_DIMS = [1,771,257];
    const WEIGHT_SIZE = 771 * 257;
    const BIAS_DIMS = [1,1542];
    const BIAS_SIZE = 1542;
    const MATMUL96_INIT_DIMS = [257,257];
    const MATMUL96_INIT_SIZE = 257 * 257;
    const ADD97_INIT_DIMS = [257];
    const ADD97_INIT_SIZE = 257;
    const INPUT_DIMS = [sequenceLength, batchSize, 257];
    const INPUT_SIZE = sequenceLength * batchSize * 257;
    // Load pre-trained constant data and initializers
    let response = await fetch(hiddenUrl);
    let buffer = await response.arrayBuffer();
    const hiddenData1 = new Float32Array(buffer, 0, HIDDEN_SIZE);
    const hiddenData2 = new Float32Array(buffer, HIDDEN_SIZE, HIDDEN_SIZE);
    const hiddenData3 = new Float32Array(buffer, 2 * HIDDEN_SIZE, HIDDEN_SIZE);
    response = await fetch(weightUrl);
    buffer = await response.arrayBuffer();
    const weightData117 = new Float32Array(buffer, 0, WEIGHT_SIZE);
    const weightData118 = new Float32Array(buffer, WEIGHT_SIZE, WEIGHT_SIZE);
    const weightData137 = new Float32Array(buffer, 2 * WEIGHT_SIZE, WEIGHT_SIZE);
    const weightData138 = new Float32Array(buffer, 3 * WEIGHT_SIZE, WEIGHT_SIZE);
    const weightData157 = new Float32Array(buffer, 4 * WEIGHT_SIZE, WEIGHT_SIZE);
    const weightData158 = new Float32Array(buffer, 5 * WEIGHT_SIZE, WEIGHT_SIZE);
    response = await fetch(biasUrl);
    buffer = await response.arrayBuffer();
    const biasData119 = new Float32Array(buffer, 0, BIAS_SIZE);
    const biasData139 = new Float32Array(buffer, BIAS_SIZE, BIAS_SIZE);
    const biasData159 = new Float32Array(buffer, 2 * BIAS_SIZE, BIAS_SIZE);
    response = await fetch(initializerUrl);
    buffer = await response.arrayBuffer();
    const initData160 = new Float32Array(buffer, 0, MATMUL96_INIT_SIZE);
    const initData170 = new Float32Array(buffer, MATMUL96_INIT_SIZE, ADD97_INIT_SIZE);
    // Create constant operands
    const builder = navigator.ml.getNeuralNetworkContext().creatModelBuilder();
    const hidden1 = builder.constant({ type: 'float32', dimensions: HIDDEN_DIMS }, hiddenData1);
    const hidden2 = builder.constant({ type: 'float32', dimensions: HIDDEN_DIMS }, hiddenData2);
    const hidden3 = builder.constant({ type: 'float32', dimensions: HIDDEN_DIMS }, hiddenData3);
    const weight117 = builder.constant({ type: 'float32', dimensions: WEIGHT_DIMS }, weightData117);
    const weight118 = builder.constant({ type: 'float32', dimensions: WEIGHT_DIMS }, weightData118);
    const weight137 = builder.constant({ type: 'float32', dimensions: WEIGHT_DIMS }, weightData137);
    const weight138 = builder.constant({ type: 'float32', dimensions: WEIGHT_DIMS }, weightData138);
    const weight157 = builder.constant({ type: 'float32', dimensions: WEIGHT_DIMS }, weightData157);
    const weight158 = builder.constant({ type: 'float32', dimensions: WEIGHT_DIMS }, weightData158);
    const bias119 = builder.constant({ type: 'float32', dimensions: BIAS_DIMS }, biasData119);
    const bias139 = builder.constant({ type: 'float32', dimensions: BIAS_DIMS }, biasData139);
    const bias159 = builder.constant({ type: 'float32', dimensions: BIAS_DIMS }, biasData159);
    const init160 = builder.constant({ type: 'float32', dimensions: MATMUL96_INIT_DIMS }, initData160);
    const init170 = builder.constant({ type: 'float32', dimensions: ADD97_INIT_DIMS }, initData170);
    // Build up the network
    const input = builder.input('input', { type:'float32', dimensions: INPUT_DIMS });
    const [gru43, gru42] = builder.gru(input, weight117, weight118, sequenceLength, 257, 
                                { bias: bias119, initialHiddenState: hidden1, returnSequence: true });
    const add45 = builder.add(input, builder.squeeze(gru42, { axes: [1] }));
    const [gru68, gru67] = builder.gru(add45, weight137, weight138, sequenceLength, 257, 
                                { bias: bias139, initialHiddenState: hidden2, returnSequence: true });
    const add70 = builder.add(add45, builder.squeeze(gru67, { axes: [1] }));
    const [gru93, gru92] = builder.gru(add70, weight157, weight158, sequenceLength, 257, 
                                { bias: bias159, initialHiddenState: hidden3, returnSequence: true });
    const output = builder.clamp(
                    builder.sigmoid(
                        builder.add(
                            builder.matmul(builder.squeeze(gru92, { axes: [1] }), init160), 
                            init170)
                        ), { minValue: builder.constant(0) });
    // Compile the model
    const model = builder.createModel({ 'output': output });
    return await model.compile();
}

async function run(model, inputBuffer) {
    // Run the compiled model with the input data
    return await model.compute({ 'input': { buffer: inputBuffer } });
}

```
## Detailed design discussion

### Do we need a neural network API? Can we standardize on just a model-loader API?

A model-loader API loads a model from a specified URL and outputs a model object on which the caller can execute. It leaves all the responsibilities of loading and processing a neural network model to the web browsers and the underlying operating systems while offering the web developers a simple API surface, akin to an image loading API.

Although this design approach has a clear benefit in its simplicity, it faces a challenge in defining a standard model format that works across the various web browsers and operating systems on the user's devices. In shifting the focus of the design towards the model format, it creates an opportunity for more fragmentation in the way AI is consumed on the web and encourages silos of vendor-specific ecosystems around the particular model formats of choice. Much like in the early days of the image format wars, the web developers will likely have a more difficult time understanding which model formats would work on which combinations of the web browsers and operating systems that they're targeting.

By defining the WebNN API as a model format-agnostic set of neural network operations, we shift the focus of the design towards the abstraction between the web browsers and the underlying operating system services and let the web applications and JavaScript frameworks continue to focus on satisfying the needs of the web developers knowing that the neural networks they create will faithfully execute regardless of the browser's underlying platform. What we believe works in our favor is the significant overlap of neural network operations and algorithms across all popular frameworks today. Models available in one format are generally convertible to another with little loss.

A model-loader API can also be built atop a neural network API without losing the appeal in its simplicity. Our view is that the two APIs are complementary and not mutually exclusive to each other; however we must start with the neural network API to ensure cross-platform interoperability, a cornerstone of the web platform.

An explainer for the model-loader API can be found [here](https://github.com/webmachinelearning/model-loader/blob/master/explainer.md).

### What is the right level of abstraction for the neural network operations?

Neural network operations are mathematical functions. There are about a hundred standard functions universally supported in popular frameworks today e.g. convolution, matrix multiplication, various reductions, and normalizations. Additionally, some frameworks provide an even more extensive set of variants of these functions for ease of use. 

In designing the WebNN operations, a proposal to decompose high-level functions to the more rudimentary mathematical operations was considered, with the key benefit of having a reduced number of operations defined. However, such an approach would make the networks more verbose and harder to construct. It'll also risk losing the opportunity to leverage known optimizations for highly reusable functions in the operating systems and in the hardware platforms underneath it. For instance, most operating systems and modern hardware today support widely-used variants of convolutions and recurrent networks out of the box. By decomposing well-known functions into networks of rudimentary mathematical operations, their identities may be lost in the process with opportunities for significant performance gains left behind.

To balance the needs of providing for future extensibility while ensuring maximum reuse and performance optimization opportunity, we chose to include both the standard functions and all the smaller operations making up the functions in the spec. For each high-level function defined, we make sure that all of its decomposed operations are also defined. This way, a newly-conceived function may be represented as a network of our decomposed operations, while a standard function can also be fully supported by the underlying platforms. An elaborate example of this principle is in the way we define the specification of the [gruCell](https://webmachinelearning.github.io/webnn/#api-modelbuilder-grucell) operation as described in its notes.

## Considered alternatives

### Stay the course and build machine learning solutions on WebGL/WebGPU

WebGL and WebGPU are Web API abstraction to the underlying graphics API, which could be used to implement neural network operations that run on the GPU. Popular JavaScript machine learning frameworks such as TensorFlow.JS already uses WebGL and plans to add a WebGPU backend shortly in the future. An alternative to the WebNN proposal is to continue with this architecture and rely on JavaScript frameworks implemented with these graphics abstraction to address the current and future needs of AI scenarios on the web.

We believe this alternative is insufficient for two reasons. First, although graphics abstraction layers provide the flexibility of general programmability of the GPU graphics pipelines, they are unable to tap into hardware-specific optimizations and special instructions that are only available to the operating system internals. Over the years, the hardware ecosystem has invested significantly in innovating in the AI space, and much of that is about improving the performance of intensive computing workload in machine learning scenarios. Key technologies such as massively parallelized matrix multiplication, weight packing, and quantization, to name a few, are fundamental to major performance breakthroughs but are not accessible to applications through generic graphics pipeline states due to its nature of being hardware-dependent.

Secondly, the hardware diversity with numerous driver generations make conformance testing of neural network operations at the framework level challenging. Conformance testing, compatibility, and quality assurance of hardware results have been the areas of strength of the operating systems, something that should be leveraged by frameworks and applications alike. Neural network models could be used in mission-critical scenarios such as in healthcare or industry processes, the trustworthiness of the results produced by the frameworks are of utmost importance to the users.

## References & acknowledgements

Thanks to all the [Machine Learning for the Web Community Group](https://www.w3.org/community/webmachinelearning/) and [W3C Workshop on Web and Machine Learning](https://www.w3.org/2020/06/machine-learning-workshop/) participants for their comments and feedbacks that have informed the design of this API.
