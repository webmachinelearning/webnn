[For an explainer of an explainer, see the [TAG explainer template](https://github.com/w3ctag/w3ctag.github.io/blob/master/explainers.md). A good concrete example is the [WebXR Device API Explained](https://github.com/immersive-web/webxr/blob/master/explainer.md) document]

# Web Neural Network API Explained

## What’s all this then?

>A brief, 4-5 paragraph explanation of the feature’s value. Outline what the feature does and how it accomplishes those goals (in prose). If your feature creates UI, this is a great place to show mocks and user flows.

With emerging AI innovations in both software and hardware ecosystem, one of the main challenges for the web is to bridge this software and hardware development and bring together a solution that scales across hardware platforms and works with any framework for web AI experiences. We propose the WebNN API as an abstraction for neural networks in the web browsers.

![WebNN architecture](content/webnn_arch.png)

As illustrated in the architecture diagram of the figure above, web browsers may implement the WebNN API using native machine learning API available in the operating system. This architecture allows Javascript frameworks to tap into cutting-edge machine learning innovations in the operating system and the hardware platform underneath it without being tied to platform-specific capabilities, bridging the gap between software and hardware through a hardware-agnostic abstraction layer. 

At the heart of neural networks is a computational graph of mathematical operations. These operations are the building blocks of modern AI and machine learning technologies in computer vision, natural language processing, and robotics. 

The WebNN API is a specification for constructing and executing computational graphs of neural networks. It provides web applications with the ability to create, compile, and run machine learning networks on the web browsers. The WebNN API may be implemented in web browsers using the available native operating system machine learning APIs for the best performance and reliability of results. The following code sample illustrates a simple usage of this API.

    const operandType = { type: 'float32', dimensions: [2, 2] };

    const context = navigator.ml.getNeuralNetworkContext().
    const builder = context.createModelBuilder();

    // 1. Create a model of the computational graph 'C = 0.2 * A + B'
    const constant = builder.constant(0.2);
    const A = builder.input('A', operandType);
    const B = builder.input('B', operandType);
    const C = builder.add(builder.mul(A, constant), B);
    const model = await builder.createModel({'C', C});

    // 2. Compile the model into executable.
    const compilation = await model.compile();

    // 3. Bind inputs to the model and execute for the result
    const bufferA = new Float32Array(4).fill(1.0);
    const bufferB = new Float32Array(4).fill(0.8);
    let outputs = await compilation.compute({'A': { buffer: bufferA }, 'B': { buffer: bufferB }});

    // The computed result of [[1, 1], [1, 1]] is in the buffer associated with the output operand
    console.log(outputs.C.buffer);

### Goals

>How will the web be better when this feature launches? And who will it help?

Web applications and frameworks can take advantage of the native operating system services for machine learning and the underlying hardware innovations available on the user's computers to implement consistent, efficient, and reliable AI experiences on the web platform.

### Non-goals

>You’re not going to solve every problem so enumerate the attractive, nearby problems that are out of scope for this effort. This may include details on the tradeoffs made due to architectural limitations made due to implementation details, and features left out either due to interoperability concerns or other hurdles, and how you plan to improve on this. This can often be the single most important part of your document, so give it careful thought.

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

>Provide a terse example for the most common use case of the feature.  If you need to show how to get the feature set up (initialized, or using permissions, etc.), include that too.

A core abstraction behind popular neural networks is a computational graph, a directed graph with its nodes corresponding to operations (ops) and input variables. One node's output value is the input to another node. The WebNN API brings this abstraction to the web.

In the WebNN API, the [`Operand`](https://webmachinelearning.github.io/webnn/#operand) objects represent input, output, and constant multi-dimensional arrays known as [tensors](https://mathworld.wolfram.com/Tensor.html). The [`NeuralNetworkContext`](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext) defines a set of operations that facilitate the construction and execution of this computational graph. Such operations may be accelerated with dedicated hardware such as the GPUs, CPUs with extensions for deep learning, or dedicated AI accelerators. These operations defined by the WebNN API are required by [models](https://github.com/webmachinelearning/webnn/blob/master/op_compatibility/first_wave_models.md) that address key application use cases. Additionally, the WebNN API provides affordances to builder a computational graph, compile the graph, execute the graph, and integrate the graph with other Web APIs that provide input data to the graph e.g. media APIs for image or video frames and sensor APIs for sensory data.

This [example](https://webmachinelearning.github.io/webnn/#examples) builds, compiles, and executes a graph comprised of three ops, takes four inputs and returns one output.

## Key scenarios

>Next, discuss the key scenarios which move beyond the most canonical example, showing how they are addressed using example code:

There are many important [application use cases](https://webmachinelearning.github.io/webnn/#usecases-application) for high-performance neural network inference. One such use cases is deep-learning noise suppression (DNS) in web-based video conferencing. The following sample shows how the [NSNet2](https://github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline) deep learning model for noise suppression can be implemented in WebNN.

## Detailed design discussion
>Talk through the tradeoffs in coming to the specific design point you want to make, hopefully:

### Do we need a neural network API? Can we standardize on just a model-loader API?

A model-loader API loads a model from a specified URL and outputs a model object on which the caller can execute. It moves all the responsibilities of loading and processing a neural network model to the web browsers and the underlying operating systems while offering the web developers a simple API surface, akin to an image loading API.

Although this design approach has a clear benefit in its simplicity, it faces a challenge in defining a standard model format that works across the various web browsers and operating systems on the user's devices. In shifting the focus of the design towards the model format, it creates an opportunity for more fragmentation in the way AI is consumed on the web and encourages silos of vendor-specific ecosystems around the particular model formats of choice. Much like in the early days of the image format wars, the web developers will likely have a more difficult time understanding which model formats would work on which combinations of the web browsers and operating systems that they're targeting.

By defining the WebNN API as a model format-agnostic set of neural network operations, we shift the focus of the design towards the abstraction between the web browsers and the underlying operating system services and let the web applications and Javascript frameworks continue to focus on satisfying the needs of the web developers knowing that the neural networks they create will faithfully execute regardless of the browser's underlying platform. What we believe works in our favor is the significant overlap of neural network operations and algorithms across all popular vendor-specific frameworks today; models available in one format are generally convertible with little loss to another.

It is also worth considering that a model-loader API can be built atop a neural network API without losing the appeal in its simplicity. Our view is that the two APIs are complementary and not mutually exclusive, but we must start with the neural network API.

An explainer for the model-loader API can be found [here](https://github.com/webmachinelearning/model-loader/blob/master/explainer.md).

### What is the right level of abstraction for the neural network operations?

Neural network operations are mathematical functions. There are about a hundred standard functions universally supported in popular frameworks today e.g. convolution, matrix multiplication, various reductions, and normalizations. Additionally, some frameworks provide an even more extensive set of variants of these functions for ease of use. 

In designing the WebNN operations, a proposal to decompose high-level functions to the more rudimentary mathematical operations was considered, with the key benefit of having a reduced number of operations defined. However, such an approach would make the networks more verbose and harder to construct. It'll also risk losing the opportunity to leverage known optimizations for highly reusable functions in the operating system. For instance, most operating systems and modern hardware today support popular recurrent networks out of the box. By decomposing well-known functions into networks of smaller operations, their identities may be lost in the process.

To balance the needs of providing for future extensibility while ensuring maximum reuse and performance optimization opportunity, we chose to include both the standard functions and all the smaller operations making up the functions in the spec. For each high-level function defined, we make sure that all of its decomposed operations are also defined. This way, a newly-conceived function may be represented as a network of our decomposed operations, while a standard function can also be fully supported by the underlying platforms.

## Considered alternatives

>One of the most important things you can do in your design process is to catalog the set of roads not taken. As you iterate on your design, you may find that major choices in your approach or API style will be revisited and enumerating the full space of alternatives can help you apply one (or more) of them later, may serve as a “graveyard” for u-turns in your design, and can give reviewers and potential users confidence that you’ve got your ducks in a row.

## References & acknowledgements

>Your design will change and be informed by many people; acknowledge them in an ongoing way! It helps builder community and, as we only get by through the contributions of many, is only fair.
