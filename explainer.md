[For an explainer of an explainer, see the [TAG explainer template](https://github.com/w3ctag/w3ctag.github.io/blob/master/explainers.md). A good concrete example is the [WebXR Device API Explained](https://github.com/immersive-web/webxr/blob/master/explainer.md) document]

# Web Neural Network API Explained

## What’s all this then?

>A brief, 4-5 paragraph explanation of the feature’s value. Outline what the feature does and how it accomplishes those goals (in prose). If your feature creates UI, this is a great place to show mocks and user flows.

### Goals

>How will the web be better when this feature launches? And who will it help?

### Non-goals

>You’re not going to solve every problem so enumerate the attractive, nearby problems that are out of scope for this effort. This may include details on the tradeoffs made due to architectural limitations made due to implementation details, and features left out either due to interoperability concerns or other hurdles, and how you plan to improve on this. This can often be the single most important part of your document, so give it careful thought.

### Target hardware

With this API web developers are able to target typical computing devices people use in their daily lives that run major operating systems. Initial prototypes have demonstrated reasonable performance on:

* smartphones e.g. Google Pixel 3 or equivalent
* laptops e.g. 13" MacBook Pro 2015 or equivalent

The APIs in scope of this group will not be tied to any particular platform and will be implementable on top of existing major platform APIs, such as:

* Android Neural Networks API
* Windows DirectML
* macOS/iOS Metal Performance Shaders and Basic Neural Network Subroutines

Depending on the underlying hardware capabilities, these platform APIs may make use of CPU parallelism, general-purpose GPU, or dedicated ML hardware accelerators. The API will provide high-level hints to web developers to enable [performance adaptation](https://webmachinelearning.github.io/webnn/#usecase-perf-adapt), but will remain hardware agnostic otherwise.

## Getting started

>Provide a terse example for the most common use case of the feature.  If you need to show how to get the feature set up (initialized, or using permissions, etc.), include that too.

A core abstraction behind popular deep learning frameworks is a computational graph. A computational graph is a directed graph with its nodes corresponding to operations (ops) and input variables. One node's output value is the input to another node. WebNN API brings this abstraction to the Web.

In WebNN API [`Operand`](https://webmachinelearning.github.io/webnn/#operand) objects represent input, output, and constant multi-dimensional arrays known as [tensors](https://mathworld.wolfram.com/Tensor.html). [`NeuralNetworkContext`](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext) defines a set of operations that may be accelerated with dedicated hardware such as the GPUs, CPUs with extensions for deep learning, or dedicated AI accelerators. These operations defined by the WebNN API are required by [models](https://github.com/webmachinelearning/webnn/blob/master/op_compatibility/first_wave_models.md) that address identified [use cases](https://webmachinelearning.github.io/webnn/#usecases). In addition, WebNN API provides affordances to build a computational graph, compile the graph, execute the graph, and integrate the graph with other Web APIs that provide input data to the graph e.g. media APIs for image or video frames and sensor APIs for sensory data.

[Example](https://webmachinelearning.github.io/webnn/#examples) builds, compiles, and executes a graph comprised of three ops, takes four inputs and returns one output.

## Key scenarios

>Next, discuss the key scenarios which move beyond the most canonical example, showing how they are addressed using example code:

…
## Detailed design discussion

### Graph API vs Loader API

>Talk through the tradeoffs in coming to the specific design point you want to make, hopefully:

### Tricky design choice N

…

## Considered alternatives

>One of the most important things you can do in your design process is to catalog the set of roads not taken. As you iterate on your design, you may find that major choices in your approach or API style will be revisited and enumerating the full space of alternatives can help you apply one (or more) of them later, may serve as a “graveyard” for u-turns in your design, and can give reviewers and potential users confidence that you’ve got your ducks in a row.

## References & acknowledgements

>Your design will change and be informed by many people; acknowledge them in an ongoing way! It helps build community and, as we only get by through the contributions of many, is only fair.
