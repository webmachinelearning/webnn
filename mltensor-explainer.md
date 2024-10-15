# `MLTensor` Explainer

## Authors

- [Austin Sullivan](asully@chromium.org) (Google)

## Participate

- [Issue tracker](https://github.com/webmachinelearning/webnn/issues)

## Introduction

This explainer proposes an `MLTensor` interface which represents a tensor which can be passed as an input and output to `MLGraph` inference.

The machine learning context underlying WebNN may require input and output tensors to be allocated in a specific fashion, such as with a given byte alignment or on a given compute unit (e.g. CPU, GPU, NPU, TPU, etc...). Currently, this requires that the implementation of an `MLGraph` copy in data from the input tensors, execute the graph, and then copy out data from the output tensors.

An `MLTensor` is an opaque tensor which may be created, written to, and read from independently from `MLGraph` inference. Each of these operations is performed on the [timeline](\#timelines) of the associated `MLContext`, with a clearly defined order of operations. Passing `MLTensor`s as input and output tensors to `MLGraph` inference - as opposed to passing `ArrayBufferView`s as is done today - allows for a decoupling of the uploading/downloading of model inputs/outputs from the model execution itself. This provides several benefits, such as buffer reuse, chained inference, explicit destruction, and the opportunity to share memory with WebGPU.

## Goals

- Provide a consistent API for passing tensors to an `MLGraph` which may run arbitrary compute units (e.g. CPU, GPU, NPU, TPU, etc...)
- Improve model throughput by minimizing the need for synchronization of work via JavaScript
- Minimize overall data copies of graph inputs and outputs
- Best-effort buffer-sharing between WebNN and WebGPU
- Allow a tensor to be reused across multiple `MLGraph`s within the same `MLContext`

## Non-Goals

* Guarantee *zero-copy* buffer-sharing between WebNN and WebGPU
* Synchronization of work between WebNN and WebGPU without CPU involvement
* Provide partial views over an `MLTensor`

## Key Scenarios

### Buffer Reuse

A user uploads a large image to a website and wants to try applying several different effects on the same input image.

The current way that WebNN passes input buffers requires a copy to be made _for each inference_.

```js
// Current approach to reuse a given input buffer, requiring many data copies

// Copy the data in `imageAsArrayBuffer` to the required format before executing `graph1`.
// Then, copy the model outputs into `outputArrayBuffer1`.
await mlContext.compute(graph1, {'input': imageAsArrayBuffer}, {'output': outputArrayBuffer1});
// Again, copy all input data in and all output data out.
await mlContext.compute(graph2, {'input': imageAsArrayBuffer}, {'output': outputArrayBuffer2});
// Yet again copy all input data in and all output data out.
await mlContext.compute(graph3, {'input': imageAsArrayBuffer}, {'output': outputArrayBuffer3});
```

`MLTensor` allows tensors and their contents to be reused. Once the image data is written to an `MLTensor`, no further copies are required.

```js
// Proposed approach to reuse a given input buffer, using an input MLTensor

// Copy the image data into the required format.
const imageAsMlTensor = await mlContext.createTensor({..., writable: true});
mlContext.writeTensor(imageAsMlTensor, imageAsArrayBuffer);

// Execute the graphs - no additional copies!
mlContext.dispatch(graph1, {'input': imageAsMlTensor}, {'output': outputMlTensor1});
mlContext.dispatch(graph2, {'input': imageAsMlTensor}, {'output': outputMlTensor2});
mlContext.dispatch(graph3, {'input': imageAsMlTensor}, {'output': outputMlTensor3});

await Promise.all([
  mlContext.readTensor(outputMlTensor1, outputArrayBuffer1);
  mlContext.readTensor(outputMlTensor2, outputArrayBuffer2);
  mlContext.readTensor(outputMlTensor3, outputArrayBuffer3);
]);
```

### Chained Inference

You may notice another benefit of the code snippet above: each call to `dispatch()` does not require an `await`. 

```js
// Current approach to execute models repeatedly, requiring many round-trips to script

// The input and output buffers are transferred. We must wait for the buffers to be returned -
// after the graph is executed and output buffers are copied out - before executing the graph
// with these inputs again.
await mlContext.compute(graph1, {'input': imageAsArrayBuffer}, {'output': outputArrayBuffer1});
// The ML context sits idle while control returns to script...
await mlContext.compute(graph2, {'input': imageAsArrayBuffer}, {'output': outputArrayBuffer2});
// ...which just wants to invoke the ML context again. There has to be a better way!
await mlContext.compute(graph3, {'input': imageAsArrayBuffer}, {'output': outputArrayBuffer3});
```

Using `MLTensor` enables a programming model similar to [WebGPU's](https://www.w3.org/TR/webgpu/#programming-model). Tasks are posted to the ML context's [timeline](#timelines) and are executed as the ML context sees fit - so far as data dependencies are respected such that each `MLTensor` is guaranteed to be modified in the order the methods using the tensor are called from script. In this example, the ML context should be working continuously from the `writeTensor()` call until the work for the last `readTensor()` completes. Better utilization of the ML context will result in significantly better throughput.

```js
// Proposed approach to queue tasks to the ML context timeline

// Post a task to the ML context timeline to allocate and zero out a tensor,
// then return to script.
const imageAsMlTensor = await mlContext.createTensor({..., writable: true});

// Post a task to the ML context timeline to write to the tensor. Note that we do
// not await completion of this write. The ML context will ensure any operations
// which depend on the contents of `imageAsMlTensor` will queue behind this task.
mlContext.writeTensor(imageAsMlTensor, imageAsArrayBuffer);

// Post a task to the ML context timeline to execute the graph. The ML context will
// ensure this queues behind the write above.
mlContext.dispatch(graph1, {'input': imageAsMlTensor}, {'output': outputMlTensor1});
// Post another task. Since input tensors will not be modified by graph execution,
// the ML context may choose to execute in parallel to the dispatch above.
mlContext.dispatch(graph2, {'input': imageAsMlTensor}, {'output': outputMlTensor2});
// Post another task, which may also execute in parallel.
mlContext.dispatch(graph3, {'input': imageAsMlTensor}, {'output': outputMlTensor3});

// Post tasks to read the output tensors. These tasks will queue behind the
// respective dispatch() calls using each tensor.
const outputs = await Promise.all([
    mlContext.readTensor(outputMlTensor1),
    mlContext.readTensor(outputMlTensor2),
    mlContext.readTensor(outputMlTensor3)
  ]);
```

Since the queueing mechanism respects data dependencies, chained inference allows an `MLTensor` to be passed as an output from one graph and then immediately as an input to the next. A collection of graphs and buffers may be repeatedly dispatched without the need for synchronization via script.

```js
// Computing the Nth Fibonacci number using chained inference 

const builder = new MLGraphBuilder(mlContext);
const fn1 = builder.input('F_n-1', {dataType: "int32", shape: [1]});
const fn2 = builder.input('F_n-2', {dataType: "int32", shape: [1]});
const add = builder.add(fn1, fn2);
const graph = await builder.build({'F_n': add});

const descriptors = [
    {dataType: "int32", shape: [1], writable: true},  // To initialize F_0
    {dataType: "int32", shape: [1], writable: true},  // To initialize F_1
    {dataType: "int32", shape: [1]}
];
descriptors[N % 3]['readable'] = true  // To read the output

const tensors = await Promise.all([
    mlContext.createTensor(descriptors[0]),
    mlContext.createTensor(descriptors[1]),
    mlContext.createTensor(descriptors[2])
]);

mlContext.writeTensor(tensors[0], new Int32Array([0]));  // F_0 = 0
mlContext.writeTensor(tensors[1], new Int32Array([1]));  // F_1 = 1

for (let n = 2; n <= N; n++) {
  // Each dispatch depends on tensors used in the previous dispatch.
  mlContext.dispatch(graph,
                     {'F_n-1': tensors[(n-1) % 3], 'F_n-2': tensors[(n-2) % 3]},
                     {'F_n':   tensors[n % 3]});
}

const f_n = new Int32Array(await mlContext.readTensor(tensors[N % 3]))[0];
```

### Resource Management

Let's continue the example above with a large image being used to generate several other large images. Once the user is satisfied with this feature, they want to caption the image using a speech-to-text model. Holding all these tensors and machine learning models at once may put memory pressure on the system.

```js
// Current approach to resource management, relying on GC

await mlContext.compute(graph1, inputs, outputs);
// We're done with `graph1`, which may contain multiple gigabytes of weights...

// ...Let's hope its memory is garbage-collected soon?

// Construct a new graph which itself needs a lot of resources.
// If the system is under memory pressure, this may fail.
const builder = new MLGraphBuilder(mlContext);
const constant = builder.constant(descriptor, veryLargeBufferOfWeights);
```

An `MLTensor`, `MLGraph`, and `MLContext` all have a respective `destroy()` method. Once these objects are no longer needed, the website may request that the memory associated with these resources be released... possibly so that it can run more models!

```js
// Proposed approach to resource management, with explicit destroy methods

mlContext.dispatch(graph1, inputs, outputs);

// We're done with `graph1`, which may contain multiple gigabytes of weights.
// Explicitly ask for its resources to be released!
graph1.destroy();

// We can selectively release only the resources we expect won't be needed
// by calling destroy() on a subset of MLTensors.
destroyBuffers(inputs);
// Don't destroy the output tensors yet, in case we want to reuse them later.

// Construct a new graph which itself needs a lot of resources. Memory pressure
// has been relieved and building this graph is much more likely to succeed. 
const builder = new MLGraphBuilder(mlContext);
const constant = builder.constant(descriptor, veryLargeBufferOfWeights);
```

### WebGPU Interop

A privacy-conscious user wants to perform real-time selfie segmentation of a video feed on their local device.

Currently, using WebNN for this task would require - for each frame - an expensive readback of `GPUBuffer` data to script, uploading the data to the ML context device (which may be the same GPU!), copying the result back to script, and then uploading the frame to be rendered back into a `GPUBuffer`.

An `MLTensor` may be imported into WebGPU, minimizing the number of buffer copies required to render the results of some ML compute. Zero-copy buffer sharing between the two APIs may be supported in some cases.

```js
// Create a couple MLTensors to be shared with WebGPU.
const mlTensor1 = await mlContext.createTensor({..., importableToWebGPU: true});
const mlTensor2 = await mlContext.createTensor({..., importableToWebGPU: true});

const applyEffectToFrame = async () => {
  const gpuVideoTexture = gpuDevice.importExternalTexture({source: video});

  // Wait for all ML work involving `mlTensor1` to complete, then rent it out to WebGPU.
  const tensorizedGpuBuffer = await gpuDevice.importExternalBuffer(mlTensor1);

  // Create a bind group for `gpuVideoTexture`, create a command encoder, etc.
  // to "tensorize" `gpuVideoTexture` and store the result in `tensorizedGpuBuffer`
  // ...

  gpuDevice.queue.submit([tensorizationCommandEncoder.finish()]);

  // Return the buffer to WebNN.
  tensorizedGpuBuffer.release();

  // Perform some inference described by `graph` on the frame
  // (e.g. selfie segmentation)
  mlContext.dispatch(
    graph,
    /*inputs=*/{'input': mlTensor1},
    /*outputs=*/{'output': mlTensor2},
  );

  // Wait for all ML work involving `mlTensor2` to complete, then rent it out to WebGPU.
  const tensorizedGpuBufferAfterInference = await gpuDevice.importExternalBuffer(mlTensor2);

  // Create a bind group for `tensorizedGpuBufferAfterInference`,
  // create a command encoder, etc to feed `tensorizedGpuBufferAfterInference`
  // into a GPU shader which may blur the frame or replace background sections
  // and then render the result
  // ...

  gpuDevice.queue.submit([texturizeAndRenderCommandEncoder.finish()]);

  // Return the buffer to WebNN for the next frame.
  tensorizedGpuBufferAfterInference.release();

  // Call this method for each frame.
  video.requestVideoFrameCallback(applyEffectToFrame);
}
```

## Design Discussion

### Timelines

WebNN uses a programming model similar to [WebGPU's](https://www.w3.org/TR/webgpu/#programming-model), in that compute tasks are posted to a timeline - which I've referred to as an "ML context timeline" throughout this document - separate from the content timeline (i.e. "script"). See [the WebGPU documentation of timelines](https://www.w3.org/TR/webgpu/#programming-model-timelines) for more details.

Specifying WebNN timelines is tracked in [#529](https://github.com/webmachinelearning/webnn/issues/529).

### Device Affinity and Relationship to a `GPUDevice`

The WebNN API requires the developer to declare how an `MLTensor` will be used (via `MLTensorDescriptor`), which the user agent may use as a hint in deciding where to allocate the memory backing an `MLTensor`. Where the memory is ultimately allocated is up to the user agent.

For example [an `MLContext` may be created with a `GPUDevice`](https://www.w3.org/TR/webnn/#dom-ml-createcontext-gpudevice), and creating an `MLTensor` from this context with the `MLTensorDescriptor.importableToWebGPU` flag expresses a clear intention to share the tensor with the given `GPUDevice`. However, there is no guarantee that sharing this tensor with WebGPU will be zero-copy.

The `MLTensorDescriptor.readable` and `MLTensorDescriptor.writable` flags likewise are hints to the user agent indicating that the underlying data will be read and written to, respectively, by script.

### Importing an `MLTensor` to WebGPU

An `MLTensor` created with the `MLTensorDescriptor.importableToWebGPU` flag may be imported as a `GPUBuffer` to a `GPUDevice`. In the best case, this requires no data copies. If the underlying buffer backing the `MLTensor` is not accessible to the `GPUDevice`, this will require copying the contents of the `MLTensor` to a new buffer, then copying the contents of this buffer back to the `MLTensor` once WebGPU releases its handle to the buffer.

While an `MLTensor` is rented to a `GPUDevice`, the `GPUDevice` has exclusive, read/write access to the imported buffer, which is created as a `GPUExternalBuffer` with `GPUBufferUsageFlags.STORAGE`, `GPUBufferUsageFlags.COPY_SRC`, and `GPUBufferUsageFlags.COPY_DST`. All WebNN work depending - directly or indirectly - on the imported `MLTensor` is blocked until the `GPUDevice` returns the tensor.

Importing and returning the `MLTensor` are each points of synchronization between the respective WebNN and WebGPU [timelines](https://www.w3.org/TR/webgpu/#programming-model-timelines). The `importExternalBuffer()` method is asynchronous to allow the user agent to await completion of WebNN operations before posting WebGPU commands with the imported buffer. This is to avoid making WebGPU workloads - which may involve compositing - explicitly dependent on WebNN operations, which may be inefficient (e.g. if ML compute is not expressed in terms of GPU commands) or impossible (e.g. [some platforms don't support enqueuing GPU work that waits on a fence to be later signaled by the CPU](https://github.com/webmachinelearning/webnn/pull/754#discussion_r1740841364)) on some platforms.

### `compute()` vs. `dispatch()`

`compute()` will be deprecated and removed in favor of `dispatch()`.

It's possible `compute()` may have a performance advantage on some platforms for some use cases, such as in one-shot inference (where the benefits of buffer re-use are not relevant) with small inputs/outputs on CPU (where the overhead of task queueing may outweigh the benefits of parallelization). At this time, we do not expect this performance impact to be substantial enough to justify providing two mostly-identical methods for executing an `MLGraph`.

### Open Questions

- How will errors be surfaced? Do we need a concept similar to [WebGPU's error scopes](https://www.w3.org/TR/webgpu/#error-scopes), or is [returning errors via a promise for select operations](https://github.com/webmachinelearning/webnn/issues/697#issuecomment-2195656878) and losing the `MLContext` sufficient? See [#477](https://github.com/webmachinelearning/webnn/issues/477)
- Does the user agent have enough information to appropriately allocate an `MLTensor` if an `MLDeviceType` or `GPUDevice` is not used to create an `MLContext`? See [#350](https://github.com/webmachinelearning/webnn/issues/350) and [#749](https://github.com/webmachinelearning/webnn/issues/749)
- Should the `dispatch()` method be a part of the `MLGraph` interface rather than `MLContext`? Should `readTensor()` and `writeTensor()` exist on an `MLTensor`? See [#697](https://github.com/webmachinelearning/webnn/issues/697).
- Is a sync variant of the `importExternalBuffer()` method feasible (1) on platforms where completion of ML compute can be signaled on a GPU timeline, or (2) when blocking WebGPU workloads which do not themselves block compositing.

## Considered Alternatives

### `MLTensor` as a generic bag of bytes (i.e. `MLBuffer`)

An `MLTensor` was at one point proposed to be a opaque bag of bytes which could be passed as an input or output buffer to a machine learning graph. However, graph inputs and outputs are expressed not as _buffers_ but _tensors_ with a data type and shape. Generic reinterpretation of an opaque buffer is [not a viable approach on platforms which require typed tensors, such as Core ML](https://github.com/webmachinelearning/webnn/issues/542#issuecomment-2067555410). Since the [use case](https://w3ctag.github.io/design-principles/#usecase-oriented-apis) of an `MLTensor` is as a tensor, an `MLTensor` is effectively typed as a tensor.

### Support taking a view over an `MLTensor`

This proposal does not include the ability to arbitrarily reinterpret the attributes or contents of an `MLTensor` - in line with the stance that an `MLTensor` represents a tensor rather than an opaque bag of bytes. Tensor operations such as taking a view over a tensor, reinterpreting a tensor as a new shape, or casting its contents to a new data type map to WebNN's `slice`, `reshape`, and `cast` operators, respectively. To reinterpret an `MLTensor`, these tensor operators may be moved _into the graph itself_ or performed in a separate `MLGraph`.

You may want to write the following code, but taking a view over an `MLTensor` is not supported:

```js
// If creating a slice of an MLTensor in script was supported

const mlTensor = mlContext.createTensor({ dataType: 'int32', shape: [4, 3]});

// Create an input which is half the size of `mlTensor`.
const operandOfDesiredShape = builder.input('a', { dataType: 'int32', shape: [2, 3] });
// ... build the rest of the graph using `operandOfDesiredShape`...

// Pass a view over the top half of the tensor as a graph input.
// NOTE: THIS SUBSCRIPT METHOD DOES THIS EXIST
const mlTensorView = mlTensor[:2];
mlContext.dispatch(graph, {'a': mlTensorView}, outputs);
```

One way to work around this is by inserting a `slice` operation _within the graph_.

```js
// Workaround which inserts a slice operation within the graph itself

const mlTensor = mlContext.createTensor({ dataType: 'int32', shape: [4, 3]});

// Create an input which is exactly the size of `mlTensor`, then add a slice
// operation within the graph itself to get the desired view over the input.
const input = builder.input('a', { dataType: 'int32', shape: [4, 3] });
const operandOfDesiredShape = builder.slice(input, /*starts=*/[0, 0], /*sizes=*/[2, 3])
// ... build the rest of the graph using `operandOfDesiredShape`...

// Pass the MLTensor directly.
mlContext.dispatch(graph, {'a': mlTensor}, outputs);
```

It's possible that demand may emerge for some mechanism to copy data between `MLTensor`s, though that is currently not planned. This may be worked around by creating another graph with an `identity` operation.

### Support passing an `MLTensor` as a `constant()`

There may be an appetite for a mechanism to stream constant weights to the ML context without needing to go through an `ArrayBuffer`, but whether that mechanism should use the `MLTensor` interface is not yet clear. It's not a natural fit because an `MLTensor` is scoped to an `MLContext`, whereas a constant `MLOperand` is scoped to a single `MLGraphBuilder` (and subsequently the `MLGraph` it builds), since sharing constant data across graphs is [not reliably supported on some platforms](https://github.com/webmachinelearning/webnn/issues/614#issuecomment-2021581363).

### Allow mapping an `MLTensor` to script

Why doesn't `MLTensor` have a [`mapAsync()`](https://www.w3.org/TR/webgpu/#dom-gpubuffer-mapasync) method, as a `GPUBuffer` does?

`MLTensor`s are used as inputs and outputs to machine learning models. In practice, we expect the size of the model inputs and outputs to be dwarfed by the size of the model weights, which are uploaded via the `MLGraphBuilder.constant()` method. We may re-evaluate this stance in the future if we discover that reading and writing data to an `MLTensor` is a bottleneck, though even in that case we may prefer to explore a solution which bypasses `ArrayBuffer`s altogether, as discussed [above](#support-passing-an-mltensor-as-a-constant).

### Hash input buffer contents

One approach to solve the [buffer reuse](#buffer-reuse) case is for the WebNN implementation to silently avoid making redundant data copies if the same buffer contents are repeatedly passed as inputs. This may be achieved by hashing the contents of each input. This approach has downsides, such as [managing the extra copies in the hash map](#resource-management) and that hashing the buffer contents may be expensive as it requires reading the entire input. This approach also does not address the other use cases.

## References & Acknowledgements

Many thanks for valuable feedback and advice from:

- Bryan Bernhart
- Joshua Bell
- Mike Wyrzykowski
- Ningxin Hu
- Phillis Tang
- Rafael Cintron
- Reilly Grant
- Zoltan Kis

---

## Appendix

### Tentative IDL

```javascript
dictionary MLTensorDescriptor : MLOperandDescriptor {
  boolean readable = false;
  boolean writable = false;
  boolean importableToWebGPU = false;
};

typedef record<DOMString, MLTensor> MLNamedTensors;

interface MLTensor {
  readonly attribute MLOperandDataType dataType;
  readonly attribute FrozenArray<unsigned long> shape;
  readonly attribute boolean readable;
  readonly attribute boolean writable;
  readonly attribute boolean importableToWebGPU;

  void destroy();
};

partial interface MLContext {
  Promise<MLTensor> createTensor(MLTensorDescriptor descriptor);

  void writeTensor(MLTensor tensor, [AllowShared] ArrayBuffer inputData);
  void writeTensor(MLTensor tensor, [AllowShared] ArrayBufferView inputData);
  
  Promise<ArrayBuffer> readTensor(MLTensor tensor);
  Promise<void> readTensor(MLTensor tensor, [AllowShared] ArrayBuffer outputData);
  Promise<void> readTensor(MLTensor tensor, [AllowShared] ArrayBufferView outputData);
  
  void dispatch(MLGraph graph, MLNamedTensors inputs, MLNamedTensors outputs);
};

// For WebGPU Interop

interface GPUExternalBuffer {
  undefined release();
};
GPUExternalBuffer includes GPUObjectBase;

dictionary GPUExternalBufferDescriptor
         : GPUObjectDescriptorBase {
    required MLTensor source;
};

partial interface GPUDevice {
  Promise<GPUExternalBuffer> importExternalBuffer(GPUExternalBufferDescriptor descriptor);
}

partial interface ML {
  Promise<MLContext> createContext(GPUDevice device);
};
```