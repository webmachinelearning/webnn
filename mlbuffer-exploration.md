# `MLBuffer` Exploration

By @a-sully

## What is this?

This is an exploration - primarily via code samples of key use cases - of what
ML compute might look like using a device-agnostic buffer, as proposed in
[#482](https://github.com/webmachinelearning/webnn/issues/482) as `MLBuffer`.

This is not intended to be a formal explainer, though it could become one if
that would be useful. My intention here is to describe our priorities (such that
we can ensure the design satisfies these priorities), bring attention to some
open questions and related issues, toss around some ideas, and encourage
discussion about how this proposal will be specified.

## Goals

- Minimize round-trips to JavaScript/CPU needed for synchronization of work on
  buffers which may not live on the CPU
- Minimize buffer copies
  - In particular, we should support zero-copy buffer sharing between WebNN and
    WebGPU if this is supported by the underlying hardware
- Support the XPU (i.e. CPU, GPU, NPU, TPU, etc...) with one consistent API
- Follow recomended [design
  principles](https://w3ctag.github.io/design-principles/)
  - In my opinion, this likely entails [mirroring WebGPU's design
    decisions](https://w3ctag.github.io/design-principles/#naming-consultation),
    where appropriate

## Overarching Questions

Many of these questions are not _specific_ to `MLBuffer`, but are important
enough that their answers will strongly influence the shape of the `MLBuffer`
proposal.

- What are WebNN's timelines and how do they interact with WebGPU's timelines?
  See [#529](https://github.com/webmachinelearning/webnn/issues/529)
- Where will an `MLBuffer`'s memory be allocated on systems where an `MLContext`
  may not be as closely tied to a given physical device as an
  [`IDMLDevice`](https://learn.microsoft.com/en-us/windows/win32/api/directml/nn-directml-idmldevice)?
  See [#350](https://github.com/webmachinelearning/webnn/issues/350)
- How will errors be surfaced? See
  [#477](https://github.com/webmachinelearning/webnn/issues/477). Do we need a
  concept similar to [WebGPU's error
  scopes](https://www.w3.org/TR/webgpu/#error-scopes)?
- Must an `MLBuffer` only be used with an `MLContext` it was created from?
  (or `MLGraph`s created from that `MLContext`, and so forth)
- If what we're building is a device-agnostic buffer, it will surely be used for
  things other than ML (in the long run). In the spirit of
  [future-proofing](https://w3ctag.github.io/design-principles/#naming-future-proofing),
  should we name it something other than `MLBuffer`?

## Use Case: Chained Inference

Here's a code sample showing how `MLBuffer`s can be used for chained inference
and then read back to an `ArrayBuffer`:

```js
// Create new MLBuffers to be used for chained inference.
const inputMlBuffer = mlContext.createBuffer({inputSize});
const intermediateMlBuffer = mlContext.createBuffer({intermediateSize});
const outputMlBuffer = mlContext.createBuffer({outputSize});

// Copy the contents of an ArrayBuffer into an MLBuffer, to be later used as inputs.
mlContext.writeBuffer(
    inputMlBuffer,
    /*dstOffset=*/0,
    /*srcData=*/someJsArrayBuffer,
);

// Perform some ✧*✧* machine learning *✧*✧ described by `graph`.
mlContext.dispatch(
    graph,
    /*inputs=*/{buffer: inputMlBuffer},
    /*outputs=*/{buffer: intermediateMlBuffer},
);

// Feed the output of one execution as the input to the next. Chained inference!
mlContext.dispatch(
    anotherGraph,
    /*inputs=*/{buffer: intermediateMlBuffer},
    /*outputs=*/{buffer: outputMlBuffer},
);

// Read back the results to script.
const resultBuffer = await outputMlBuffer.mapAsync();
```

Let's dive into what happens at each of these steps:

### `MLBuffer` creation

```js
const inputMlBuffer = mlContext.createBuffer({inputSize});
```
#### How it works:

- Enqueue a request on some WebNN timeline to allocate memory on the device
  associated with `mlContext`
- The memory allocation will be zeroed (as it is for [WebGPU's `createBuffer()`
  method](https://www.w3.org/TR/webgpu/#dom-gpudevice-createbuffer))

#### Questions:

- Can an `MLBuffer`'s size always be known at the time of buffer allocation?
  - In this case and many other cases it seems possible; it's presumably a
    function of the model and/or video input. But since WebNN always rents a
    buffer to WebGPU - never the other way around - this introduces a constraint
    that the size of an `MLBuffer` must always be known at the time of buffer
    allocation
- When will `inputMlBuffer` be deallocated if `destroy()` is not called?

### Writing to an `MLBuffer`

```js
mlContext.writeBuffer(
    inputMlBuffer,
    /*dstOffset=*/0,
    /*srcData=*/someJsArrayBuffer,
);
```

#### How it works:

- Enqueue a request on some WebNN timeline to copy the contents of
  `someJsArrayBuffer` to `inputMlBuffer`. This is very similar to [the
  corresponding WebGPU
  method](https://www.w3.org/TR/webgpu/#dom-gpuqueue-writebuffer), though the
  implementation details will vary depending on which device `inputMlBuffer` is
  allocated on. For example, if allocated on:
  - a CPU, the buffer contents will be copied directly (i.e. `memcpy()`)
  - a GPU, the behavior will likely match `GPUQueue.writeBuffer()`. On UMA
    systems, a `memcpy()` might suffice. Other implementations may use a hidden
    "upload" buffer to get the data onto the GPU. This implies two copies:\
    *&emsp;&emsp;`ArrayBuffer` &rarr; "upload" buffer &rarr; high-GPU-bandwidth
    buffer*
  - an XPU... it depends!
- `someJsArrayBuffer` is unaffected, since the bytes are copied
  - Note that the aforementioned copies are _in addition_ to any copies needed
    to get the data into the `ArrayBuffer` in the first place. If the data is
    weights being read from a `File`, for example, this will require first
    copying the bytes from the `File` into the `ArrayBuffer`. This means
    **copying the weights into GPU-accessible memory could take as many as four
    copies!**

#### Questions:

- Should there be a corresponding
  [`mappedAtCreation`](https://www.w3.org/TR/webgpu/#dom-gpubufferdescriptor-mappedatcreation)
  capability?
  - If the data is not already in an `ArrayBuffer`, this eliminates the data
    copy into an `ArrayBuffer` altogether, since we could write to the "upload"
    buffer directly:
    ```js
    const mlBuffer = mlContext.createBuffer({size, mappedAtCreation: true});

    const floatArray = new Float32Array(mlBuffer.getMappedRange()),

    // Write to `floatArray`
    // ...

    // Write the buffer contents to the XPU
    mlBuffer.unmap();
    ```
    &emsp;&emsp;Before: *some source &rarr; `ArrayBuffer` &rarr; "upload" buffer
      &rarr; high-GPU-bandwidth buffer*\
    &emsp;&emsp;After: *some source &rarr; "upload" buffer
      &rarr; high-GPU-bandwidth buffer*
- Should there be the equivalent of
  [`MAP_WRITE`](https://www.w3.org/TR/webgpu/#dom-gpubufferusage-map_write) +
  [`COPY_SRC`](https://www.w3.org/TR/webgpu/#dom-gpubufferusage-copy_src) for
  `MLBuffer`s?
  - If we know the usage of `inputMlBuffer` (e.g. that it's read-only by WebNN)
    then we may be able to eliminate the data copy from the "upload" buffer to
    the high-GPU-bandwidth buffer in the non-UMA case:
    ```js
    const mlBuffer = mlContext.createBuffer({size, usage: MAP_WRITE | INPUT_ONLY});
    ```
    This may not make a difference for DirectML, which appears to require [bound
    resources](https://learn.microsoft.com/en-us/windows/win32/api/directml/nf-directml-idmlbindingtable-bindpersistentresource)
    to use `D3D12_HEAP_TYPE_DEFAULT`, but it could eliminate a copy on other
    systems. I'm not familiar enough with other systems to know the answer here!
  - Combining this with the above techniques brings (as many as) 4 copies down
    to (as few as) 2:\
    &emsp;&emsp;Before: *some source &rarr; `ArrayBuffer` &rarr; "upload" buffer
      &rarr; high-GPU-bandwidth buffer*\
    &emsp;&emsp;After: *some source &rarr; "upload" buffer*

### Execute an `MLGraph`

```js
mlContext.dispatch(
    graph,
    /*inputs=*/{buffer: inputMlBuffer},
    /*outputs=*/{buffer: intermediateMlBuffer},
);
```

#### How it works:

- Enqueues a request to compute the graph onto some WebNN timeline
- Execution cannot start until all input and output `MLBuffer`s are available
- All input and output `MLBuffer`s are unavailable while execution is in
  progress
- All work submitted after this `dispatch()` call which relies on an input or
  output `MLBuffer` will be queued behind this execution

#### Questions:

- This approach is flexible enough to allow for graph execution on all backends.
  Do we need a separate `compute()` method?
- Should this method be on the `MLGraph` (related to
  [#303](https://github.com/webmachinelearning/webnn/issues/303))? Is there a
  use case not satisfied by the following?
  ```js
  graph.dispatch(
    /*inputs=*/{buffer: inputMlBuffer},
    /*outputs=*/{buffer: intermediateMlBuffer},
  );
  ```
- Is it valid to pass the same `MLBuffer` as both an input and output of the
  same `dispatch()` call? e.g.
  ```js
  graph.dispatch(
    /*inputs=*/{buffer: someMlBuffer},
    /*outputs=*/{buffer: someMlBuffer},
  );
  ```

### Read back data from an `MLBuffer`

```js
const resultBuffer = await outputMlBuffer.mapAsync();
```

#### How it works:

- After the completion of all currently-enqueued operations that use
  `outputMlBuffer`, WebNN will copy the contents of `outputMlBuffer` to
  `resultBuffer`. This is very similar to
  [`GPUBuffer.mapAsync()`](https://www.w3.org/TR/webgpu/#dom-gpubuffer-mapasync),
  with a key difference being that WebGPU only allows `mapAsync()` on buffers
  which have the `MAP_READ` usage flag. In this case, if using a GPU, we may
  need to create an intermediate "readback" buffer to facilitate the transfer.
  This may require two copies:\
  *&emsp;&emsp;high-GPU-bandwidth buffer &rarr; "readback" buffer &rarr;
  `ArrayBuffer`*\

#### Questions:

- What should this method be called? I've proposed `mapAsync()` here to mirror
  WebGPU since the behavior is very similar.
- Should there be the equivalent of
  [`MAP_READ`](https://www.w3.org/TR/webgpu/#dom-gpubufferusage-map_read) +
  [`COPY_DST`](https://www.w3.org/TR/webgpu/#dom-gpubufferusage-copy_dst) for
  `MLBuffer`s?
  - If we know the usage of `outputMlBuffer` (e.g. that it's
    write-once by WebNN) then we could eliminate the data copy from the
    high-GPU-bandwidth buffer to the "readback" buffer in the non-UMA case
    ```js
    // The buffer may be allocated on a "readback" buffer
    const mlBuffer = mlContext.createBuffer({size, usage: MAP_READ | OUTPUT_ONLY});

    // `mlBuffer` may be used as an output to MLGraph execution
    // ...

    // Read back with fewer data copies!
    const resultBuffer = await mlBuffer.mapAsync();
    ```
    Again, this will not help DirectML and may or may not help other systems.

## Use Case: WebGPU Interop

Here’s a code example in which WebNN performs selfie segmentation on a video
frame without needing round-trips to JavaScript to synchronize WebNN and WebGPU
compute:

```js
const applyEffectToFrame = () => {
  const gpuVideoTexture = gpuDevice.importExternalTexture({source: video});

  // Create a new MLBuffer to be used to facilitate WebGPU interop.
  //
  // Note that a more optimized implementation might allocate this buffer - or a
  // ring of buffers - ahead of time such that memory can be reused.
  const tensorizedMlBuffer = mlContext.createBuffer({size: tensorizedBufferSize});

  // Rent out the MLBuffer to WebGPU.
  const tensorizedGpuBuffer = tensorizedMlBuffer.mapAsGpuBuffer(gpuDevice);

  // Create a bind group for `gpuVideoTexture`, create a command encoder, etc.
  // to "tensorize" `gpuVideoTexture` and store the result in `tensorizedGpuBuffer`
  // ...

  gpuDevice.queue.submit([tensorizationCommandEncoder.finish()]);

  // Return the buffer to WebNN.
  tensorizedMlBuffer.unmapFromGpuBuffer();

  // Perform some inference described by `graph` on the frame
  // (e.g. selfie segmentation)
  mlContext.dispatch(
    graph,
    /*inputs=*/{buffer: tensorizedMlBuffer},
    /*outputs=*/{buffer: tensorizedMlBuffer},
  );

  // Rent the MLBuffer back out to WebGPU.
  const tensorizedGpuBufferAfterInference = tensorizedMlBuffer.mapAsGpuBuffer(gpuDevice);

  // Create a bind group for `tensorizedGpuBufferAfterInference`,
  // create a command encoder, etc to feed `tensorizedGpuBufferAfterInference`
  // into a GPU shader which may blur the frame or replace background sections
  // and then render the result
  // ...

  gpuDevice.queue.submit([texturizeAndRenderCommandEncoder.finish()]);

  // Call this method for each frame.
  video.requestVideoFrameCallback(applyEffectToFrame);
}
```

Let's again dive into what happens at each of these steps. Some of these steps
are covered above, which I'll skip over here:

### Rent out an `MLBuffer` to WebGPU

```js
const tensorizedGpuBuffer = tensorizedMlBuffer.mapAsGpuBuffer(gpuDevice);
```

#### How it works:

- Two fences are created:
  1. a "start access" fence which is to be signaled by WebNN and waited on by
     WebGPU
  2. an "end access" fence which is to be signaled by WebGPU and waited on by
     WebNN
- `gpuDevice` enqueues a command to its `GPUQueue` to wait for the "start
  access" fence to be signaled
- WebNN (on some queue or timeline yet to be specified) will signal the "start
  access" fence after the completion of all currently-enqueued operations that
  use `tensorizedMlBuffer`. This is very similar to how `mapAsync()` works
  - In this case, there is only one currently-enqueued operation:
    `MLContext.createBuffer()`
  - In the latter `mapAsGpuBuffer()` call, the "start access" fence will not be
    signaled by WebNN until the `dispatch()` call is complete. This implicitly
    blocks execution of the commands in `texturizeAndRenderCommandEncoder` that
    are enqueued to WebGPU until WebNN is finished with `tensorizedMlBuffer`
- WebNN will wait for the "end access" fence to be signaled. In the meantime,
  all work involving `tensorizedMlBuffer` is blocked
- `gpuDevice` has exclusive, read/write access to this memory for as long as the
  "end access" fence is not signaled
- If `tensorizedMlBuffer` was allocated in memory shared by `gpuDevice`, this
  will be a zero-copy mapping. Otherwise a new buffer will be allocated on
  `gpuDevice` and the contents of `tensorizedMlBuffer` will be copied into this
  buffer
- The memory backing `tensorizedMlBuffer` becomes inaccessible to WebNN (or
  script, or anything else), regardless of whether a copy is made.
  - Ideally these states and their transitions can be expressed similarly to a
    `GPUBuffer`'s [internal
    state](https://www.w3.org/TR/webgpu/#buffer-internals-state)

#### Questions:

- What are the usage flags of `tensorizedGpuBuffer`?
- While `tensorizedMlBuffer` is rented out to WebGPU as `tensorizedGpuBuffer`:
  - What happens if `destroy()` is called on `tensorizedMlBuffer`?
  - What happens if `destroy()` is called on `tensorizedGpuBuffer`?

### Return a rented-out `MLBuffer` back to WebNN

```js
tensorizedMlBuffer.unmapFromGpuBuffer();
```

#### How it works:

- If `tensorizedMlBuffer` was allocated in memory shared by `gpuDevice`, this
  will be a zero-copy unmapping. Otherwise the contents of `tensorizedGpuBuffer`
  will be copied into `tensorizedMlBuffer`
- Informs `gpuDevice` to signal the "end access" fence created in the
  `mapAsGpuBuffer()` method after the completion of currently-enqueued
  operations that use `tensorizedGpuBuffer`. This is very similar to how
  `mapAsync()` works
- The WebNN timeline receives the signal and may resume execution
- WebNN has exclusive, read/write access to this memory until further notice
- `tensorizerGpuBuffer` is expired
  https://gpuweb.github.io/gpuweb/#dom-gpuexternaltexture-expired-slot

#### Questions:

- What happens to `tensorizedMlBuffer` if this method is never called?
