# WebNN MLGraph Cache Explainer

## Authors

WebML Working Group participants

## Participate

- https://github.com/webmachinelearning/webnn/issues/807

## Table of contents

1. [Introduction](#introduction)
1. [Goals](#goals)
1. [Non-goals](#non-goals)
1. [User research](#user-research)
1. [Use cases](#use-cases)
1. [Proposed API](#proposed-api)
1. [Considered alternatives](#considered-alternatives)
1. [Related work](#related-work)
1. [Privacy and security considerations](#privacy-and-security-considerations)
1. [References](#references)

## Introduction

The WebNN API enables web applications to perform ML model inference by constructing a graph representation of the model ([`MLGraphBuilder`](https://www.w3.org/TR/webnn/#mlgraphbuilder)), compiling it into a native format ([`MLGraph`](https://www.w3.org/TR/webnn/#mlgraph)), and executing it via [`MLContext.dispatch()`](https://www.w3.org/TR/webnn/#api-mlcontext-dispatch). However, compiling large models for certain devices, such as NPUs, can be time-consuming. This can be particularly difficult since compilation must happen on potentially slower end-user devices rather than ahead-of-time. To address this, we propose an explicit API for caching compiled graphs, allowing web applications to save and reuse them, thereby reducing the overhead of repeated compilation.

This proposal documents ongoing discussions in the W3C WebML Working Group and builds on existing mechanisms in frameworks like ONNX Runtime.

## Goals

- Provide a mechanism for web applications to save and load compiled `MLGraph` objects.
- Reduce the time required for repeated ML model inference by avoiding redundant graph compilation.
- Ensure compatibility with existing WebNN API constructs and workflows.

## Non-goals

- This proposal does not aim to define a universal format for graph serialization across all frameworks.
- It does not address caching mechanisms for non-WebNN APIs or other types of computational graphs.
- Cross-origin model sharing is out of scope.

## User research

[If any user research has been conducted to inform the design choices presented,
discuss the process and findings.
We strongly encourage that API designers consider conducting user research to
verify that their designs meet user needs and iterate on them,
though we understand this is not always feasible.]

## Use cases

### Reduce time to first inference on reload

A web application performing real-time image recognition can save the compiled graph after the first inference. If the page is reloaded, subsequent inferences reuse the cached graph, significantly reducing latency by avoiding both the model redownload and recompilation steps.

## Proposed API

```webidl
partial interface MLContext {
  Promise<sequence<DOMString>> listGraphs();
  Promise<MLGraph> loadGraph(DOMString key);
  Promise<undefined> saveGraph(DOMString key, MLGraph graph);
  undefined deleteGraph(DOMString key);
};
```

1. **`listGraphs()`**: Returns a list of keys for all cached graphs.
2. **`loadGraph(key)`**: Loads a cached graph associated with the given key.
3. **`saveGraph(key, graph)`**: Saves the provided graph under the specified key.
4. **`deleteGraph(key)`**: Deletes the cached graph associated with the given key.

### A note on persistence

A graph may be evicted from the cache due to storage pressure or browser/platform updates which render previously compiled graphs invalid. Developers should consider the level of durability to be somewhere between IndexedDB and the HTTP cache. [For specification purposes, reuse the [Storage standard concepts](https://storage.spec.whatwg.org/#model) as applicable.]

### Input and output descriptors

A JS ML framework, such as ONNX Runtime Web, may need to know the input and output operands info (name, shape and data type) to construct input and output tensors for an inference session. The input and output operands info is known if users pass the source model, e.g. ONNX model. With model cache, user may only pass the model key, the framework needs to fetch the input and output operands info from an `MLGraph`. It would be necessary to expose the `inputDescriptors` and `outputDescriptors` internal slots of `MLGraph` interface.

```webidl
partial interface MLGraph {
  record<USVString, MLOperandDescriptor> inputs;
  record<USVString, MLOperandDescriptor> outputs;
};
```

## Considered alternatives

### Combined build and save

A separate `saveGraph()` API might introduce overhead on some native ML frameworks, such as ONNX Runtime, because its implementation may need to hold the source model in the memory and recompile the source model when user code calls `saveGraph()`.

An alternative consideration is to have a `buildAndSave()` method. The implementation can just compile the graph once and drop the source model after the compilation.

```webidl
partial interface MLGraphBuilder {
  Promise<MLGraph> buildAndSave(MLNamedOperands outputs, DOMString key);
};
```

However, a compliant implementation of `build()` could save the compiled model into a temporary file which is deleted unless `saveGraph()` is called later, rendering an explicit `buildAndSave()` unnecessary.

### Explicit vs implicit API

GPU shader caching is implicit, however the difference is that a shader program is a small input and so it's easy for the site to regenerate the shader so the browser can hash it to compare with the cache. ML models on the other hand are large because of the weights. Loading all the weights just to discover that a cached version of the model is available would be a waste of time and resources. (via [comment](https://github.com/webmachinelearning/webnn/issues/807#issuecomment-2608135598))

Furthermore, an ML model can't be compiled without the weights because the implementation may perform device-specific constant folding and memory layout optimizations.

## Related work

### ONNX Runtime EPContext

ONNX Runtime introduced the `EPContext` mechanism to encapsulate compiled blobs into ONNX models. This approach inspired the WebNN caching proposal but is tailored to ONNX-specific workflows.

### WebGPU shader cache

The WebGPU API employs a shader caching mechanism. While similar in concept, it is designed for GPU shaders rather than ML model graphs.

## Privacy and security considerations

### Storage partitioning

To prevent cross-origin data leakage, cached graphs must be partitioned per origin. This ensures that a graph saved by one website cannot be accessed by another.

### Implementation-specific sandbox constraints

For security reasons, model compilation and inference will typically happen in sandboxed processes. This will introduce implementation challenges and care must be taken in how the caching mechanism allows data to be read from and written to disk.

## References

- [WebNN API Specification](https://github.com/webmachinelearning/webnn)
- [ONNX Runtime EPContext Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html#onnxruntime-ep-context-cache-feature-design)
- [OpenVINO Model Caching Overview](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html)
- [Chromium Sandbox Design](https://chromium.googlesource.com/chromium/src/+/main/docs/design/sandbox.md)
- [WebGPU Shader Cache](https://docs.google.com/document/d/1CtgsUWTBe6pVEDq3ZksSEc_6eSAqvHZ-h0_zoPu21po/edit?tab=t.0#heading=h.fshi85nj57x0)
