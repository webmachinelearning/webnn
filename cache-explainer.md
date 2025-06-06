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

The WebNN API enables web applications to perform ML model inference by constructing a graph representation of the model ([`MLGraphBuilder`](https://www.w3.org/TR/webnn/#mlgraphbuilder)), compiling it into a native format ([`MLGraph`](https://www.w3.org/TR/webnn/#mlgraph)), and executing it via [`MLContext.dispatch()`](https://www.w3.org/TR/webnn/#api-mlcontext-dispatch). However, compiling large models on certain devices, such as NPUs, can be time-consuming. To address this, we propose an explicit API for caching compiled graphs, allowing web applications to save and reuse them, thereby reducing the overhead of repeated compilation.

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

### Reduce time to first inference

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

## Considered alternatives

### Explicit vs implicit API

>GPU shader caching is implicit, however the difference is that a shader program is a small input and so it's easy for the site to regenerate the shader so the browser can hash it to compare with the cache. ML models on the other hand are large because of the weights. Loading all the weights just to discover that a cached version of the model is available would be a waste of time and resources. (via [comment](https://github.com/webmachinelearning/webnn/issues/807#issuecomment-2608135598))

## Related work

### ONNX Runtime EPContext

ONNX Runtime introduced the `EPContext` mechanism to encapsulate compiled blobs into ONNX models. This approach inspired the WebNN caching proposal but is tailored to ONNX-specific workflows.

### WebGPU shader cache

The WebGPU API employs a shader caching mechanism. While similar in concept, it is designed for GPU shaders rather than ML model graphs.

## Privacy and security considerations

### Storage partitioning

To prevent cross-origin data leakage, cached graphs must be partitioned per origin. This ensures that a graph saved by one website cannot be accessed by another.

### Implementation-specific sandbox constraints

Chromium's sandboxing model restricts file system access for GPU processes. Any implementation must comply with these constraints to ensure security.

## References

- [WebNN API Specification](https://github.com/webmachinelearning/webnn)
- [ONNX Runtime EPContext Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html#onnxruntime-ep-context-cache-feature-design)
- [OpenVINO Model Caching Overview](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html)
- [Chromium Sandbox Design](https://chromium.googlesource.com/chromium/src/+/main/docs/design/sandbox.md)
- [WebGPU Shader Cache](https://docs.google.com/document/d/1CtgsUWTBe6pVEDq3ZksSEc_6eSAqvHZ-h0_zoPu21po/edit?tab=t.0#heading=h.fshi85nj57x0)
