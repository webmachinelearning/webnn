# Device Selection Explainer

## Participate

Feedback on this explainer is welcome via the issue tracker:

- https://github.com/webmachinelearning/webnn/labels/device%20selection - existing discussions about device selection mechanisms
- see also [all issues](https://github.com/webmachinelearning/webnn/issues) and feel free to open a new issue as appropriate

## Introduction

This explainer summarizes the discussion and background on [WebNN device selection](https://webmachinelearning.github.io/webnn/#programming-model-device-selection).

The goal is to help make design decisions on how to handle compute device selection for a WebNN [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext).

A context represents the global state of WebNN model graph execution, including the compute devices (e.g., CPU, GPU, NPU) on which the [WebNN graph](https://webmachinelearning.github.io/webnn/#mlgraph) is executed.

When creating a context, an application may want to provide hints to the implementation on what device(s) are preferred for execution.

Implementations, browsers, and the underlying OS may want to control the allocation of compute devices for various use cases and system conditions.

The question is who should be able to, and to what extent, control the execution context state and capabilities.

This was previously captured by context options including `deviceType` (`"cpu"`, `"gpu"`, `"npu"`) and `powerPreference` (`"high-performance"`, `"low-power"`). While `deviceType` has been removed as a direct option for context creation, `powerPreference` remains a key hint for implementations.

For more background on prior discussions, check out the [History](#history) section.

## Key use cases and requirements

To highlight the tensions between developer intents, API capabilities, and platform support, consider the following developer scenario. Before downloading a model (e.g. from [Hugging Face](https://huggingface.co/)), a developer wants to know how the model can be run with the WebNN implementation on a given client platform (e.g. on GPU or NPU).

One use case is that if the model cannot be accelerated on GPU or NPU, then don't execute on CPU (i.e. prevent CPU fallback).

After that, one option is to allow WebNN to silently defer inference to other means (e.g. by using [WebGPU](https://www.w3.org/TR/webgpu/)), if that is supported by the particular implementation.

Another option is to just ask for an error in this case, then the developer would take control and do inference by other means.

However, there are platforms on which preventing CPU fallback cannot be implemented since there always is an automatic CPU fallback. In those cases, another API, e.g. a capability introspection interface might be more useful.

In addition, the developer may inquire capabilities inferred from collected historical information about running models on the given client. Note that the client platform may choose any accelerators in any combination and sequence, depending on actual system conditions that may change between runs.

Also, a developer may provide hints which may be silently (no feedback) or explicitly (with feedback) overridden by the client platform.

The emerging main use cases are the following.

### 1. Pre-download capability check
Before downloading a model, determine if the specific model can be used for inference as expected.
- This may need a prior model introspection step (out of scope for this document), e.g. checking quantization, data types, memory/buffering requirements, etc. for the model. To obtain this data, one can check for instance `config.json`, `model_card.md`, metadata embedded in the models, model file name conventions, tags, library compatibility and specific target optimizations, example code/notebooks, etc.
- The obtained model information may be compared against the local capabilities queried by an API in order to determine if the model is suitable.

**Requirement**: need an API for capability query / capability matching between models and platform.
Possible means:
- use an explicit capability query API, such as "is acceleration available" (meaning GPU or NPU), "what data types are supported", "is this data type supported", "are these operators supported", "is this quantization supported", or "tell me all local capabilities" etc. The exact API shape is to be determined.
- use collected historical data to provide on query a high level capability overview. This is justified by the need of knowing fast the capabilities also on platforms that need to actually dispatch a graph before being able to tell what is supported.

Note that these have been discussed in [Query mechanism for supported devices #815](https://github.com/webmachinelearning/webnn/issues/815) and this [proposal](https://github.com/webmachinelearning/webnn/issues/749#issuecomment-2429821928), also tracked in [MLOpSupportLimits should be opt-in #759](https://github.com/webmachinelearning/webnn/issues/759). This is to allow listing operator support limits outside of a context, which would return all available devices with their operator support limits. Then, the web app could choose one of them to initialize a context.

### 2. Pre-download or pre-build hints and constraints

**Requirement**: support for context creation hints and constraints (e.g. limit fallback scenarios).
Possible means:
- identify hints/constraints that may be silently overridden by implementations, e.g. "low-power", "high-performance", "low-latency", etc.
- identify hints/constraints that require a feedback (error) if not supported, for instance "avoid CPU fallback" or "need low power and low latency acceleration".

### 3. Post-compile query of inference details
**Requirement**:
- Query a compiled graph for details on how may it be run (subject to being overridden by the platform).
- Query if CPU fallback is active for a context.

This is being discussed in [Get devices used for a graph after graph compilation #836](https://github.com/webmachinelearning/webnn/issues/836)
and being explored in PR [#854 (define graph.devices)](https://github.com/webmachinelearning/webnn/pull/854).
Initially, the proposal was to obtain the list/combination of devices usable for running the graph, but the utility of this needs to be proven. However, this requirement covers querying information on a graph in general, with the details to be determined later.

## Design considerations

Design decisions may take the following into account:

1. Allow the underlying platform to ultimately choose the appropriate compute device(s).

2. Allow scripts to express hints/options when creating contexts, such as preference for low power consumption, high performance (throughput), low latency, stable sustained performance, accuracy, etc.

3. Allow an easy way to create a context with a GPU device, i.e., without specifying an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) (e.g., via `powerPreference`).

4. Allow selection from available GPU devices, for instance, by allowing specification of an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) obtained from available [GPUAdapters](https://gpuweb.github.io/gpuweb/#gpuadapter) using [WebGPU](https://gpuweb.github.io/gpuweb) mechanisms via [GPURequestAdapterOptions](https://gpuweb.github.io/gpuweb/#dictdef-gpurequestadapteroptions), such as feature level or power preference.

5. Allow selection from available various AI accelerators, including NPUs, GPUs or a combination of accelerators. This may happen using a (to-be-specified) algorithmic mapping from context options. Or, allow web apps to hint a preferred fallback order for the given context, or fallbacks to avoid (if that is supported). (Related to discussions in Issue #815).

6. Add a context creation option/hint for telling app preference for being simply ["accelerated"](https://github.com/webmachinelearning/webnn/issues/815#issuecomment-2658627753), meaning NPU, GPU or both.

7. Allow enumeration of [OpSupportLimits](https://webmachinelearning.github.io/webnn/#api-mlcontext-opsupportlimits-dictionary) before creating a context so that web apps can select the best device that would work with the intended model. This needs more developer input and examples. (Related to discussions in Issue #815).

8. As a corollary to 6, allow creating a context using options for [OpSupportLimits](https://webmachinelearning.github.io/webnn/#api-mlcontext-opsupportlimits-dictionary). (Related to discussions in Issue #815).

9. Expose a context property (or event) to tell whether CPU fallback is active (or likely active) for the context.


## Scenarios, examples, design discussion

Examples for user scenarios:

```js
// simple context creation with implementation defaults
context = await navigator.ml.createContext();

// create a context that will likely map to an NPU or NPU+CPU
context =  await navigator.ml.createContext({powerPreference: 'low-power'});

// create a context that will likely map to GPU
context = await navigator.ml.createContext({powerPreference: 'high-performance'});

// create a context that should use massive parallel processing (e.g. GPU/NPU)
context = await navigator.ml.createContext({accelerated: true});
if (context.accelerated) {
    // the context will mostly use GPU/NPU, but CPU fallback may happen
} else {
    // the platform tells it likely cannot provide NPU or GPU, so try something else
}

// create a context that should preferably use NPU
context = await navigator.ml.createContext({accelerated: true, powerPreference: 'low-power'});
if (context.accelerated) {
    // NPU is likely used -- further requirements could be set by opSupportLimitsPerDevice
} else {
    // NPU is likely not available, and since GPU needs high power, it is not used
}

// enumerate devices and limits (as allowed by policy/implementation)
// and select one of them to create a context
const limitsMap = await navigator.ml.opSupportLimitsPerDevice();
// analyze the map and select an op support limit set
// ...
const context = await navigator.ml.createContext({
    limits: limitsMap['npu1']
});
(Note: `opSupportLimitsPerDevice()` and using `limits` in `createContext()` are illustrative of a potential future API being discussed in Issue #815 and are not yet standardized features.)

// as an alternative, hint a preferred fallback order ["npu", "cpu"]
// i.e. try executing the graph on NPU and avoid GPU as much as possible
// but do as it's best fit with the rest of the context options
const context = await navigator.ml.createContext({ fallback: ['npu', 'cpu'] });
(Note: The `fallback` option here is a hypothetical example for discussion and not a current standard option.)

```

## Open questions

- WebGPU provides a way to select a GPU device via [GPUAdapter](https://gpuweb.github.io/gpuweb/#gpuadapter). Should WebNN expose a similar adapter API for NPUs? The current take is to not expose explicit adapters.

- How should WebNN extend the context options? What exactly is best to pass as context options? Operator support limits? Supported features, similar to [GPUSupportedFeatures](https://gpuweb.github.io/gpuweb/#gpusupportedfeatures)? Others?

- Concerning security and privacy, would the proposals here increase the fingerprinting surface? If so, what mitigations can be made? The current understanding is that any extra information exposed to web apps in these proposals could be obtained by other methods as well. However, security hardening and relevant mitigations are recommended. For instance, implementations could choose the level of information (e.g., operator support limits) exposed to a given origin.

- How should the API expose which device(s) a compiled graph is actually utilizing, to allow developers to adapt if the allocation is suboptimal (see Issue #836 and PR #854)?

## Considered alternatives

1. Keep the `MLDeviceType` enumeration (see [CRD 20250131](https://www.w3.org/TR/2025/CRD-webnn-20250131/#enumdef-mldevicetype)) as a context option, but improve the device type names and specify an algorithm for mapping these names to various real adapters (with their given characteristics). However, this would be more limited than being able to specify device-specific limits for context creation. (This was the approach prior to PR #809).

2. Remove `MLDeviceType`, but define a set of [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions) that map well to GPU adapter/device selection and also to NPU device selection. (This is the current approach, implemented in PR #809).

3. Follow this [proposal](https://github.com/webmachinelearning/webnn/issues/749#issuecomment-2429821928), also tracked in [MLOpSupportLimits should be opt-in #759](https://github.com/webmachinelearning/webnn/issues/759). This is to allow listing operator support limits outside of a context, which would return all available devices with their operator support limits. Then, the web app could choose one of them to initialize a context. (This is a suggested longer-term discussion topic.)

For extending the context options, consider also e.g. the following.

### Device Preference Options in ONNX Runtime

A WebNN application may have specific device preferences for model execution. The following use cases map to such preferences, informed by existing APIs such as ONNX Runtime's [`OrtExecutionProviderDevicePolicy`](https://onnxruntime.ai/docs/api/c/group___global.html#gaf26ca954c79d297a31a66187dd1b4e24):

*   **Prefer execution on the main CPU**:
    *   *Preference*: `"prefer CPU"`
    *   *Description*: The application developer hints that the model should ideally run on the device component primarily responsible for general computation, typically "where JS and Wasm execute." This could be due to the model's characteristics (e.g., heavy control flow, operations best suited for CPU) or to reserve other accelerators for different tasks.
*   **Prefer execution on a Neural Processing Unit (NPU)**:
    *   *Preference*: `"prefer NPU"`
    *   *Description*: The application developer hints that the model is well-suited for an NPU. NPUs are specialized hardware accelerators, distinct from CPUs (typically "where JS and Wasm execute") and GPUs (typically "where WebGL and WebGPU programs execute"). In a future-proof context, NPUs fall under the category of "other" compute devices, encompassing various current and future specialized ML accelerators. This preference is often chosen for models optimized for low-power and sustained performance.
*   **Prefer execution on a Graphics Processing Unit (GPU)**:
    *   *Preference*: `"prefer GPU"`
    *   *Description*: The application developer hints that the model should run on the GPU (the device "where WebGL and WebGPU programs execute"). This is common for models with highly parallelizable operations.
*   **Maximize Performance**:
    *   *Preference*: `"maximum performance"`
    *   *Description*: The application developer desires the highest possible throughput or lowest latency for the model execution, regardless of power consumption. The underlying system will choose the device or combination of devices (e.g., "where WebGL and WebGPU programs execute" or other specialized hardware) that can achieve this.
*   **Maximize Power Efficiency**:
    *   *Preference*: `"maximum efficiency"`
    *   *Description*: The application developer prioritizes executing the model in the most power-efficient manner, which might involve using an NPU or a low-power mode of the CPU ("where JS and Wasm execute"). This is crucial for battery-constrained devices or long-running tasks.
*   **Minimize Overall System Power**:
    *   *Preference*: `"minimum overall power"`
    *   *Description*: The application developer hints that the model execution should contribute as little as possible to the overall system power draw. This is a broader consideration than just the model's own efficiency, potentially influencing scheduling and resource allocation across the system. The implementation may choose any device ("where JS and Wasm execute," "where WebGL and WebGPU programs execute," or "other") that best achieves this goal.


## Minimum Viable Solution (MVS, completed)

Based on the discussion above, the best starting point was a simple solution that can be extended and refined later. A first contribution could include the following changes:
- Remove `MLDeviceType` (see [CRD 20250131](https://www.w3.org/TR/2025/CRD-webnn-20250131/#enumdef-mldevicetype)) as an explicit [context option](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions).
- Update `MLContext` so that it becomes device-agnostic, or a _default_/_generic_ context. Allow supporting multiple devices with one context.
- Add algorithmic steps or notes for implementations on how to map `powerPreference` to devices.
- Also, to align with `GPUPowerPreference`, remove the `"default"` `MLPowerPreference` value, i.e., the lack of hints will result in creating a generic context.

This was implemented in [Remove MLDeviceType #809](https://github.com/webmachinelearning/webnn/pull/809).

Besides, the following topics have been discussed:
- Improve the device selection hints in [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions) and define their implementation mappings. For instance, discuss whether to also include a `"low-latency"` performance option.
- Document the valid use cases for requesting a certain device type or combination of devices, and under what error conditions. Currently, after these changes, there remains explicit support for a GPU-only context when an `MLContext` is created from a `GPUDevice` in `createContext()`.
- Discuss option #3 from [Considered alternatives](#considered-alternatives).

## Next discussion phase after MVS

In [Remove MLDeviceType #809](https://github.com/webmachinelearning/webnn/pull/809), this [comment](https://github.com/webmachinelearning/webnn/pull/809#discussion_r1936856070) raised a new use case:

> about the likely need for a caller to know whether a particular device is supported or not, because an app may want to (if say GPU is not supported) use a different, more performant fallback than for WebNN to silently fall back to CPU. For example, if GPU was unavailable (even though you preferred high performance), then it might be faster to execute the model with WebGPU shaders than WebNN CPU, or it might be okay to use CPU, but the app could load a different model that's more CPU-friendly if it knew that was the case.

This sparked a discussion in [Query mechanism for supported devices #815](https://github.com/webmachinelearning/webnn/issues/815) about possible solutions. For instance, [this shape](https://github.com/webmachinelearning/webnn/issues/815#issuecomment-2657101952) emerged as a possible starting point for further exploration, drafting a generic mechanism for capability introspection with examples of possible parameters and outcomes.

```js
const support = await context.querySupport({
  dataTypes: ['float16', 'int8'],
  maximumRank: 6,
  operators: ['lstm', 'hardSwish'],
});
console.log(support); // "optimized" or "fallback"
```

The next phase in developing device selection is, therefore, to explore this proposal and eventually others.

Other use cases were also raised in [this comment](https://github.com/webmachinelearning/webnn/issues/815#issuecomment-2658627753) for real-time video processing:

> 1. If the user selects functionality like background blur, we want to offer the best quality the device can offer. So, the product has a small set of candidate models and technologies (WebNN, WebGPU, WASM) that it has to choose between. Accelerated technologies come with an allowance for beefier models.

> 2. The model/tech chooser algorithm needs to be fast, and we need to avoid spending seconds or even hundreds of milliseconds to figure out if a given model should be able to run accelerated. So, for example, downloading the entirety (could be large), compiling, and try-running a model seems infeasible.

Given the discussion in Issue #815 ([comment](https://github.com/webmachinelearning/webnn/issues/815#issuecomment-2635299222), [comment](https://github.com/webmachinelearning/webnn/issues/815#issuecomment-2638389869)), the developer use case (for frameworks, not for websites) seems to be:
- Before downloading/loading a model, the developer wants to know if, e.g., the GPU can be used for inference with WebNN.
- If not, then they might want to try a path other than WebNN, e.g., WebGPU.
- If yes, then in some cases (e.g., CoreML), the model needs to be dispatched before knowing for sure whether it can be executed on the GPU. For that, a new API is needed, as discussed in [Get devices used for a graph after graph compilation #836](https://github.com/webmachinelearning/webnn/issues/836) and being explored in PR [#854 (define graph.devices)](https://github.com/webmachinelearning/webnn/pull/854).
Based on the answer, the developer may choose an option other than WebNN. Besides that, the feature permits gathering data on typical graph allocations (note: fingerprintable), which might help the specification work on the device selection API.

## Simple accelerator mapping solution

The following [proposal](https://github.com/webmachinelearning/webnn/issues/815#issuecomment-3198261369) gained support for a simple accelerator mapping solution (before using the previously discussed fine grained constraints):
- Expose a context property (or event) to tell whether CPU fallback is active (or likely active).
- Add a context creation option/hint (e.g. `accelerated: true`) for telling app preference for NPU and/or GPU accelerated ["massively parallel"](https://en.wikipedia.org/wiki/Massively_parallel) processing (MPP).
Note that in [certain use cases](https://www.w3.org/2025/09/25-webmachinelearning-minutes.html) applications might prefer CPU inference, therefore specifying `accelerated: false` has legit use cases as well.
- Add a context property named `"accelerated"` with possible values: `false` (for likely no support for neither GPU nor NPU), and `true` (e.g. fully controlled by the underlying platform which makes a best effort for MPP, yet CPU fallback may occur).

The following Web IDL changes are proposed:

```js
partial dictionary MLContextOptions {
  boolean accelerated = true;
};

partial interface MLContext {
  readonly attribute boolean accelerated;
};
```

The behavior of [createContext()](https://webmachinelearning.github.io/webnn/#dom-ml-createcontext) is proposed to follow this policy:
- Set the `accelerated` property to `false` when the platform could in principle provide massive parallel processing which may or may not be available at the moment. Applications may poll this property.

In the future, more policy options could be considered, for instance:
- Return an error [in step 4](https://webmachinelearning.github.io/webnn/#create-a-context) if the context option `accelerated` has been set to `true`, but the platform cannot provide massive parallel processing at all.

## History

Previous discussion covered the following main topics:
- Who controls the execution context: script vs. user agent (OS).
- CPU vs. GPU device selection, including handling multiple GPUs.
- How to handle NPU devices, quantization/dequantization.

In [Simplify MLContext creation #322](https://github.com/webmachinelearning/webnn/pull/322), the proposal was to always use an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) object to initialize a context and remove the `"gpu"` [context option](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions). Also, remove the `'high-performance'` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), since it was used for the GPU option, which now becomes explicit.

Explicit GPU selection also provides clarity when there are multiple GPU devices, as implementations need to use [WebGPU](https://gpuweb.github.io/gpuweb/) to select a [GPUAdapter](https://gpuweb.github.io/gpuweb/#gpuadapter), from which they can request a [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) object.
A counter-argument was that it becomes more complex to use an implementation-selected default GPU, as there is no simple way anymore to tell implementations to use any GPU device for creating an [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext). This concern could eventually be alleviated by keeping the `'high-performance'` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), but on some devices, the NPU might be faster than the GPU.

In [Need to understand how WebNN supports implementation that involves multiple devices and timelines #350](https://github.com/webmachinelearning/webnn/issues/350), it was pointed out that [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext) supports only a single device, while some frameworks support working with a single graph over multiple devices (e.g., CoreML). The proposal was to create a _default_ context that has no explicitly associated device (it could also be named a _generic_ context), where the implementation may choose the underlying device(s).

In [API simplification: context types, context options #302](https://github.com/webmachinelearning/webnn/issues/302), the [proposal](https://github.com/webmachinelearning/webnn/issues/302#issuecomment-1960407195) was that the default behavior should be to delegate device selection to the implementation and remove [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype).
However, the hints/options mechanism should be kept, with an improved mapping to use cases.
For instance, device selection is not about mandating where to execute but, e.g., telling what to avoid if possible (e.g., don't use the GPU). In this case, the [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions), such as [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) and [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), could be used for mapping user hints into device selection logic by implementations. The list of options could be extended based on future needs. Note that the current hints don't guarantee the selection of a particular device type (such as GPU) or a given combination of devices (such as CPU+NPU). For instance, using the `"high-performance"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference) may not guarantee GPU execution, depending on the underlying platform.

In [WebNN should support NPU and QDQ operations #623](https://github.com/webmachinelearning/webnn/issues/623), an explicit request to support NPU device selection was discussed, along with quantization use cases. Several [options](https://github.com/webmachinelearning/webnn/issues/623#issuecomment-2063954107) were proposed, and the simplest one was chosen: extending the [device type enum](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) with the `"npu"` value and updating the relevant algorithms, as added in [PR #696](https://github.com/webmachinelearning/webnn/pull/696).
However, the `deviceType` option, including the `"npu"` value, was later removed from `MLContextOptions` as part of a broader shift to make the context device-agnostic by default (see discussion around Issue #749 and PR #809).
Alternative policies for error handling and fallback scenarios remained open questions.

Later, the need for explicit device selection support was challenged in [MLContextOptions.deviceType seems unnecessary outside of conformance testing #749](https://github.com/webmachinelearning/webnn/issues/749), with the main arguments also summarized in a W3C TPAC group meeting [presentation](https://lists.w3.org/Archives/Public/www-archive/2024Sep/att-0006/MLDeviceType.pdf). The main points were:
- The [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) option is hard to standardize because of the heterogeneity of compute units across various platforms and even across their versions. For instance, `"npu"` might not be a standalone available option, only a combined form of `"npu"` and `"cpu"`.
- As for error management vs. fallback policies: fallback is preferable to failing, and implementations/the underlying platforms should determine the fallback type based on runtime information.
- Implementations, browsers, or the OS have a better grasp of the system, compute, runtime, and application state than websites, and therefore control should be relinquished to them. For instance, if rendering performance degrades, the implementation/underlying platform can possibly fix it the best way, not the web app.
This led to the changes implemented in PR #809.

## Background thoughts

### Representing NPUs

There have been ideas to represent NPUs in a similar way as WebGPU [adapters](https://gpuweb.github.io/gpuweb/#gpuadapter), basically exposing basic string information, features, limits, and whether they can be used as a fallback device.

However, this would likely be premature standardization, as NPUs are very heterogeneous in their implementations; for instance, memory and processing unit architecture can be significantly different. Also, they can be either standalone devices (e.g., TPUs) or integrated as SoC modules, together with CPUs and even GPUs.

There is a fundamental difference between programming NPUs and programming GPUs. From a programming point of view, NPUs are very specific and need specialized drivers, which integrate into AI libraries and frameworks. Therefore, they don't need explicitly exposed abstractions like in WebGPU, but they might have specific quantization requirements and limitations.

Currently the main use case for NPUs is to offload the more general purpose computing devices (CPU and GPU) from machine learning compute loads. Power efficient performance is the main characteristic.

Therefore, use cases that include NPUs could be represented by the `"low-power"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), which could mean the following mappings (controlled by the underlying platform):
- Pure NPU execution.
- NPU preferred, fallback to CPU.
- Combined (multiple) NPU and CPU or GPU execution.

### Selecting from multiple [types] of NPUs

The proposal above uses WebGPU mechanisms to select a GPU device for a context. This covers support for multiple GPUs, even with different types and capabilities.

We don't have such mechanisms to select NPUs. Also, enumerating and managing adapters are not very Web-idiomatic designs. For instance, to avoid this complexity and minimize the fingerprinting surface, the [Presentation API](https://www.w3.org/TR/presentation-api/) outsourced selecting the target device to the user agent so that the web app can achieve the use case without being exposed to platform-specific details.

In the WebNN case, we cannot use such selection mechanisms delegated to the user agent, because the API is used by frameworks, not by web pages.

As such, currently, the handling of multiple NPUs (e.g., a single model on multiple NPUs, or multiple models on multiple NPUs) is delegated to the implementations and underlying platforms.

### Hybrid execution scenarios using NPU, CPU and GPU

Many platforms support various hybrid execution scenarios involving NPUs, CPUs, and GPUs (e.g., NPU-CPU, NPU-GPU, NPU-CPU-GPU), but these are not explicitly exposed and controlled in WebNN. They are best selected and controlled by the implementations. However, we should distill the main use cases behind hybrid execution and define a hinting/mapping mechanism, such as the `powerPreference` option mentioned earlier.

As an example for handling hybrid execution as well as the underlying challenges, take a look at [OpenVINO device selection](https://blog.openvino.ai/blog-posts/automatic-device-selection-and-configuration).

### An Example Hardware Selection Guide

When distributing compute nodes across GPUs, NPUs, and CPUs during AI model inference, optimal strategies depend on operation type, model architecture, and system constraints. Below are key approaches based on performance characteristics and hardware capabilities:

#### **1. Operation-Type Optimization**

- **Matrix multiplication (compute-bound)**:
Use **GPUs** for large matrix operations (e.g., transformer prefill stages). GPUs achieve approximately 22% lower latency and 2x higher throughput than NPUs for these tasks due to parallel compute units.
    - Example: Serving Llama 70B with TensorRT-LLM on NVIDIA Hopper GPUs.
- **Matrix-vector multiplication (memory-bound)**:
Deploy **NPUs**, which reduce latency by 58.5% compared to GPUs. NPUs leverage DMA for efficient memory access, ideal for LLM decode phases.
    - Example: NPUs process TinyLlama inference 3.2x faster than GPUs.
- **Low-complexity operations (e.g., dot product)**:
Assign to **CPUs**, which avoid GPU/NPU memory overhead and achieve lower latency for non-parallel tasks.

#### **2. Model Architecture Considerations**

- **Large Language Models (LLMs)**:
    - **Prefill**: GPU clusters (compute-heavy).
    - **Decode**: NPUs (memory-bound, sequential token generation).
    - Use **disaggregated serving** to split phases across devices, boosting throughput up to 30x.
- **LSTM/RNN Models**:
Prefer **GPUs**, which outperform NPUs by 2.7x due to irregular memory access patterns.
- **Vision Models (e.g., MobileNetV2)**:
    - **Small batches**: NPUs (consistent latency).
    - **Large batches**: GPUs (scaling throughput).

#### **3. Batch Size and Latency Tradeoffs**

| Scenario          | Preferred Hardware | Rationale                                       |
| :---------------- | :----------------- | :---------------------------------------------- |
| Small batch (1-8) | NPU                | 3x lower latency for video classification[^2] |
| Large batch (>32) | GPU                | Throughput scales with parallel compute[^2]     |
| Real-time SLOs    | NPU + CPU          | NPU for decode, CPU for lightweight ops[^3]     |

#### **4. Power-Constrained Deployments**

- **NPUs** consume 50% or less power than GPUs for equivalent performance, making them ideal for edge devices.
- Use **CPU/NPU hybrids** for latency-sensitive applications requiring energy efficiency.

#### **5. Dynamic Orchestration**

Tools may monitor GPU/NPU utilization and automatically:

- Shift decode GPUs to prefill during traffic spikes.
- Select optimal tensor parallelism strategies (e.g., separately for prefill and decode).
- Leverage support for low-latency data movement between heterogeneous devices.

By combining hardware-specific strengths with adaptive resource management, developers can achieve 2x–30x throughput improvements while maintaining strict latency targets.

(Editor's Note: The following simplified guide is based on ongoing discussions. There's a suggestion to use qualitative terms for latency, map devices to latency-sensitive use cases, and exclude training considerations from this specific guide. See PR #860 discussion for details.)
### Simplified guide

| Factor               | CPU                             | GPU                                | NPU                            |
| :------------------- | :------------------------------ | :--------------------------------- | :----------------------------- |
| **Best For**         | Sequential logic, small models  | Performance, large batches         | Edge inference, low-power AI   |
| **Power Efficiency** | Moderate                        | High consumption                   | Ultra-efficient                |
| **Latency**          | High (50–100 ms)                | Medium (10–30 ms)                  | Low (2–10 ms)                  |
| **Typical Use**      | Preprocessing, decision trees   | Large LLMs, computer vision        | Light laptops, smartphones, IoT devices |

---

#### Key Decision Criteria

- **Throughput needs:**
    - GPUs handle >10k queries/sec, while NPUs typically manage 1–5k queries/sec.
- **Model complexity:**
    - NPUs are optimized for transformer layers; GPUs excel at CNN/RNN workloads.
- **Deployment environment:**
    - NPUs dominate mobile and edge devices; GPUs are standard in cloud and data center environments but are also usable in client environments.

Modern systems often combine all three:

- **CPUs** for input handling.
- **GPUs** for model execution.
- **NPUs** for post-processing—balancing performance and efficiency.
