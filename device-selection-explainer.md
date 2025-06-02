# Device Selection Explainer

## Participate

Feedback on this explainer is welcome via the issue tracker:

- https://github.com/webmachinelearning/webnn/labels/device%20selection - existing discussions about device selection mechanisms
- see also [all issues](https://github.com/webmachinelearning/webnn/issues) and feel free to open a new issue as appropriate

## Introduction

This explainer summarizes the discussion and background on [WebNN device selection](https://webmachinelearning.github.io/webnn/#programming-model-device-selection).

The goal is to help making design decisions on how to handle compute device selection for a WebNN [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext).

A context represents the global state of WebNN model graph execution, including the compute devices (e.g. CPU, GPU, NPU) the [WebNN graph](https://webmachinelearning.github.io/webnn/#mlgraph) is executed on.

When creating a context, an application may want to provide hints to the implementation on what device(s) are preferred for execution.

Implementations, browsers, and the underlying OS may want to control the allocation of compute devices for various use cases and system conditions.

The question is who should be able to, and to what extent, control the execution context state and capabilities.

This has been captured by [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions), such as [device type](https://www.w3.org/TR/2025/CRD-webnn-20250131/#enumdef-mldevicetype) and [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference).

## History

Previous discussion covered the following main topics:
- who controls the execution context: script vs. user agent (OS);
- CPU vs GPU device selection, including handling multiple GPUs;
- how to handle NPU devices, quantization/dequantization.

In [[Simplify MLContext creation #322]](https://github.com/webmachinelearning/webnn/pull/322), the proposal was to always use an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) object to initialize a context and remove the `"gpu"` [context option](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions). Also, remove the `'high-performance"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), since it was used for the GPU option, which now becomes explicit.

Explicit GPU selection also provides clarity when there are multiple GPU devices, as implementations need to use [WebGPU](https://gpuweb.github.io/gpuweb/) in order to select a [GPUAdapter](https://gpuweb.github.io/gpuweb/#gpuadapter), from where they can request a [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) object.
A counter-argument was that it becomes more complex to use an implementation selected default GPU, as there is no simple way any more to tell implementations to use any GPU device for creating an [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext). This concern could eventually be alleviated by keeping the `'high-performance"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), but on some devices the NPU might be faster than the GPU.

In [[Need to understand how WebNN supports implementation that involves multiple devices and timelines #350]](https://github.com/webmachinelearning/webnn/issues/350) it was pointed out that [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext) supports only a single device, while there are frameworks that support working with a single graph over multiple devices (e.g. CoreML). The proposal was to create a _default_ context that has no explicitly associated device (it could be also named a _generic_ context), where the implementation may choose the underlying device(s).

In [[API simplification: context types, context options #302]](https://github.com/webmachinelearning/webnn/issues/302), the [proposal](https://github.com/webmachinelearning/webnn/issues/302#issuecomment-1960407195) was that the default behaviour should be to  delegate device selection to the implementation, and remove [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype).
However, keep the hints/options mechanism, with an improved mapping to use cases.
For instance, device selection is not about mandating where to execute, but e.g. tell what to avoid if possible (e.g. don't use the GPU). In this case, the [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions), such as [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) and [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference) could be used for mapping user hints into device selection logic by implementations. The list of options could be extended based on future needs. Note that the current hints don't guarantee the selection of a particular device type (such as GPU) or a given combination of devices (such as CPU+NPU). For instance using the `"high-performance"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference) may not guarantee GPU execution, depending on the underlying platform.

In [[WebNN should support NPU and QDQ operations #623]](https://github.com/webmachinelearning/webnn/issues/623), an explicit request to support NPU device selection was discussed, along with quantization use cases. Several [options](https://github.com/webmachinelearning/webnn/issues/623#issuecomment-2063954107) were proposed, and the simplest one was chosen, i.e. extending the [device type enum](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) with the `"npu"` value and update the relevant algorithms, as added in [PR #696](https://github.com/webmachinelearning/webnn/pull/696).
However, alternative policies for error handling and fallback scenarios remained open questions.

Later the need for explicit device selection support was challenged in [[MLContextOptions.deviceType seems unnecessary outside of conformance testing #749]](https://github.com/webmachinelearning/webnn/issues/749), with the main arguments also summarized in a W3C TPAC group meeting [presentation](https://lists.w3.org/Archives/Public/www-archive/2024Sep/att-0006/MLDeviceType.pdf). The main points were the following:
- The [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) option is hard to standardize because of the heterogeneity of the compute units across various platforms, and even across their versions, for instance `"npu"` might not be a standalone option available, only a combined form of `"npu"` and `"cpu"`.
- As for error management vs. fallback policies: fallback is preferable instead of failing, and implementations/the underlying platforms should determine the fallback type based on runtime information.
- Implementation / browser / OS have better grasp of the system/compute/runtime/apps state than websites, and therefore control should be relished to them. For instance, if rendering performance degrades, the implementation/underlying platform can possibly fix it the best way, not the web app.

## Key use cases and requirements

### Device Preference Use Cases

A WebNN application may have specific device preferences for model execution. These preferences can be mapped to the following use cases:

*   **Prefer execution on the main CPU**:
    *   *Preference*: `"prefer CPU"`
    *   *Description*: The application developer hints that the model should ideally run on the device component primarily responsible for general computation, typically "where JS and Wasm execute". This could be due to the model's characteristics (e.g., heavy control flow, operations best suited for CPU) or to reserve other accelerators for different tasks.
*   **Prefer execution on a Neural Processing Unit (NPU)**:
    *   *Preference*: `"prefer NPU"`
    *   *Description*: The application developer hints that the model is well-suited for an NPU. NPUs are specialized hardware accelerators, distinct from CPUs (typically "where JS and Wasm execute") and GPUs (typically "where WebGL and WebGPU programs execute"). In a future-proof context, NPUs fall under the category of "other" compute devices, encompassing various current and future specialized ML accelerators. This preference is often chosen for models optimized for low power and sustained performance.
*   **Prefer execution on a Graphics Processing Unit (GPU)**:
    *   *Preference*: `"prefer GPU"`
    *   *Description*: The application developer hints that the model should run on the GPU (the device "where WebGL and WebGPU programs execute"). This is common for models with highly parallelizable operations.
*   **Maximize Performance**:
    *   *Preference*: `"maximum performance"`
    *   *Description*: The application developer desires the highest possible throughput or lowest latency for the model execution, regardless of power consumption. The underlying system will choose the device or combination of devices (e.g., "where WebGL and WebGPU programs execute", or "other" specialized hardware) that can achieve this.
*   **Maximize Power Efficiency**:
    *   *Preference*: `"maximum efficiency"`
    *   *Description*: The application developer prioritizes executing the model in the most power-efficient manner, which might involve using an NPU ("other") or a low-power mode of the CPU ("where JS and Wasm execute"). This is crucial for battery-constrained devices or long-running tasks.
*   **Minimize Overall System Power**:
    *   *Preference*: `"minimum overall power"`
    *   *Description*: The application developer hints that the model execution should contribute as little as possible to the overall system power draw. This is a broader consideration than just the model's own efficiency, potentially influencing scheduling and resource allocation across the system. The implementation may choose any device ("where JS and Wasm execute", "where WebGL and WebGPU programs execute", or "other") that best achieves this goal.

Design decisions may take the following into account:

1. Allow the underlying platform to ultimately choose the compute device.

2. Allow scripts to express hints/options when creating contexts, such as preference for low power consumption, or high performance (throughput), low latency, stable sustained performance, accuracy, etc.

3. Allow an easy way to create a context with a GPU device, i.e. without specifying an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice).

4. Allow selection from available GPU devices, for instance by allowing specifying an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) obtained from available [GPUAdapters](https://gpuweb.github.io/gpuweb/#gpuadapter) using the [WebGPU](https://gpuweb.github.io/gpuweb) mechanisms via [GPURequestAdapterOptions](https://gpuweb.github.io/gpuweb/#dictdef-gpurequestadapteroptions), such as feature level or power preference.

5. Allow selection from available various AI accelerators, including NPUs or a combination of accelerators. This may happen using a (to be specified) algorithmic mapping from context options. Or, allow web apps to hint a preferred fallback order for the given context, for instance `["npu", "cpu"]`, meaning that implementations should try executing the graph on NPU as much as possible and try to avoid GPU. Basically `"cpu"` could even be omitted, as it could be the default fallback device, therefore specifying `"npu"` alone would mean the same. However, this can become complex with all possible device variations, so we must specify and standardize the supported fallback orders.

6. Allow enumeration of [OpSupportLimits](https://webmachinelearning.github.io/webnn/#api-mlcontext-opsupportlimits-dictionary) before creating a context, so that web apps could select the best device which would work with the intended model. This needs more developer input and examples.

7. As a corollary to 6, allow creating a context using also options for [OpSupportLimits](https://webmachinelearning.github.io/webnn/#api-mlcontext-opsupportlimits-dictionary).


## Scenarios, examples, design discussion

Examples for user scenarios:

```js
// simple context creation with implementation defaults
context = await navigator.ml.createContext();

// create a context that will likely map to NPU, or NPU+CPU
context =  await navigator.ml.createContext({powerPreference: 'low-power'});

// create a context that will likely map to GPU
context = await navigator.ml.createContext({powerPreference: 'high-performance'});

// enumerate devices and limits (as allowed by policy/implementation)
// and select one of them to create a context
const limitsMap = await navigator.ml.opSupportLimitsPerDevice();
// analyze the map and select an op support limit set
// ...
const context = await navigator.ml.createContext({
    limits: limitsMap['npu1']
});

// as an alternative, hint a preferred fallback order ["npu", "cpu"]
// i.e. try executing the graph on NPU and avoid GPU as much as possible
// but do as it's best fit with the rest of the context options
const context = await navigator.ml.createContext({ fallback: ['npu', 'cpu'] });

```

## Open questions

- [WebGPU](https://gpuweb.github.io/gpuweb/) provides a way to select a GPU device via [GPUAdapter](https://gpuweb.github.io/gpuweb/#gpuadapter). Should we expose a similar adapter API for NPUs?

- How should we extend the context options?
What exactly is best to pass as context options? Op support limits? Supported features, similar to [GPUSupportedFeatures](https://gpuweb.github.io/gpuweb/#gpusupportedfeatures)? Others?

- Concerning security and privacy, would the proposals here increase the fingerprinting surface? If yes, what mitigations can be made? The current understanding is that any extra information exposed to web apps in these proposals could be obtained by other methods as well. However, security hardening and relevant mitigations are recommended. For instance, implementations could choose the level of information (e.g. op support limits) exposed to a given origin.


## Background thoughts

### Representing NPUs

There have been ideas to represent NPUs in a similar way as WebGPU [adapters](https://gpuweb.github.io/gpuweb/#gpuadapter), basically exposing basic string information, features, limits, and whether they can be used as a fallback device.

However, this would likely be premature standardization, as NPUs are very heterogeneous in their implementations, for instance memory and processing unit architecture can be significantly different. Also, they can be either standalone devices (e.g. TPUs), or integrated as SoC modules, together with CPUs, and even GPUs.

There is a fundamental difference between programming NPUs vs. programming GPUs. From programming point of view, NPUs are very specific and need specialized drivers, which integrate into AI libraries and frameworks. Therefore they don't need explicitly exposed abstractions like in [WebGPU](https://gpuweb.github.io/gpuweb/), but they might have specific quantization requirements and limitations.

Currently the main use case for NPUs is to offload the more general purpose computing devices (CPU and GPU) from machine learning compute loads. Power efficient performance is the main characteristic.

Therefore, use cases that include NPUs could be euphemistically represented by the `"low-power"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), which could mean the following mappings (controlled by the underlying platform):
- pure NPU execution,
- NPU preferred, fallback to CPU,
- combined [multiple] NPU and CPU or GPU execution.

### Selecting from multiple [types] of NPUs

The proposal above uses [Web GPU](https://gpuweb.github.io/gpuweb) mechanisms to select a GPU device for a context. This covers support for multiple GPUs, even with different type and capabilities.

We don't have such mechanisms to select NPUs. Also, enumerating and managing adapters are not very web'ish designs. For instance, in order to avoid this complexity and also to minimize fingerprinting surface, the [Presentation API](https://www.w3.org/TR/presentation-api/) outsourced selecting the target device to the user agent, so that the web app can achieve the use case without being exposed with platform specific details.

In the WebNN case, we cannot use such selection mechanisms delegated to the user agent, because the API is used by frameworks, not by web pages.

As such, currently the handling of multiple NPUs (e.g. single model on multiple NPUs, or multiple models on multiple NPUs) is delegated to the implementations and underlying platforms.

### Hybrid execution scenarios using NPU, CPU and GPU

Many platforms support various hybrid execution scenarios involving NPU, CPU, and GPU (e.g. NPU-CPU, NPU-GPU, NPU-CPU-GPU), but these are not explicitly exposed and controlled in WebNN. They are best selected and controlled by the implementations. However, we should distill the main use cases behind hybrid execution and define a hinting/mapping mechanism, such as the power preference mentioned earlier.

As an example for handling hybrid execution as well as the underlying challenges, take a look at [OpenVINO device selection](https://blog.openvino.ai/blog-posts/automatic-device-selection-and-configuration).

## Considered alternatives

1. Keep the current [MLDeviceType](https://www.w3.org/TR/2025/CRD-webnn-20250131/#enumdef-mldevicetype) as a context option, but improve the device type names and specify an algorithm for a mapping of these names to various real adapters (with their given characteristics). However, this would be more limited than being able to specify device specific limits to context creation. (This is the current approach).

2. Remove [MLDeviceType](https://www.w3.org/TR/2025/CRD-webnn-20250131/#enumdef-mldevicetype), but define a set of [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions) that map well to GPU adapter/device selection and also to NPU device selection. (This is the proposed first approach.)

3. Follow this [proposal](https://github.com/webmachinelearning/webnn/issues/749#issuecomment-2429821928), also tracked in [[MLOpSupportLimits should be opt-in #759]](https://github.com/webmachinelearning/webnn/issues/759). That is, allow listing op support limits outside of a context, which would return all available devices with their op support limits. Then the web app could choose one of them to initialize a context with. (This is a suggested longer term discussion topic.)


## Minimum Viable Solution

Based on the discussion above, the best starting point was a simple solution that can be extended and refined later. A first contribution could include the following changes:
- Remove [MLDeviceType](https://www.w3.org/TR/2025/CRD-webnn-20250131/#enumdef-mldevicetype) as explicit [context option](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions).
- Update [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext) so that it becomes device agnostic, or _default_/_generic_ context. Allow supporting multiple devices with one context.
- Add algorithmic steps or notes to implementations on how to map [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference) to devices.
- Also, to align with [GPUPowerPreference](https://gpuweb.github.io/gpuweb/#enumdef-gpupowerpreference), we should remove the `"default"` [MLPowerPreference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), i.e. the lack of hints will result in creating a generic context.

This was implemented in [Remove MLDeviceType #809](https://github.com/webmachinelearning/webnn/pull/809).

Besides, the following topics have been discussed:
- Improve the device selection hints in [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions) and define their implementation mappings. For instance, discuss whether to also include a `"low-latency"` performance option.

- Document the valid use cases for requesting a certain device type or combination of devices, and within what error conditions. Currently, after these changes there remains explicit support for GPU-only context when an [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext) is created from a [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) in [createContext()](https://webmachinelearning.github.io/webnn/#api-ml-createcontext).
- Discuss option #3 from [Considered alternatives](#considered-alternatives).

## Next Phase Device Selection Solution

In [Remove MLDeviceType #809](https://github.com/webmachinelearning/webnn/pull/809)  this [comment](https://github.com/webmachinelearning/webnn/pull/809#discussion_r1936856070) raised a new use case:

> about the likely need for a caller to know whether a particular device is supported or not, because an app may want to (if say GPU is not supported) use a different more performant fallback than for WebNN to silently fall back to CPU. For example, if GPU was unavailable (even though you preferred high performance), then it might be faster to execute the model with WebGPU shaders than WebNN CPU, or it might be okay to use CPU, but the app could load a different model that's more CPU-friendly, if it knew that was the case.

That sparked a discussion in [Query mechanism for supported devices #815
](https://github.com/webmachinelearning/webnn/issues/815) about possible solutions, for instance [this shape](https://github.com/webmachinelearning/webnn/issues/815#issuecomment-2657101952) emerged as a possible starting point for further exploration, drafting a generic mechanism for capability introspection and examples of possible parameters and outcomes.

```js
const support = await context.querySupport({
  dataTypes: ['float16', 'int8'],
  maximumRank: 6,
  operators: ['lstm', 'hardSwish'],
});
console.log(support); // "optimized" or "fallback"
```

The next phase in developing device selection is therefore to explore this proposal and eventually others.

Other use cases were raised as well, in [this comment](https://github.com/webmachinelearning/webnn/issues/815#issuecomment-2658627753) for realtime video processing:

> 1. If the user selects to use functionality like background blur, we want to offer the best quality the device can offer. So the product has a small set of candidate models and technologies (WebNN, WebGPU, WASM) that it has to choose between. Accelerated technologies come with allowance for beefier models.

> 2. The model/tech choser algorithm needs to be fast, and we need to avoid spending seconds or even hundreds of milliseconds to figure out if a given model should be able to run accelerated. So for example downloading the entirety (could be large things..), compiling & try-running a model seems infeasible.
