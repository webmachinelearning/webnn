# Device Selection Explainer

## Introduction

This explainer summarizes the discussion and background on [Web NN device selection](https://webmachinelearning.github.io/webnn/#programming-model-device-selection).

The goal is to help making design decisions on how to handle compute device selection for a Web NN [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext).

A context represents the global state of Web NN model graph execution, including the compute devices (e.g. CPU, GPU, NPU) the [Web NN graph](https://webmachinelearning.github.io/webnn/#mlgraph) is executed on.

When creating a context, an application may want to provide hints to the implementation on what device(s) are preferred for execution.

Implementations, browsers and underlying OS may want to control the allocation of compute devices for various use cases and system conditions.

The question is in what use cases who and how much should control the execution context.

Currently this is captured by [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions), such as [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) and [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference).

## History

Previous discussion covered the following main topics:MLContext
- who controls the execution context: script vs. user agent (OS);
- CPU vs GPU device selection, including handling multiple GPUs;
- how to handle NPU devices, quantization/dequantization.

In [[Simplify MLContext creation #322]](https://github.com/webmachinelearning/webnn/pull/322), the proposal was to always use an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) object to initialize a context and remove the `"gpu"` [context option](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions).
Also, remove the `'high-performance"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), since it was used for the GPU option, which now becomes explicit.
Explicit GPU selection also provides clarity when there are multiple GPU devices, as implementations need to use [WebGPU](https://gpuweb.github.io/gpuweb/) in order to select a [GPUAdapter](https://gpuweb.github.io/gpuweb/#gpuadapter), from where they can request a [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) object.
A counter-argument was that it becomes more complex to use an implementation selected default GPU, as there is no simple way any more to tell implementations to use any GPU device for creating an [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext). This concern could eventually be alleviated by keeping the `'high-performance"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference).

In [[Need to understand how WebNN supports implementation that involves multiple devices and timelines #350]](https://github.com/webmachinelearning/webnn/issues/350) it was pointed out that [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext) supports only a single device, while there are frameworks that support working with a single graph over multiple devices (e.g. CoreML). The proposal was to create a _default_ context that has no explicitly associated device (it could be also named a _generic_ context), where the implementation may choose the underlying device(s).

In [[API simplification: context types, context options #302]](https://github.com/webmachinelearning/webnn/issues/302), the [proposal](https://github.com/webmachinelearning/webnn/issues/302#issuecomment-1960407195) was that the default behaviour should be to  delegate device selection to the implementation, and remove [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype).
However, keep the hints/options mechanism, with an improved mapping to use cases.
For instance, device selection is not about mandating where to execute, but e.g. tell what to avoid if possible (e.g. don't use the GPU). In this case, the [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions), such as [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) and [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference) could be used for mapping user hints into device selection logic by implementations. The list of options could be extended based on future needs. Note that the current hints don't guarantee the selection of a particular device type (such as GPU) or a given combination of devices (such as CPU+NPU). For instance using the `"high-performance"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference) may not guarantee GPU execution, depending on the underlying platform.

In [[WebNN should support NPU and QDQ operations #623]](https://github.com/webmachinelearning/webnn/issues/623), an explicit request to support NPU device selection was discussed, along with quantization use cases. Several [options](https://github.com/webmachinelearning/webnn/issues/623#issuecomment-2063954107) were proposed, and the simplest one was chosen, i.e. extending the [device type enum](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) with the `"npu"` value and update the relevant algorithms, as added in [PR #696](https://github.com/webmachinelearning/webnn/pull/696).
However, alternative policies for error handling and fallback scenarios remained open questions.

Later the need for explicit device selection support was challenged in [[MLContextOptions.deviceType seems unnecessary outside of conformance testing #749]](https://github.com/webmachinelearning/webnn/issues/749), with the main arguments also summarized in a W3C TPAC group meeting [presentation](https://lists.w3.org/Archives/Public/www-archive/2024Sep/att-0006/MLDeviceType.pdf). The main points were the following:
- The [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) option is hard to standardize because of the heterogeneity of the compute units across various platforms, and even across their versions, for instance `"npu"` might not be a standalone option available, only a combined form of `"npu"` and `"cpu"`.
- As for error management vs. fallback policies: fallback is preferable instead of failing, and implementations/the underlying platforms should determine the fallback type based on runtime information.
- Implementation / browser / OS have better grasp of the system/compute/runtime/apps state then websites, therefore control should be relished to them. For instance, if rendering performance degrades, the implementation/underlying platform can possibly fix it the best way, not the web app.

## Key use cases and requirements

Design decisions should take the following into account:

1. Allow the underlying platform ultimately choose the compute device.

2. Allow scripts to express hints/options when creating contexts, such as preference for low power consumption, or high performance (throughput), low latency, stable sustained performance etc.

3. Allow an easy way to create a context with a GPU device, i.e. without specifying an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice).

4. Allow selection from available GPU devices, for instance by allowing specifying an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) obtained from available [GPUAdapters](https://gpuweb.github.io/gpuweb/#gpuadapter) using the [WebGPU](https://gpuweb.github.io/gpuweb) mechanisms via [GPURequestAdapterOptions](https://gpuweb.github.io/gpuweb/#dictdef-gpurequestadapteroptions), such as feature level or power preference.

5. Allow selection from available various AI accelerators, including NPUs or a combination of accelerators. This may happen using a (to be specified) algorithmic mapping from context options. Or, allow web apps to hint a preferred fallback order for the given context, for instance `["npu", "cpu"]`, meaning that implementations should try executing the graph on NPU as much as possible and try to avoid GPU. Basically `"cpu"` could even be omitted, as it could be the default fallback device, therefore specifying `"npu"` alone would mean the same. However, this can become complex with all possible device variations, so we must specify and standardize the supported fallback orders.

6. Allow enumeration of [OpSupportLimits](https://webmachinelearning.github.io/webnn/#api-mlcontext-opsupportlimits-dictionary) before creating a context, so that web apps could select the best device which would work with the intended model. This needs more developer input and examples.

7. As a corollary to 6, allow creating a context using also options for [OpSupportLimits](https://webmachinelearning.github.io/webnn/#api-mlcontext-opsupportlimits-dictionary).

## Considered alternatives

1. Keep the current [MLDeviceType](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) as a context option, but improve the device type names and specify an algorithm for a mapping these names to various real adaptors (with their given characteristics). However, this would be more limited than being able to specify device specific limits to context creation.

2. Remove [MLDeviceType](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype), but define a set of [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions) that map well to GPU adapter/device selection and also to NPU device selection.

3. Follow this [proposal](https://github.com/webmachinelearning/webnn/issues/749#issuecomment-2429821928), also tracked in [[MLOpSupportLimits should be opt-in #759]](https://github.com/webmachinelearning/webnn/issues/759). That is, allow listing op support limits outside of a context, which would return all available devices with their op support limits. Then the web app could choose one of them to initialize a context with.

## Scenarios, examples, design discussion

Examples for user scenarios:

```js
// simple context creation with implementation defaults
context = await navigator.ml.createContext();

// create a context that will likely map to NPU, or NPU+CPU
context =
  await navigator.ml.createContext({powerPreference: 'low-power'});

// create a context that will likely map to GPU
context =
  await navigator.ml.createContext({powerPreference: 'high-performance'});

// enumerate devices and limits (as allowed by policy/implementation)
// and select one of them to create a context
const limitsMap = await navigator.ml.opSupportLimitsPerDevice();
// analyze the map and select an op support limit set
// ...
const context = await navigator.ml.createContext({
    limits: deviceLimitsMap['npu1']
});

// as an alternative, hint a preferred fallback order ["npu", "cpu"]
// i.e. try executing the graph on NPU and avoid GPU as much as possible
// but do as it's best fit with the rest of the context options
const context = await navigator.ml.createContext({ fallback: ['npu', 'cpu'] });

```

## Open questions

[WebGPU](https://gpuweb.github.io/gpuweb/) provides a way to select a GPU device, called [GPUAdapter](https://gpuweb.github.io/gpuweb/#gpuadapter). Should we align the naming between GPU adapter and WebNN device?

Should we expose a similar adapter API for NPUs? Or could NPUs be represented as [GPUAdapter](https://gpuweb.github.io/gpuweb/#gpuadapter) (basically a few text attributes)?

How should we extend the context options?
What exactly is best to pass as context options? Op support limits? Supported features, similar to [GPUSupportedFeatures](https://gpuweb.github.io/gpuweb/#gpusupportedfeatures)? Others?

Update the security and privacy section. Would the proposals here increase the fingerprinting surface? If yes, what mitigations can be made? The current understanding is that any extra information exposed to web apps in these proposals could be obtained by other methods as well. However, security hardening and relevant mitigations are recommended. For instance, implementations could choose the level of information (e.g. op support limits) exposed to a given origin.

## Background thoughts

### Representing NPUs

Earlier there have been ideas to represent NPUs in a similar way as WebGPU [adapters](https://gpuweb.github.io/gpuweb/#gpuadapter), basically exposing basic string information, features, limits, and whether they can be used as a fallback device.

However, this would likely be premature standardization, as NPUs are very heterogeneous in their implementations, for instance memory and processing unit architecture can be significantly different. Also, they can be either standalone devices (e.g. TPUs), or integrated as SoC modules, together with CPUs, and even GPUs.

There is a fundamental difference between programming NPUs vs. programming GPUs. From programming point of view, NPUs are very specific and need specialized drivers, which integrate into libraries and frameworks. Therefore they don't need explicitly exposed abstractions like in [WebGPU](https://gpuweb.github.io/gpuweb/), but they might have specific quantization requirements and limitations.

The main use case for NPUs currently is mainly to offload more general purpose computing devices (CPU and even GPU) from machine learning compute loads. Power efficient performance is the main characteristic.

Therefore, use cases that include NPUs could be euphemistically represented by the `"low-power"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), which could mean the following mappings (controlled by the underlying platform):
- pure NPU execution,
- NPU preferred, fallback to CPU,
- combined [multiple] NPU and CPU execution.

### Selecting from multiple [types] of NPUs

The proposal above uses [Web GPU](https://gpuweb.github.io/gpuweb) mechanisms to select a GPU device for a context. This covers support for multiple GPUs, even with different type and capabilities.

We don't have such mechanisms to select NPUs. Earlier there have been ideas to use a similar, if not the same approach as Web GPU.

However, enumerating and managing adapters are not very web'ish designs. For instance, to avoid complexity and to minimize fingerprinting surfaces, the [Presentation API](https://www.w3.org/TR/presentation-api/) outsourced selecting the target device to the user agent, so that the web app can achieve the use case without being exposed with platform specific details.

In the Web NN case, we cannot use such mechanisms, because the API is used by frameworks, not by web pages.

As such, currently the handling of multiple NPUs (e.g. single model on multiple NPUs, or multiple models on multiple NPUs) is delegated to the implementations and underlying platforms.

### Hybrid execution scenarios using NPU, CPU and GPU

Many platforms support various hybrid execution scenarios involving NPU, CPU, and GPU (e.g. NPU-CPU, NPU-GPU, NPU-CPU-GPU), but these are not explicitly exposed and controlled in Web NN. They are best selected and controlled by the implementations. However, we should distillate the main use cases behind hybrid execution and define a hinting/mapping mechanism, such as the power preference mentioned earlier.

As an example for handling hybrid execution as well as the underlying challenges, take a look at [OpenVINO device selection](https://blog.openvino.ai/blog-posts/automatic-device-selection-and-configuration).

## Minimum Viable Solution

Based on the discussion above, the best starting point would be a simple solution that can be extended and refined later. Namely,
- Remove [MLDeviceType](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) as explicit [context option](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions).
- Update [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext) so that it becomes device agnostic, or _default_/_generic_ context. Allow supporting multiple devices with one context.
- Add notes to implementations on how to map [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference) to devices.
- Improve the device selection hints in [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions) and define their implementation mappings. For instance, should we also include `"low-latency"` as a performance option, or should we rename `"default"` to `"auto"` (alluding to an underlying process, rather than a default setting).
- Document the valid use cases for requesting a certain device type or combination of devices, and within what error conditions. Currently, after these changes there remains explicit support for GPU-only context when an [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext) is created from a [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) in [createContext()](https://webmachinelearning.github.io/webnn/#api-ml-createcontext).