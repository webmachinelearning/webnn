# Device Selection Explainer

## Introduction

This explainer summarizes the discussion and background on [Web NN device selection](https://webmachinelearning.github.io/webnn/#programming-model-device-selection).

The goal is to help making design decisions on whether and how to handle compute device selection for a Web NN [MLContext](https://webmachinelearning.github.io/webnn/#mlcontext).

A context represents the global state of Web NN model graph execution, including the compute devices (e.g. CPU, GPU, NPU) the [Web NN graph](https://webmachinelearning.github.io/webnn/#mlgraph) is executed on.

When creating a context, an application may want to provide hints to the implementation on what device(s) are preferred for execution.

Implementations, browsers and underlying OS may want to control the allocation of compute devices for various use cases and system conditions.

The question is in what use cases who and how much should control the execution context.

Currently this is captured by [context options](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions), such as [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) and [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference).

## History

Previous discussion covered the following main topics (see References):
- who control of context: script vs. user agent (OS);
- CPU vs GPU device selection, including handling multiple GPUs;
- how to handle NPU execution, quantization/dequantization.

In [[Simplify MLContext creation #322]](https://github.com/webmachinelearning/webnn/pull/322), the proposal was to always use an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) object to initialize a context and remove the `"gpu"` [context option](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions).
Also, remove the `'high-performance"` [power preference](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference), since it was used for the GPU option which became explicit.
Explicit GPU selection also provides clarity when there are multiple GPU devices, as implementations don't need to rely on hints such as power preference to deduce which GPU device to select.
A counter-argument was that it becomes more complex to use an implementation selected default GPU, and there was value in that simplicity.

In [[API simplification: context types, context options #302]](https://github.com/webmachinelearning/webnn/issues/302), the [proposal](https://github.com/webmachinelearning/webnn/issues/302#issuecomment-1960407195) was to make delegating device selection to the implementation should be the default behaviour and remove [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype).
However, keep the hints/options mechanism, with an improved mapping to use cases.
For instance, device selection is not about mandating where to execute, but e.g. tell what to avoid if possible (e.g. don't use the GPU).

In [[WebNN should support NPU and QDQ operations #623]](https://github.com/webmachinelearning/webnn/issues/623), an explicit request to support NPU device selection was discussed, along with quantization use cases. Several [options](https://github.com/webmachinelearning/webnn/issues/623#issuecomment-2063954107) were proposed, and the simplest one was chosen, i.e. extending the [device type enum](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) with the `"npu"` value and update the relevant algorithms, as added in [PR #696](https://github.com/webmachinelearning/webnn/pull/696).
However, alternative policies for error handling and fallback scenarios remained open questions.

Later the need for explicit device selection support was challenged in [[MLContextOptions.deviceType seems unnecessary outside of conformance testing #749]](https://github.com/webmachinelearning/webnn/issues/749), with the main arguments also summarized in a W3C TPAC group meeting [presentation](https://lists.w3.org/Archives/Public/www-archive/2024Sep/att-0006/MLDeviceType.pdf).

The main points were the following:
- The [device type](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) option is hard to standardize because of the heterogeneity of the compute units across various platforms, and even across their versions, for instance `"npu"` might not be a standalone option available, only a combined form of `"npu"` and `"cpu"`.
- As for error management vs. fallback policies: fallback is preferable instead of failing, and implementations/the underlying platforms should determine the fallback type based on runtime information.
- Implementation / browser / OS have better grasp of the system/compute/runtime/apps state then websites, therefore control should be relished to them. For instance, if rendering performance degrades, the implementation/underlying platform can possibly fix it the best way, not the web app.

## Key use cases and requirements

Design decisions should take the following into account:

1. Allow the underlying platform ultimately choose the compute device.

2. Allow scripts to express hints/options when creating contexts, such as preference for low power consumption, or high performance, low latency, or stable sustained performance etc.

3. Allow an easy way to create a context with a GPU device, i.e. without specifying an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice).

4. Allow selection from available GPU devices, for instance by allowing specifying an explicit [GPUDevice](https://gpuweb.github.io/gpuweb/#gpudevice) obtained from available devices using the [WebGPU](https://gpuweb.github.io/gpuweb) mechanisms.

5. Allow selection from available various AI accelerators, including NPUs or a combination of accelerators. This may happen using a (to be specified) algorithmic mapping from context options. Or, allow web apps to hint a preferred fallback order for the given context, for instance `["npu", "cpu"]`, meaning that implementations should try executing the graph on NPU as much as possible and try to avoid GPU. Basically `"cpu"` could even be omitted, as it could be the default fallback device, therefore specifying `"npu"` alone would mean the same. However, this can become complex with all possible device variations, so we must specify and standardize the supported fallback orders.

6. Allow enumeration of [OpSupportLimits](https://webmachinelearning.github.io/webnn/#api-mlcontext-opsupportlimits-dictionary) before creating a context, so that web apps could select the best device which would work with the intended model.

7. As a corollary to 6, allow creating a context using also options for [OpSupportLimits](https://webmachinelearning.github.io/webnn/#api-mlcontext-opsupportlimits-dictionary).

## Considered alternatives

1. Keep the current [MLDeviceType](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype) as a context option, but improve the device type names and specify an algorithm for a mapping these names to various real adaptors (with their given characteristics). However, this would be more limited than being able to specify device specific limits to context creation.

2. Follow this [proposal](https://github.com/webmachinelearning/webnn/issues/749#issuecomment-2429821928), also tracked in [[MLOpSupportLimits should be opt-in #759]](https://github.com/webmachinelearning/webnn/issues/759).

## Scenarios, examples, design discussion

Examples for user scenarios:

```js
// simple context creation with implementation defaults
context = await navigator.ml.createContext();

// create a context that will likely map to NPU
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

[WebGPU](https://gpuweb.github.io/gpuweb/) provides a way to select a GPU device, called [GPUAdapter](https://gpuweb.github.io/gpuweb/#gpuadapter). Should we align the naming between adapter and device?

Should we expose a similar adapter API for NPUs? Or could NPUs be represented as [GPUAdapter](https://gpuweb.github.io/gpuweb/#gpuadapter) (basically a few text attributes)?

How should we extend the context options?
What exactly is best to pass as context options? Op support limits? Supported features, similar to [GPUSupportedFeatures](https://gpuweb.github.io/gpuweb/#gpusupportedfeatures)? Others?

Update the security and privacy section. Would the proposals here increase the fingerprinting vector? If yes, what mitigations can be made? The current understanding is that any extra information exposed to web apps in these proposals could be obtained by other methods as well. However, security hardening and relevant mitigations are recommended. For instance, implementations could choose the level of information (e.g. op support limits) exposed to a given origin.
