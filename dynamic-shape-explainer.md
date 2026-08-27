# Dynamic Shape Explainer

## Authors
- [Bin Miao](mailto:bin.miao@intel.com), [Wanming Lin](mailto:wanming.lin@intel.com) (Intel)

## Participate
- [Support flexible input sizes #883](https://github.com/webmachinelearning/webnn/issues/883)

## Table of contents
1. [Introduction](#introduction)
2. [Goals](#goals)
3. [Non-goals](#non-goals)
4. [Use Cases](#use-cases)
5. [Proposed API](#proposed-api)
6. [Design Discussion](#design-discussion)
7. [Considered Alternatives](#considered-alternatives)
8. [Privacy & Security Considerations](#privacy--security-considerations)
9. [Future Consideration](#future-consideration)
10. [Open Questions](#open-questions)
11. [References & Acknowledgements](#references--acknowledgements)

## Introduction
Currently, all `MLOperand` instances within a WebNN graph are constrained to static shapes. Dimensions must be fully resolved during the [MLGraphBuilder.build()](https://www.w3.org/TR/webnn/#api-mlgraphbuilder-build) compilation phase. This constraint significantly limits the API's adaptability for modern machine learning workloads, where input dimensions often remain indeterminate until the point of inference:

- **Transformer / LLM Decoding:** These workloads involve iterative execution of identical compute graphs with varying sequence lengths. Static shaping necessitates graph recompilation for each distinct sequence length, incurring prohibitive latency.

- **Vision Encoders:** These architectures often process inputs of arbitrary resolutions, causing spatial dimensions to fluctuate dynamically throughout the network.

- **Generative Image Models:** These models frequently compute intermediate tensor shapes at runtime (e.g., dynamic padding to block-size alignments), where padding requirements are strictly dependent on input dimension variance.

Existing framework workarounds — such as frequent graph recompilation or falling back to slower, CPU-based execution environments (e.g., ONNX Runtime Web via WebAssembly) — introduce significant performance bottlenecks, particularly in latency-sensitive autoregressive decoding tasks.

This proposal introduces **Dynamic Shape** for WebNN, enabling graph dimensions to remain unresolved during compilation and instead derive concrete values from input tensors at dispatch time. This mechanism permits a single compiled graph to accommodate a diverse range of runtime input sizes. This document details the design explored in the Chromium implementation and serves as a technical reference for the Working Group's resolution of [Issue #883, "Support flexible input sizes"](https://github.com/webmachinelearning/webnn/issues/883).

## Goals
- Allow a single compiled `MLGraph` to execute across varying runtime input sizes, without rebuilding.

- A dimension is either a static size or a named dynamic dimension (a symbolic name). Requiring every dynamic dimension to carry a name costs no expressive power and gives every dimension a debuggable symbol.

- Defer shape validation that cannot be decided at build time to the point where concrete input shapes are known, without weakening the checks that can be decided at build time.

- Let a framework learn a graph's concrete output shapes for a given set of input shapes **before** dispatching it, so it can allocate output tensors of the right size.

- Support models whose intermediate shapes are computed at runtime from other shapes (e.g. `padded_len = seq_len + (-seq_len % block)`), not just models that thread an input dimension straight through.

## Non-goals
- **We do not resolve shapes from input tensor data.** Dynamism that depends on the *values* inside a tensor is out of scope. We resolve only *Symbolic sizes*: dimensions derivable by algebra from the input shapes and build-time constants.

- **We do not redefine operator semantics.** The operators added here mirror the semantics of their existing static counterparts; they only move shape parameters from build-time attributes to runtime operands.

## Use Cases

### Variable sequence length
A web app runs an SLM decoder. The sequence length grows by one every step, so with static shapes the graph must be rebuilt (or a family of graphs pre-built) for each length.

```js
// Before: the sequence length is baked into the graph.
const input = builder.input('attention_mask', {dataType: 'float32', shape: [1, 128]});
// ...a different compiled graph is required for each sequence length.
```

With dynamic shapes, the sequence length is declared as a named dynamic dimension and the graph is compiled once:

```js
// After: 'sequence_length' is a named dynamic dimension.
const input = builder.input('attention_mask', {dataType: 'float32', shape: [1, 'sequence_length']});
const graph = await builder.build({output});

// The same graph runs at any sequence length.
mlContext.dispatch(graph, {'attention_mask': tensorWithLen(1)}, {'output': out1});
mlContext.dispatch(graph, {'attention_mask': tensorWithLen(37)}, {'output': out2});
```

Every operand whose shape depends on `sequence_length` carries the dynamism through the graph automatically.

### Shapes computed at runtime
Some models contain genuine shape arithmetic that must run at inference time. For example, a "pad the sequence up to a multiple of the block size" step computes the padding amount from the (dynamic) sequence length:

```js
padding_needed = (-seq_len) mod block        // e.g. block = 32
padded         = pad(x, /*end=*/[padding_needed, 0])  // now a multiple of 32
```

Expressing this requires reading a shape *as data*, doing arithmetic on it, and feeding the result back into an operator as a shape parameter:

```js
const shape = builder.shape(x);  // uint32 1-D tensor [seq_len, 2560]
const seqLen = builder.slice(shape, [0], [1]);  // [seq_len]

// (-seq_len) mod 32
const pad = builder.modulusFloor(builder.neg(seqLen), builder.constant(..., [32]));
const zero = builder.constant(..., [0]);  // no padding on the trailing axis
const padded = builder.padDynamic(x, builder.constant(..., [0, 0]), builder.concat([pad, zero], 0));
```

This motivates the **shape-as-data operators** described below.

### Getting output shapes before dispatch
A framework such as ONNX Runtime Web partitions a model and hands WebNN a subgraph. That subgraph's output can carry a dynamic dimension that is not present on any of its inputs - it is derived inside the subgraph, and so carries a name the user agent synthesized rather than one the framework supplied. To allocate the output tensor, the framework needs the concrete output shape before it dispatches:

```js
const outShapes = await graph.computeShapes({'attention_mask': [1, 37]});
// => {'output': [1, 74]}

// Allocate output MLTensors of the resolved size, then dispatch.
```

## Proposed API
The web-facing surface changes fall into three groups: 1) the dimension model and descriptors, 2) new `computeShapes()` methods and 3) new operators. The runtime behavior behind these — the two mechanisms that carry dynamism at dispatch, **shape inference** and **shape computation** — is described under [Design Discussion](#design-discussion).

### 1. Dimension model and descriptors
A graph input's dimension is either a number (a static integer) or a string (a **named dynamic** dimension mapped to a runtime symbolic name). There is no third state. This is expressed with a new `MLDimension` type and a new `MLInputOperandDescriptor`:

```webidl
// A single input dimension: a number is static; a string is a named dynamic
// dimension. Every dynamic dimension carries a name.
typedef ([EnforceRange] unsigned long or DOMString) MLDimension;

dictionary MLInputOperandDescriptor {
  required MLOperandDataType dataType;
  required sequence<MLDimension> shape;
};
```

The empty string is **not** a valid dimension name; `input()` throws a `TypeError`. A name exists to relate dimensions to one another, and an empty name relates nothing — admitting it would either silently force every `""` dimension in the graph to resolve to the same value, or create a "half-named" dimension whose meaning differs from layer to layer. ONNX uses `dim_param == ""` to mean *anonymous*, so a caller reaching for `""` is asking for the state this design deliberately does not have.

A framework that receives anonymous dimensions from its own model format synthesizes names before calling WebNN. It already knows a stable identity for each one — the tensor's name and the axis index — so a name such as `"encoder_input_axis1"` is both free to produce and more useful in a diagnostic than an anonymous placeholder. (This is caller-side naming; it is distinct from the names a user agent synthesizes for *derived* dimensions, described in [Dimension semantics](#dimension-semantics).)

Reading a shape back, `MLOperand.shape` is widened accordingly: its elements may now be strings, and the whole attribute is `null` for an **unranked** operand — one whose rank is not yet known (see [Dimension semantics](#dimension-semantics)):

```webidl
interface MLOperand {
  readonly attribute MLOperandDataType dataType;
  // Was: FrozenArray<unsigned long> shape;
  readonly attribute FrozenArray<MLDimension>? shape;
};
```

Because every dynamic dimension is named, an element of this array is always either a number or a non-empty string — there is no sentinel value to interpret. The outer `null` is unrelated: it reports an unranked operand, not an unknown dimension.

Names that the user agent synthesized for derived dimensions are observable here, which is what makes them useful for debugging, but their content and format are implementation-defined and **not stable**: rebuilding the same graph, or a change in how the implementation walks it, may produce different names. They are diagnostic labels, not identifiers to be read back and fed into `input()` as a cross-tensor constraint.

The plain `MLOperandDescriptor` is deliberately **unchanged** (static-only): it describes `constant()` data, and `computeShapes()` likewise returns fully concrete (static) shapes. Dynamism thus lives only on graph *inputs* and propagates from there.

```webidl
// Unchanged — always concrete.
dictionary MLOperandDescriptor {
  required MLOperandDataType dataType;
  required sequence<[EnforceRange] unsigned long> shape;
};
```

### 2. `computeShapes()`
`computeShapes()` runs the same shape inference and shape computation as dispatch, but early and without executing the graph, returning the concrete output shape for each output given concrete input shapes:

```webidl
partial interface MLGraph {
  record<DOMString, sequence<[EnforceRange] unsigned long>> computeShapes(
      record<DOMString, sequence<[EnforceRange] unsigned long>> inputShapes);
};
```

The programming model becomes **build → (optionally) computeShapes → dispatch**.

This method enables frameworks to determine the output tensor sizes for dynamic subgraphs whose shapes are difficult to infer. Exposing it reduces redundant computation overhead, and it gives an implementation an opportunity to prepare for the dispatch that follows — see [Shape specialization and preparation](#shape-specialization-and-preparation).

### 3. Shape-as-data operators
Threading a dynamic input dimension through the graph is not sufficient on its own; real models compute with shapes. This proposal adds a family of operators that treat a shape as a runtime tensor, plus dynamic variants of existing operators that take their shape parameters as **operands** rather than build-time attributes.

```webidl
dictionary MLSqueezeOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> axes;
};

dictionary MLReshapeTo2dOptions : MLOperatorOptions {
  [EnforceRange] unsigned long axis = 1;
};

dictionary MLSliceDynamicOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> strides;
};

// Mirrors MLResample2dOptions, except `sizes` is a runtime operand.
dictionary MLResample2dDynamicOptions : MLOperatorOptions {
  MLInterpolationMode mode = "nearest-neighbor";
  sequence<float> scales;
  MLOperand sizes;
  sequence<[EnforceRange] unsigned long> axes;
};

partial interface MLGraphBuilder {
  // Read an operand's shape as a runtime uint32 1-D tensor.
  MLOperand shape(MLOperand input, optional MLOperatorOptions options = {});

  // Shape generators / arithmetic on shape tensors.
  MLOperand range(MLOperand start, MLOperand limit, MLOperand delta,
                   optional MLOperatorOptions options = {});
  MLOperand modulusFloor(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  MLOperand modulusTruncate(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});

  // Rank-changing operators (the seam where dynamic rank originates).
  MLOperand squeeze(MLOperand input, optional MLSqueezeOptions options = {});
  MLOperand unsqueeze(MLOperand input, sequence<[EnforceRange] unsigned long> axes,
                       optional MLOperatorOptions options = {});
  MLOperand reshapeTo2d(MLOperand input, optional MLReshapeTo2dOptions options = {});

  // Dynamic variants: shape parameters are operands, evaluated at dispatch.
  MLOperand reshapeDynamic(MLOperand input, MLOperand newShape, optional MLOperatorOptions options = {});
  MLOperand expandDynamic(MLOperand input, MLOperand newShape, optional MLOperatorOptions options = {});
  MLOperand sliceDynamic(MLOperand input, MLOperand starts, MLOperand sizes, optional MLSliceDynamicOptions options = {});
  MLOperand padDynamic(MLOperand input, MLOperand beginningPadding, MLOperand endingPadding, optional MLOperatorOptions options = {});

  sequence<MLOperand> splitDynamic(MLOperand input, MLOperand splits, optional MLSplitOptions options = {});
  MLOperand resample2dDynamic(MLOperand input, optional MLResample2dDynamicOptions options = {});
  MLOperand tileDynamic(MLOperand input, MLOperand repetitions, optional MLOperatorOptions options = {});
};
```

Three notes on the design of this family, rather than a per-operator walkthrough:

- **Each `*Dynamic` operator is a full dynamic mirror of its static counterpart.** `sliceDynamic` uses the same `starts + sizes (+ strides)` semantics as static `slice`; `splitDynamic` mirrors `split`'s explicit-splits form and its `axis` option. Keeping them one-to-one lets the static operators stay static-input-only and avoids divergent validation.

  `splitDynamic` deliberately has **no** equal-split (scalar count) overload: that count is a build-time constant even when the axis is dynamic, so static `split` already expresses it. The divisibility check it would normally perform at build time simply defers to dispatch under the three-valued predicates described below, so no dynamic variant is needed to make it work on a dynamic axis.

- **`squeeze` / `unsqueeze` / `reshapeTo2d` are the origin of dynamic rank.** E.g. a no-axes `squeeze` removes every size-1 dimension, so its output rank depends on runtime data. Emulating this logic requires complex subgraph chains to calculate runtime shapes, resulting in significant graph bloat and excessive shape inference overhead. Native operator support provides a more efficient and direct representation of these rank-changing transformations.

- **We do not adopt ONNX's -1 (auto-infer) / 0 (copy) reshape conventions.** These are framework-level conveniences that a framework can lower into a shape subgraph before calling WebNN, and not all runtimes support them; keeping them out of WebNN avoids baking one framework's convention into the platform. So frameworks must instead use a subgraph chain to calculate these dimensions dynamically:

For example, when the target shape is a runtime operand (values unknown at build time), the framework resolves the dimensions element-wise across the entire shape vector:

1. **Resolve 0:** `equal(targetShape, 0)` → `where(mask, shape(input), targetShape)` → `shapeNoZero`.
2. **Resolve -1:**
   - `total = reduceProduct(shape(input))`
   - `known = reduceProduct(where(equal(shapeNoZero, -1), 1, shapeNoZero))`
   - `inferred = div(total, known)`
3. **Final Assembly:** `where(isNeg1, inferred, shapeNoZero)` → `cast(uint32)` → `reshapeDynamic`.

*(Optimization: This chain may be skippable if the framework can prove the operand is already sentinel-free.)*

## Design Discussion
This section describes the runtime behavior that gives it meaning — the two mechanisms that carry dynamism at dispatch, **shape inference** and **shape computation**, plus the dimension semantics they honor. The two are easy to confuse by name, so to be precise: *shape inference* propagates each operand's **shape** forward across the whole graph, while *shape computation* evaluates one `shape()`-rooted chain down to the concrete **integer values** a shape parameter needs.

### Dimension semantics
- **Shared names are constraints.** Two dynamic dimensions with the **same name** are guaranteed to take the **same** concrete value throughout the graph (e.g. "query and key sequence lengths are equal"); the implementation enforces this across inputs and uses it to cancel dimensions in operations such as `reshape`. A name that appears only once constrains nothing. A dynamic dimension has **no min/max bound** — anything not provably static simply defers.

  Two kinds of names share one namespace: those the **caller** supplies, which may establish identity across tensors, and those the **user agent synthesizes** for derived dimensions, which identify a dimension without implying any relationship. Hence the isolation requirement below.

- **Derived dimensions get a unique synthesized name.** A caller-supplied name survives unchanged only on a 1:1 pass-through; any dimension *computed* from a dynamic one (a `concat` sum, a `conv2d` window, a `resample2d` scale) receives a fresh name that is unique within the graph. Uniqueness is what makes this safe: a name that is guaranteed to appear exactly once asserts no identity with any other dimension, so nothing is claimed that the runtime cannot honor — while the graph still reads back with a symbol at every position, showing where a pass-through ended.

- **Unranked operands.** For example, a no-axes `squeeze` removes every size-1 dimension, so its output rank depends on runtime data — the operand is unranked (its `MLOperand.shape` is null) until `computeShapes()`/`dispatch()` recovers it.

### Deferred validation
Validation splits across two phases:

- **At build time**, we validate only what is knowable without concrete shapes: data-type compatibility, rank constraints (e.g. `conv2d` needs rank 4), same-name symbolic consistency, and any **definite static contradiction** (e.g. reshaping a static `[2, 3]` to `[7]`).

- **At dispatch time** (and at `computeShapes()`), once concrete input shapes are known, we run **shape inference** — a forward propagation of each operand's concrete *shape* — over the whole graph, and validate the resulting concrete shapes against every constraint, including buffer sizes.

Deferring this work is the inherent trade-off of dynamic shapes: the shape resolution and validation a static graph completes once at build time now runs at inference time — as a gatekeeper on every `dispatch`, before the graph executes. This adds per-inference overhead, so a user agent should optimize the common cases: because the result is a pure function of the input shapes, it can skip re-validation when a dispatch repeats a set of input shapes it has already validated.

The build-time checks are expressed as **three-valued** dimension predicates. Instead of "equal / not-equal", a comparison is *provably-equal*, *provably-unequal*, or *unknown (defer)*. Only a provable contradiction is rejected at build time:

```cpp
// Reject only when BOTH dims are static AND differ; otherwise defer to dispatch.
bool DimensionsAreDefinitelyUnequal(Dimension a, Dimension b) {
    if (IsStatic(a) && IsStatic(b))
      return StaticValue(a) != StaticValue(b);
    return false;  // unknown -> defer
}
```

The same predicate is applied uniformly across operators that impose cross-dimension constraints, e.g. `matmul`'s contraction dimension, concat's non-concatenated axes, `reshape`'s element-count product, broadcasting, and so on.

### Shape computation at dispatch
*Shape computation* is the dispatch-time evaluation of a `shape()`-rooted chain down to the concrete values a shape parameter needs. It is the value-computing counterpart to *shape inference* ([Deferred validation](#deferred-validation)), which propagates operand shapes across the whole graph.

There is no blessed list of operators that may appear on a shape chain. In principle any operator can compute a shape — `matmul` reducing two 1-D vectors to an element count is a legitimate, if unusual, way to do it.

The set the prototype implements — arithmetic and structural transforms on shape tensors, with a little floating-point support for cases such as `reciprocal`, is therefore an implementation-cost and performance trade-off, not a design boundary, and the right balance is something to settle as the feature develops.

A small chain makes this concrete. To reshape a `[1, 'seqlen', 512]` tensor into `[1, 'seqlen', 8, 64]` (splitting the static hidden size into 8 heads × 64) while keeping the dynamic sequence length:

```js
const s = builder.shape(x);                               // uint32 1-D: the runtime dims of x
const batchSeq = builder.slice(s, [0], [2]);              // first two dims → [1, seqlen]
const heads = builder.constant(/*uint32*/ ..., [8, 64]);  // static tail
const newShape = builder.concat([batchSeq, heads], 0);    // [1, seqlen, 8, 64]
const y = builder.reshapeDynamic(x, newShape);            // y: [1, 'seqlen', 8, 64]
```

At dispatch with `seqlen = 37`, shape computation walks the `newShape` chain: `shape(x) `→` [1, 37, 512], slice `→` [1, 37], concat([1, 37], [8, 64]) `→` [1, 37, 8, 64]`. That computed value becomes reshapeDynamic's inferred output shape, `[1, 37, 8, 64]`. Note what it read: `x`'s **shape** and the **constant** `[8, 64]` — never `x`'s data.

What shape computation may read is bounded not by the operators on the chain but by its **root**: `shape()` outputs and build-time constants only. A chain that reaches an input's tensor **data** is unresolvable by design and is rejected — the out-of-scope [Tensor-Derived](#open-questions) case.

### Backends mapping
The dimension model maps cleanly onto all three backends, as shown below. The **shape-as-data operator family** is currently implemented only on the ORT backend.

#### ORT
- **Dynamic dimension** → ONNX symbolic name ([OrtApi::SetSymbolicDimensions](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#aa5b1654064d833a515f3acfcdcc5e81d)):

```cpp
WebNN: shape=['batch', 512]
ONNX:  shape=[dim_param:'batch', 512]
```

#### LiteRT
- **Dynamic dimension** → LiteRT unknown dimensions:

```cpp
WebNN:   shape=["batch", "height", 512]
LiteRT:  shape=[-1, -1, 512]
```

Represented in the LiteRT flatbuffer schema as:

```cpp
const flatbuffers::Offset<flatbuffers::Vector<int32_t>> dimensions
```

#### Core ML
- **Dynamic dimension** → Core ML unbounded ranges dimensions:

```cpp
WebNN:
  shape=["batch", "height", 512]
```

```cpp
// Core ML:
auto* size_range = shape_range->add_sizeranges();
size_range->set_lowerbound(1);
// set upper bound to maximum long value
size_range->set_upperbound(std::numeric_limits<int>::max());
```

## Considered Alternatives

### Per-operator build-time loosening
The first cut loosened each operator's build-time validator independently to tolerate dynamic dimensions. This scattered subtly different "is this okay?" logic across dozens of operators. We unified it behind the three-valued dimension predicates in [Deferred validation](#deferred-validation), so every operator defers the same definition of "unknown" and only rejects provable contradictions.

### Bespoke dynamic operators
We considered giving each `*Dynamic` operator a shape signature tailored to its most common use, rather than mirroring the static operator. We rejected it for the reasons in [Shape-as-data operators](#3-shape-as-data-operators): faithful mirroring keeps the mental model small and lets the static operators stay static-input-only.

## Privacy & Security Considerations
Dynamic shapes add no new fingerprinting or cross-origin surface: they expose no device or environment information a static graph does not, and shape computation and inference read only shapes the page itself supplied. The synthesized names for derived dimensions are newly observable output, but they are derived from the graph the page itself constructed (operation type, index, and axis) and carry nothing about the device or the environment; they are also explicitly unstable and not part of the API contract, so nothing may be inferred from a change in one. The considerations below are therefore about security.

The renderer is untrusted, so the service must remain safe on any graph a compromised renderer can construct, including ill-formed dynamic graphs:

- **No dereference of an absent rank.** Unranked operands (e.g. from a no-axes squeeze) are handled uniformly by each shared validator — propagate, resolve, or cleanly reject — and unranked *graph inputs* are rejected at build time (a graph input always has a known rank). A dispatch-time exit gate fails cleanly if any operand is still unranked after inference, so no unranked operand ever reaches a backend.

- **Shape computation never reads input data**, as described above, which also bounds what a graph can make the interpreter do.

- **Bounded constant collection.** The interpreter seeds only from the shape operands of the dynamic operators and walks back along the shape-computation chain, copying only the constants actually on that chain (and skipping oversized constants). Weight tensors are never on a shape chain and are not copied, bounding both memory and shape-computation work (a DoS/OOM guard).

## Future Consideration

### Bounded (min/max) dimensions
Currently, the proposed model is intentionally unbounded; a dynamic dimension is either provably static or deferred without a specific size range. As a future enhancement, we plan to allow dynamic dimensions to optionally declare a `minSize` and `maxSize` bound (and potentially an "optimal" size). For example: `{name: 'seqlen', minSize: 1, maxSize: 2048}`.

Rather than introducing a new source of dynamism, these bounds will serve as critical implementation hints to underlying runtimes, unlocking ahead-of-time memory allocations and graph optimizations that an unbounded model cannot achieve. Some runtime already accepts a constraint of this shape and acts on it. TensorRT's optimization profiles (min / opt / max), Core ML's `RangeDim`, and OpenVINO's bounded partial shapes, so the hint has somewhere to go rather than terminating in the user agent.

### Shape specialization and preparation
Knowing the shapes is not the same as being ready to run them: a backend may still have to plan memory, select kernels, or recompile. What that costs varies a great deal between runtimes (some absorb a shape change almost for free), while others may re-compile the graph. So on some backends a caller that changes shape often pays a real price.

`computeShapes()` gives an implementation a natural place to do that preparation, since the caller has just named the shapes it is about to run. That helps when a shape is then reused, but not when a caller keeps switching between shapes, and a user agent cannot tell which of them are worth keeping ready. One direction worth exploring is to let the caller say so: build with difference input sizes returning several `MLGraph`s, each bound to a set of concrete shapes and all sharing one copy of the weights.

### Fine-grained Shape Queries
Currently, the `shape()` operator returns an operand's entire shape as a 1D tensor. To provide more fine-grained shape retrieval, we may consider introducing two additional operators:

- `rank()` → Returns a 0D scalar tensor representing the operand's rank (similar to `tfl.rank`).

- `dimension(axis)` → Returns a 0D scalar tensor representing the size of a specific dimension (similar to StableHLO's `get_dimension_size`).

## Open Questions
- **Tensor-Derived sizes.** Whether, and how, to admit dimensions that depend on tensor *values*, which this proposal places out of scope.

- **Bidirectional broadcasting for Expand.** In dynamic shape scenarios, the shape operand of `expand` may contain flexible dimensions (e.g., `[1, 1, 1]`) that require bidirectional broadcasting against the input tensor. The original WebNN specification was limited to unidirectional broadcasting. Extending the `expand` operator to support bidirectional broadcasting aligns with the ONNX `Expand` operator and ensures correct handling of these dynamic shape cases.

## References & Acknowledgements
- [Support flexible input sizes #883](https://github.com/webmachinelearning/webnn/issues/883)

- [ORT API SetDimensions()](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a6575872736b924b47a382deb97e2fc17)

- [TFLite Flatbuffer Tensor & TfLiteTensor](https://developers.google.com/edge/api/tflite/c/struct/tf-lite-tensor)

- [Core ML Flexible Input Shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)

- Many thanks for valuable feedback and advice from:

  - [Ningxin Hu](mailto:ningxin.hu@intel.com)

  - [Dwayne Robinson](mailto:dwayner@microsoft.com)
