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
    - [Dimension semantics](#dimension-semantics)
    - [Deferred validation](#deferred-validation)
    - [Shape computation at dispatch](#shape-computation-at-dispatch)
    - [Backends mapping](#backends-mapping)
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
A framework such as ONNX Runtime Web partitions a model and hands WebNN a subgraph. That subgraph's output can carry a dynamic dimension that is not present on any of its inputs — it is derived inside the subgraph, and so carries a name the implementation synthesized rather than one the framework supplied. To allocate the output tensor, the framework needs the concrete output shape before it dispatches:

```js
const outShapes = graph.computeShapes({'attention_mask': [1, 37]});
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

A framework that receives anonymous dimensions from its own model format synthesizes names before calling WebNN. It already knows a stable identity for each one — the tensor's name and the axis index — so a name such as `"encoder_input_axis1"` is both free to produce and more useful in a diagnostic than an anonymous placeholder. (This is caller-side naming; it is distinct from the names an implementation synthesizes for *derived* dimensions, described in [Dimension semantics](#dimension-semantics).)

Reading a shape back, `MLOperand.shape` is widened accordingly: its elements may now be strings, and the whole attribute is `null` for an **unranked** operand — one whose rank is not yet known (see [Dimension semantics](#dimension-semantics)):

```webidl
interface MLOperand {
  readonly attribute MLOperandDataType dataType;
  // Was: FrozenArray<unsigned long> shape;
  readonly attribute FrozenArray<MLDimension>? shape;
};
```

Because every dynamic dimension is named, an element of this array is always either a number or a non-empty string — there is no sentinel value to interpret. The outer `null` is unrelated: it reports an unranked operand, not an unknown dimension.

Names that the implementation synthesized for derived dimensions are observable here, which is what makes them useful for debugging, but their content and format are implementation-defined and **not stable**: rebuilding the same graph, or a change in how the implementation walks it, may produce different names. They are diagnostic labels, not identifiers to be read back and fed into `input()` as a cross-tensor constraint.

The plain `MLOperandDescriptor` is deliberately **unchanged** (static-only): it describes `constant()` data, and `computeShapes()` likewise returns fully concrete (static) shapes. Dynamism thus lives only on graph *inputs* and propagates from there.

```webidl
// Unchanged — always concrete.
dictionary MLOperandDescriptor {
  required MLOperandDataType dataType;
  required sequence<[EnforceRange] unsigned long> shape;
};
```

### 2. `computeShapes()`
`computeShapes()` runs the same shape inference and shape computation as dispatch, but early and without executing the graph: one forward pass over the graph resolves every operand, and it returns the concrete shape of each output for the given input shapes:

```webidl
typedef record<USVString, sequence<[EnforceRange] unsigned long>> MLNamedShapes;

partial interface MLGraph {
  MLNamedShapes computeShapes(MLNamedShapes inputShapes);
};
```

The programming model becomes **build → (optionally) computeShapes → dispatch**. `dispatch()` resolves shapes itself regardless: `computeShapes()` is optional, and the tensors actually bound at dispatch may differ from the shapes it was asked about. Because the result is a pure function of the input shapes, an implementation can cache it, so a repeat is a lookup rather than a second pass.

Frameworks need this because `dispatch()` takes caller-allocated output tensors: the output shape has to be known *before* the call, not during it. Without `computeShapes()`, a framework would have to reimplement WebNN's shape inference just to size those allocations — and for a subgraph whose dynamic dimensions are derived internally, that means reproducing the whole shape chain. The method exposes the resolution that `dispatch()` performs anyway, early enough to be useful, and it gives an implementation an opportunity to prepare for a dispatch that may follow — see [Shape specialization and preparation](#shape-specialization-and-preparation).

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

  // Sequence generator: output length follows from start, limit and delta.
  MLOperand range(MLOperand start, MLOperand limit, MLOperand delta,
                  optional MLOperatorOptions options = {});

  // Arithmetic on shape tensors.
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
This section describes the runtime behavior behind the proposed API and the rules that dynamic dimensions follow. Two mechanisms handle dynamic shapes:

- **Shape inference** is a pass over the whole graph that computes each operand's shape. At build time it works with symbolic dimensions; at dispatch time it works with concrete ones.

- **Shape computation** is a step inside shape inference. When shape inference reaches `reshapeDynamic(x, newShape)`, the shape of `newShape` (for example, a 1-D tensor of length 4) says nothing about the output; the output shape is given by the *values* in `newShape`. So shape inference evaluates the chain that produces `newShape` down to concrete integers, and uses the result as the output shape.

### Dimension semantics
- **Shared names are constraints.** Two dynamic dimensions with the **same name** are guaranteed to take the **same** concrete value throughout the graph (e.g. "query and key sequence lengths are equal"); the implementation enforces this across inputs and uses it to cancel dimensions in operations such as `reshape`. A name that appears only once constrains nothing. A dynamic dimension has **no min/max bound** — anything not provably static simply defers.

  Two kinds of names share one namespace: those the **caller** supplies, which may establish identity across tensors, and those the **implementation synthesizes** for derived dimensions, which identify a dimension without implying any relationship. The next rule keeps synthesized names from colliding with caller-supplied ones.

- **Derived dimensions get a unique synthesized name.** A caller-supplied name survives unchanged only on a 1:1 pass-through; any dimension *computed* from a dynamic one (a `concat` sum, a `conv2d` window, a `resample2d` scale) receives a fresh name that is unique within the graph. Uniqueness is what makes this safe: a name that is guaranteed to appear exactly once asserts no identity with any other dimension, so nothing is claimed that the runtime cannot honor — while the graph still reads back with a symbol at every position, showing where a pass-through ended.

- **Unranked operands.** For example, a no-axes `squeeze` removes every size-1 dimension, so its output rank depends on runtime data — the operand is unranked (its `MLOperand.shape` is null) until `computeShapes()`/`dispatch()` recovers it.

### Deferred validation
Validation splits across two phases:

- **At build time**, shapes propagate symbolically — every operand gets a shape expressed in names and static sizes — and we validate only what is knowable without concrete shapes: data-type compatibility, rank constraints (e.g. `conv2d` needs rank 4), same-name symbolic consistency, and any **definite static contradiction** (e.g. reshaping a static `[2, 3]` to `[7]`).

- **At dispatch time** (and at `computeShapes()`), once concrete input shapes are known, we run **shape inference**: the same forward propagation, now carrying each operand's concrete *shape* across the whole graph. We then validate the resulting concrete shapes against every constraint, including buffer sizes.

Some checks can't run until concrete shapes are known. How many checks wait until then is a design choice, and this proposal defers nearly all of them. So inference now computes the shape resolution and shape validation that would have previously completed at build time, preceding every `dispatch`. This adds overhead to each dispatch. Since the result depends only on the input shapes, an implementation can cache it and skip validation when a dispatch repeats input shapes that it has already validated.

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

An implementation **must not** reject a graph if there exists a set of input shapes that makes the graph valid. The predicate above meets this requirement with the fewest checks: it only fails at build time when two static dimensions differ. An implementation that tracks symbolic *expressions* may reject more graphs, and earlier. For example, concatenating two `[batch, seq]` tensors along axis 1 gives `[batch, 2 * seq]`, so reshaping the result back to `[batch, seq]` can never succeed. This proposal only sees a new name for the concatenated axis, so it defers the check to dispatch. An implementation that knows the axis is `2 * seq` may fail `build()` instead, as some backends already do. This doesn't change which graphs can run, since that graph would fail at every dispatch anyway. It does mean that a successful `build()` doesn't guarantee a graph will ever run; it only means that this implementation couldn't prove the graph invalid.

### Shape computation at dispatch
*Shape computation* is performed on the CPU by a **shape interpreter**, which evaluates a shape chain down to the concrete integer values a shape parameter needs. Shape inference ([Deferred validation](#deferred-validation)) invokes it at dispatch each time it reaches a `*Dynamic` operator, and uses the values it returns as that operator's output shape.

Together, the shape chains in a graph form a subgraph. Every chain in it has a fixed start and end:

- A chain **ends** at an operator whose output shape is set by the values the chain computes: the `*Dynamic` operators, and `range()`, whose output length follows from `start`, `limit` and `delta`. This is what makes it a shape chain, and it's the only way a computed value can affect a shape.

- A chain **begins** at a `shape()` output or a build-time constant, and nowhere else, because these are the only values the interpreter can read.

The operators in between are ordinary operators, evaluated on small integer vectors.

Two consequences follow from this: A shape can never depend on a value that the graph computes from tensor data, and `computeShapes()` is a pure function of the input *shapes* - the same input shapes always give the same output shapes, whatever the tensors contain.

Identifying this subgraph doesn't partition the graph. Its operators stay in the graph that's handed to the backend and run there as usual. The interpreter's evaluation is a separate computation on the CPU that resolves shapes before dispatch.

Which operators may appear on a shape chain? This is really two questions: what a graph may *express*, and what an implementation must be able to *resolve*.

Operator identity doesn't restrict the first. In principle any operator can compute a shape: for example, `matmul` of two 1-D vectors is an unusual but valid way to compute an element count. Only the chain's start is restricted, as described above, and that restriction is what prevents a shape chain from computing on the model's tensor data.

The second question is where interoperability comes in: a shape that resolves in one implementation should resolve in another, so implementations need a common required minimum. ONNX defines a deliberately narrow closed set: the operators that have a data-propagation function (`Add`, `Cast`, `Concat`, `Gather`, `Mul`, `Shape`, `Size`, `Slice`, `Squeeze`, `Sub`, `Unsqueeze`). ORT's [`symbolic_shape_infer.py`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/symbolic_shape_infer.py) covers a much wider set in practice. WebNN's required minimum belongs between the two. This proposal doesn't define that set yet, and still needs to. The intent is for it to be normative and extensible: later revisions can add operators to it, the same way they add operators to WebNN, and implementations may resolve more than the minimum. The prototype today implements a strict superset of the ONNX set: arithmetic and structural operators on shape vectors, plus enough floating-point support for cases such as `reciprocal`.

Beyond that minimum, resolving more is a quality-of-implementation matter, with one requirement: an implementation that can't evaluate a chain must fail with a clear "cannot resolve this shape" error. It must not run the chain on the accelerator or guess a value. Heavy operators such as `conv2d`, `matmul` on model tensors, or attention aren't expected to be resolvable, and an implementation may reject them on a shape chain. Evaluating a chain is integer work on small vectors on the CPU; it doesn't execute the graph. These properties (small, integer, 1-D, evaluated on the CPU without touching the device) follow from where chains start and end, not from the operator list, so they hold even when a chain uses an operator that no implementation resolves.

A small chain makes this concrete. To reshape a `[1, 'seqlen', 512]` tensor into `[1, 'seqlen', 8, 64]` (splitting the static hidden size into 8 heads × 64) while keeping the dynamic sequence length:

```js
const s = builder.shape(x);                               // uint32 1-D: the runtime dims of x
const batchSeq = builder.slice(s, [0], [2]);              // first two dims → [1, seqlen]
const heads = builder.constant(/*uint32*/ ..., [8, 64]);  // static tail
const newShape = builder.concat([batchSeq, heads], 0);    // [1, seqlen, 8, 64]
const y = builder.reshapeDynamic(x, newShape);            // y: [1, 'seqlen', 8, 64]
```

At dispatch with `seqlen = 37`, the interpreter walks the `newShape` chain: `shape(x)` → `[1, 37, 512]`, `slice` → `[1, 37]`, `concat([1, 37], [8, 64])` → `[1, 37, 8, 64]`. That computed value becomes reshapeDynamic's inferred output shape, `[1, 37, 8, 64]`. Note what it read: `x`'s **shape** and the **constant** `[8, 64]` — never `x`'s data.

A chain that reaches an input's tensor **data** is unresolvable by design and is rejected — the out-of-scope [Tensor-Derived](#open-questions) case.

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

### Symbolic shape expressions
A stronger alternative is to carry a symbolic *expression* for every dimension instead of an opaque name. Build time could then prove or disprove a graph's shape constraints and report the constraints that a model needs (such as divisibility or bounds), and `computeShapes()` would only have to substitute values into the expressions. TensorRT and NNEF's SkriptND both work this way.

We start with opaque names for three reasons:

- Some shapes have no closed-form expression. With shape-as-data operators, a target shape is a runtime operand produced by an arbitrary chain, which can include value-dependent selection such as the `where` used to lower ONNX's `-1`. A no-axes `squeeze` even makes the rank depend on the input shapes. Symbolic inference could cover a large subset of graphs, but it couldn't replace validation at dispatch.

- Backends can't consume expressions: ONNX takes a `dim_param` string, LiteRT takes `-1`, and Core ML takes a `RangeDim`. Expressions would only exist inside WebNN.

- The graph comes from an untrusted renderer, so proving constraints at build time is attacker-influenced work in a privileged process. A time limit would make build results depend on the machine, so a deterministic limit on expression size and depth would be needed instead.

Expressions remain attractive as a later layer on top of opaque names, for diagnostics and earlier rejection. They'd work well together with [Bounded (min/max) dimensions](#bounded-minmax-dimensions), which provide the input constraints that most build-time proofs need.

## Privacy & Security Considerations
Dynamic shapes add no new fingerprinting or cross-origin surface through the values the API returns: none of them carries device or environment information that a static graph wouldn't already expose, and shape computation and inference only read shapes that the page itself supplied. The synthesized names for derived dimensions are new output, but they're derived from the graph that the page built (operation type, index, and axis) and carry nothing about the device or the environment. They're also explicitly unstable and not part of the API contract, so a caller can't infer anything from a change in one.

However, dynamic shapes do widen an existing **timing** surface. Because one graph now accepts many shapes, a page can dispatch it over a range of shapes and time each dispatch. This gives a cost *curve*, whereas a static graph only exposes the duration of its `build()`. Jumps in that curve, such as an alignment or tiling threshold, or a shape that falls back to a slower path, can hint at the backend or the class of hardware. Compilation time in `build()`, and shader compilation in other web APIs, already expose the same kind of signal. But when WebNN runs on an accelerator that other web APIs can't reach, those APIs don't expose this signal. It can't be fully mitigated in an API that exists to run computation on device-specific accelerators, since running in constant time across all shapes isn't realistic.

Two things limit it. First, the work that WebNN itself adds at dispatch (shape inference and validation) doesn't depend on tensor data, and its result is cached per set of input shapes, so a repeated shape costs the same each time. The part that varies is the backend re-specializing for a new shape, which the underlying runtime does with or without WebNN. Second, explicit specialization ([Shape specialization and preparation](#shape-specialization-and-preparation)) would move that cost into a step that the caller requested, instead of leaving it implicit at dispatch.

The considerations below are about security.

The renderer is untrusted, so the service must remain safe on any graph a compromised renderer can construct, including ill-formed dynamic graphs:

- **No dereference of an absent rank.** Unranked operands (e.g. from a no-axes squeeze) are handled uniformly by each shared validator — propagate, resolve, or cleanly reject — and unranked *graph inputs* are rejected at build time (a graph input always has a known rank). A dispatch-time exit gate fails cleanly if any operand is still unranked after shape inference, so no unranked operand ever reaches a backend.

- **Shape computation never reads input data**, as described above, which also bounds what a graph can make the interpreter do.

- **Bounded constant collection.** The constants that the interpreter may need are gathered ahead of time: the collector seeds from the shape operands of the dynamic operators, walks back along the chain, and copies only the constants it finds there. Current weight tensors are not on a shape chain and are not copied, which keeps the memory and work involved independent of the model's size (a DoS/OOM guard).

## Future Consideration

### Bounded (min/max) dimensions
Currently, the proposed model is intentionally unbounded; a dynamic dimension is either provably static or deferred without a specific size range. As a future enhancement, we plan to allow dynamic dimensions to optionally declare a `minSize` and `maxSize` bound (and potentially an "optimal" size). For example: `{name: 'seqlen', minSize: 1, maxSize: 2048}`.

Rather than introducing a new source of dynamism, these bounds will serve as critical implementation hints to underlying runtimes, unlocking ahead-of-time memory allocations and graph optimizations that an unbounded model cannot achieve. Several runtimes already accept a constraint of this shape and act on it — TensorRT's optimization profiles (min / opt / max), Core ML's `RangeDim`, and OpenVINO's bounded partial shapes — so the hint has somewhere to go rather than stopping at the WebNN layer.

### Shape specialization and preparation
Knowing the shapes is not the same as being ready to run them: a backend may still have to plan memory, select kernels, or recompile. What that costs varies a great deal between runtimes: some absorb a shape change almost for free, while others re-compile the graph. So on some backends a caller that changes shape often pays a real price.

`computeShapes()` gives an implementation a natural place to do that preparation, since the caller has just named a set of concrete shapes. However, a caller may call `computeShapes()` without dispatching afterwards (for example, to compare candidate output shapes), and the call is synchronous, which limits how much work it can do. Shape preparation only pays off if the shapes are reused, and an implementation doesn't know which shapes are worth keeping ready. One direction worth exploring is to let the caller say so: build with different input sizes returning several `MLGraph`s, each bound to a set of concrete shapes and all sharing one copy of the weights.

### Fine-grained Shape Queries
Currently, the `shape()` operator returns an operand's entire shape as a 1D tensor. To provide more fine-grained shape retrieval, we may consider introducing two additional operators:

- `rank()` → Returns a 0D scalar tensor representing the operand's rank (similar to `tfl.rank`).

- `dimension(axis)` → Returns a 0D scalar tensor representing the size of a specific dimension (similar to StableHLO's `get_dimension_size`).

## Open Questions
- **Tensor-Derived sizes.** Dimensions that depend on tensor *values*, such as the output size of `NonZero`, are out of scope for this proposal. A shape chain that traces back to tensor data, rather than to `shape()` outputs and build-time constants, is rejected. Supporting these dimensions needs more than a stronger interpreter. The output would have to be allocated before its size is known, which requires an upper bound to allocate against (see [Bounded (min/max) dimensions](#bounded-minmax-dimensions)) and a way to report the size actually produced. `computeShapes()` also couldn't answer for such an output, because its size isn't known until the graph runs.

- **Making the shape subgraph explicit.** Values on a shape chain are already separate from tensor data: a chain starts only at `shape()` outputs and build-time constants, so nothing on it reads a tensor, and an implementation evaluates it on the CPU without executing the graph. But this separation is *implicit*. It follows from where chains start, not from the type system, and `MLOperand` is used for both kinds of values. This has already led readers to expect that `computeShapes()` might have to run the model. Two options would make the separation visible: an operand property such as `isShape`, which documents it without adding constraints, or a separate `MLShapeOperand` type, which enforces it in the type system.

- **Bidirectional broadcasting for Expand.** In dynamic shape scenarios, the shape operand of `expand` may contain flexible dimensions (e.g., `[1, 1, 1]`) that require bidirectional broadcasting against the input tensor. The original WebNN specification was limited to unidirectional broadcasting. Extending the `expand` operator to support bidirectional broadcasting aligns with the ONNX `Expand` operator and ensures correct handling of these dynamic shape cases.

## References & Acknowledgements
- [Support flexible input sizes #883](https://github.com/webmachinelearning/webnn/issues/883)

- [ORT API SetDimensions()](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a6575872736b924b47a382deb97e2fc17)

- [TFLite Flatbuffer Tensor & TfLiteTensor](https://developers.google.com/edge/api/tflite/c/struct/tf-lite-tensor)

- [Core ML Flexible Input Shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)

- Many thanks for valuable feedback and advice from:

  - [Ningxin Hu](mailto:ningxin.hu@intel.com)

  - [Dwayne Robinson](mailto:dwayner@microsoft.com)
