# Transformer Port Plan

## Scope

This document plans only the `SequenceProcessing.Classification.Transformer`
port.

It does not implement `Transformer` in this step.
It does not modify sibling repositories.

## Java Transformer Surface

Java source of truth:
- [Transformer.java](/home/cengiz/dev/ozu/SequenceProcessing/src/main/java/SequenceProcessing/Classification/Transformer.java)

### Java Fields

- `dictionary : VectorizedDictionary`
- `startIndex : int`
- `endIndex : int`
- inherited from `ComputationalGraph`:
  - `parameters`
  - `inputNodes`
  - `outputNode`

### Java Constructor

- `Transformer(NeuralNetworkParameter parameter, VectorizedDictionary dictionary)`

Constructor behavior:
- stores borrowed dictionary reference
- scans the dictionary for `<S>` and `</S>` token indices

### Java Helper Methods

- `positionalEncoding(Tensor tensor, int wordEmbeddingLength)`
- `createInputTensors(Tensor instance, ComputationalNode input1, ComputationalNode input2, int wordEmbeddingLength)`
- `layerNormalization(ComputationalNode input, TransformerParameter parameter, boolean isInput, int[] lnSize)`
- `multiHeadAttention(ComputationalNode input, TransformerParameter parameter, boolean isMasked, Random random)`
- `feedforwardNeuralNetwork(ComputationalNode current, int currentLayerSize, TransformerParameter parameter, Random random, boolean isInput)`
- `setInputNode(int bound, Vector vector, ComputationalNode node)`
- overridden `getOutputValue(ComputationalNode computationalNode)`

### Java Public Model Methods

- `train(ArrayList<Tensor> trainSet)`
- `test(ArrayList<Tensor> testSet)`

## What The Current SequenceProcessing-C Stack Already Supports

### Already Ported And Reusable

- `TransformerParameter`
  - constructor/getter parity already exists in
    [TransformerParameter.h](/home/cengiz/dev/ozu/SequenceProcessing-C/src/Parameters/TransformerParameter.h)
    and
    [TransformerParameter.c](/home/cengiz/dev/ozu/SequenceProcessing-C/src/Parameters/TransformerParameter.c)
- custom Transformer-used functions already ported:
  - `Mask`
  - `Mean`
  - `Variance`
  - `SquareRoot`
  - `Inverse`
  - `Transpose`
  - `MultiplyByConstant`
- shared local helpers already exist:
  - Java-compatible RNG via
    [JavaRandomCompat.h](/home/cengiz/dev/ozu/SequenceProcessing-C/src/Classification/JavaRandomCompat.h)
  - borrowed activation proxy via
    [BorrowedFunctionProxy.h](/home/cengiz/dev/ozu/SequenceProcessing-C/src/Classification/BorrowedFunctionProxy.h)
- graph wrappers already exist for:
  - input-node registration
  - unary function-edge insertion
  - multiplication edges
  - hadamard edges
  - addition edges
  - concatenation
  - output-node registration
  - forward/predict
  - class-index backprop

### Sibling Types That Exist Locally

- `Vectorized_dictionary` exists in `Dictionary-C`
- `Vectorized_word` exists in `Dictionary-C`
- `Vector` exists in `Math-C`
- dictionary indexing and lookup primitives exist in `Dictionary-C`

That means the dictionary/token side is available enough for a Transformer shell
and helper ports.

## Exact Transformer Parts That Still Need New Local Helpers

### 1. Non-Recurrent Model Bridge

Current graph compatibility is packaged as a recurrent-oriented bridge:
- [RecurrentModelGraphBridge.h](/home/cengiz/dev/ozu/SequenceProcessing-C/src/Classification/RecurrentModelGraphBridge.h)

Transformer does not need:
- recurrent `switches`
- recurrent time-step input allocation semantics

Transformer does need:
- multiple ordinary input nodes
- output extractor
- forward/predict/backprop wrappers
- graph-owned lifecycle

Recommended helper:
- a narrow local `TransformerModelGraphBridge` or a generalized
  non-recurrent graph bridge extracted from the recurrent one

This is not an upstream blocker. It is a local structuring task.

### 2. Borrowed-Function Attachment For TransformerParameter Activations

`feedforwardNeuralNetwork(...)` and parts of attention/layer-normalization
attach activation functions borrowed from `TransformerParameter`.

Current status:
- already solved by `BorrowedFunctionProxy`

No new architecture is needed here.

### 3. Transformer-Specific Weight Node Helper Usage

Transformer uses repeated Java `Random`-driven weight initialization just like
GRU/LSTM.

Current status:
- shared Java-compatible RNG exists
- recurrent weight-node helper exists

Likely need:
- either reuse the existing weight-node helper from the recurrent base
- or extract it into a smaller shared model utility if keeping Transformer
  independent from recurrent structs

This is a local refactor/placement issue, not a mathematical blocker.

### 4. Input Packing Helpers

Transformer has two model-specific input helpers that do not exist yet in C:

- encoder/decoder split using `Double.MAX_VALUE` sentinel
- positional encoding

These are local helper ports and are safe to implement later.

### 5. Layer Normalization Helper

`layerNormalization(...)` is graph-construction heavy but uses only already
ported functions and standard graph edges.

Missing piece:
- a Transformer-local helper to allocate gamma/beta learnable nodes from
  `TransformerParameter`

This is local model glue, not an upstream blocker.

### 6. Decoder Test-Time Auto-Regressive Loop

Java `test(...)` repeatedly:
- mutates decoder input node value
- calls `predict()`
- reads the last class label
- stops on `</S>`

The current stack can likely support this structurally, but no local
Transformer shell exists yet to own:
- dictionary reference
- start/end token indices
- encoder input node
- decoder input node
- class-label node

This is a model-shell task, not a graph-engine blocker.

## Are The Current Function Ports Sufficient?

For Java helper and forward graph construction:
- yes, mostly

Specifically already covered:
- `Mean`
- `Variance`
- `SquareRoot`
- `Inverse`
- `Transpose`
- `MultiplyByConstant`
- `Mask`

Also available from `ComputationalGraph-C`:
- `Softmax`
- `Tanh`
- `Negation`

Remaining caution:
- the custom function ports intentionally mirror Java behavior, including the
  nonstandard derivative formulas in `Inverse`, `SquareRoot`, and `Variance`
- that is acceptable because the port target is Java parity, not independent
  mathematical correction

Conclusion:
- the current function layer is sufficient for a Transformer shell and forward
  helper work
- no new custom function class is currently missing

## Are Current Graph Execution Semantics Sufficient?

### Forward Graph Construction

- yes, for the unary-function and binary-operator pieces used by Transformer
- attention, layer norm, feedforward, and residual paths are expressible with
  the currently wrapped edge types

### Training Flow

- partially yes

The same deliberate compromise used for GRU/LSTM applies here:
- class-index backprop is available
- multi-input loss-node forward parity is not

For grounded training parity:
- Java training still calls `forwardCalculation()` and `backpropagation()`
- `ComputationalGraph-C` class-index backprop is a defensible local substitute
  only if we continue to explicitly defer loss-node forward-value parity

So:
- training can likely be ported with the same documented limitation as GRU/LSTM
- no new Transformer-only training blocker is visible beyond the already known
  loss-node forward mismatch

### Test Flow

- partially yes

The graph engine can likely support repeated `predict()` calls with updated
decoder input values, but this should be validated only after a shell exists.

Main non-graph dependency:
- dictionary-driven decoding and `<S>` / `</S>` lookup

## Exact Known Blockers

### Blocker 1: No Transformer-Specific Model Shell

There is no local `Transformer` model struct yet to own:
- borrowed `TransformerParameter`
- borrowed `Vectorized_dictionary`
- start/end token indices
- graph bridge
- encoder/decoder/class-label input nodes

### Blocker 2: Shared Model Bridge Placement

The only local model bridge today is recurrent-specific.

Before implementing Transformer cleanly, choose one:
- extract a smaller generic graph bridge
- or add a separate Transformer-local bridge reusing the same low-level pattern

### Blocker 3: ClassificationPerformance Boundary

Java `test(...)` returns `ClassificationPerformance`.
That type boundary is still unresolved in the local C stack.

This blocks full Java `test(...)` parity, not a constructor shell or partial
forward/training path.

### Blocker 4: Loss-Node Forward Parity

Still deferred by design.

Transformer training can likely proceed with class-index backprop, but the plan
must continue to state clearly:
- no claim of multi-input loss-node forward parity

## Recommended Implementation Order

### Phase 1: Shell Only

Add:
- `src/Classification/Transformer.h`
- `src/Classification/Transformer.c`

Implement only:
- model struct
- constructor
- destructor
- dictionary scan for `<S>` / `</S>`
- borrowed base fields and ownership docs

This is the safest first slice.

### Phase 2: Helper Layers

Implement local helpers only:
- positional encoding
- encoder/decoder input packing from flattened sequence tensor
- `getOutputValue(...)`
- dictionary token lookup helpers
- possibly a small non-recurrent graph bridge

This phase is still low-risk and testable without committing to full training.

### Phase 3: Partial Forward Path

Implement graph construction helpers:
- `layerNormalization(...)`
- `multiHeadAttention(...)`
- `feedforwardNeuralNetwork(...)`

Recommended scope:
- build graph on a fresh Transformer instance
- no `test(...)` yet
- no claim of full train/test parity yet

### Phase 4: Train Later

Implement `train(...)` only after:
- the shell exists
- helper layers exist
- bridge placement is settled

Training should explicitly reuse:
- Java-compatible RNG
- borrowed-function proxies
- class-index backprop path

### Phase 5: Test Last

Implement `test(...)` only after:
- decoder auto-regressive loop is validated
- dictionary integration is stable
- `ClassificationPerformance` boundary is resolved or intentionally wrapped

## Recommended Smallest First Transformer Slice

Start with:
- a Transformer shell only

Concrete scope:
- constructor
- destructor
- borrowed dictionary
- `startIndex`
- `endIndex`
- no training
- no test
- no graph construction yet

That slice is clearly grounded, low-risk, and gives a clean anchor for later
helper ports.

## Readiness Assessment

Transformer is not ready for a full implementation in one step.

Transformer is ready to start in a narrow, staged way:
- shell now
- helper layers next
- partial forward graph after that
- train/test later

## Recommendation

Recommended next Transformer step:
- implement the shell only, or at most shell plus pure local helpers

Do not jump directly to full `train(...)` or `test(...)`.
