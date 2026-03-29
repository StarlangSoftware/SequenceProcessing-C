# Model Port Plan

## Current Recurrent Base Status

`RecurrentNeuralNetworkModel` is only partially ported.

Implemented local helper parity:
- constructor-owned model shell
- `createInputTensors(...)`
- `findTimeStep(...)`
- `getOutputValue(...)`

Deferred:
- graph construction in `train(...)`
- epoch training loop
- loss-node insertion
- prediction/test integration

## Exact Missing Shared Recurrent / Model Operations

The remaining Java parity depends on a small shared recurrent/model bridge on
top of `ComputationalGraph-C`.

### 1. Graph-Owned Input Node Registration

Java uses inherited `inputNodes` directly from the graph object. The local C
slice currently owns `input_nodes` inside `Recurrent_neural_network_model`
instead of registering them in a reusable graph wrapper.

Minimal missing operation:
- create an input node
- append it to both model-owned state and the underlying graph input-node list

### 2. Generic Function-Edge / Loss-Node Insertion

The local `SequenceFunctionEdge` helper only supports:
- one parent node
- one `Function*`
- one returned child node

The Java recurrent base also needs the inherited graph helper that inserts a
loss node over multiple input nodes:
- output node
- class-label node
- loss function

Minimal missing operation:
- local wrapper for "attach function node over N inputs"
- for the current recurrent base, the immediate case is `N = 2` for the loss
  node

### 3. Shared Recurrent Base Graph Wrapper

Java `RecurrentNeuralNetworkModel` extends `ComputationalGraph` and relies on
inherited operations such as:
- `addEdge(...)`
- `addAdditionEdge(...)`
- `concatEdges(...)`
- `forwardCalculation()`
- `backpropagation()`
- `predict()`

`ComputationalGraph-C` exposes low-level free functions for most of this, but
not as a ready local model wrapper that also carries:
- borrowed parameter pointer
- owned switch list
- owned input-node setup state
- output extraction callback

Minimal missing operation:
- a local recurrent-base compatibility struct that owns or borrows a
  `Computational_graph_ptr`
- thin wrappers around the relevant graph operations
- one local place to keep recurrent-model state beside the graph

### 4. Classification Performance Boundary

Java `test(...)` returns `ClassificationPerformance`.
That return type is not cleanly consumable from the current local slice because
`ComputationalGraph-C/src/ComputationalGraph.h` includes
`Performance/ClassificationPerformance.h`, which is not currently available in
the local include path.

Minimal missing operation:
- either provide the missing local dependency cleanly
- or defer `test(...)` until that sibling API is available

## Recommended Next Step

Build a local recurrent-base compatibility layer first.

That layer should be narrow:
- wrap a `Computational_graph_ptr`
- own the recurrent helper state already ported
- expose thin wrappers for input-node creation, edge insertion, addition,
  concatenation, forward, backward, and prediction
- add one local multi-input function-edge helper for the loss node

GRU and LSTM should wait until that bridge exists so they can share the same
base setup rather than reimplement graph plumbing independently.
