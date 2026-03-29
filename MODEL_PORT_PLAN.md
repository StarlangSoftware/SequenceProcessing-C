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
- full GRU/LSTM train parity

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

Current blocker after inspecting `ComputationalGraph-C`:
- the C execution model does not currently support a true multi-input function
  node the way Java/C++ do for the recurrent loss path
- in `ComputationalGraph-C`, a node with `function != NULL` is evaluated from
  the first arriving parent value only; later parents are combined through the
  generic non-function accumulation path
- that means wiring a two-parent loss node locally would not execute as
  `loss(output, classLabel)`; it would execute as `loss(output)` followed by
  tensor addition with the class-label tensor
- backprop in `ComputationalGraph-C` also starts from class-label indices via
  `calculate_r_minus_y(...)`, not from a graph loss node

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

Current status:
- this bridge now exists locally for input-node registration, unary function
  edges, multiplication/addition/hadamard edges, concatenation, output-node
  registration, and forward/predict
- it is still intentionally incomplete for loss-node insertion and training
  because of the execution-model mismatch above

### 4. Classification Performance Boundary

Java `test(...)` returns `ClassificationPerformance`.
That return type is not cleanly consumable from the current local slice because
`ComputationalGraph-C/src/ComputationalGraph.h` includes
`Performance/ClassificationPerformance.h`, which is not currently available in
the local include path.

Minimal missing operation:
- either provide the missing local dependency cleanly
- or defer `test(...)` until that sibling API is available

### 5. Weight Initialization Bridge

GRU/LSTM Java training uses `parameters.initializeWeights(row, column, random)`
repeatedly with a shared `Random` instance.

Current blocker:
- `ComputationalGraph-C` exposes only stateless initialization helpers such as
  `random_initialization(...)`, `he_uniform_initialization(...)`, and
  `uniform_xavier_initialization(...)`
- there is no shared local helper yet that maps the parameter's
  `Initialization` enum plus evolving Java `Random` usage into the same call
  sequence for recurrent model weight creation

Minimal missing operation:
- one local helper that creates weight tensors from
  `(row, column, parameter->initialization, evolving_seed_or_rng_state)`
- this is needed for grounded GRU/LSTM train parity once the loss/training
  bridge problem is solved

## Recommended Next Step

Build a local recurrent-base compatibility layer first.

That layer should be narrow:
- wrap a `Computational_graph_ptr`
- own the recurrent helper state already ported
- expose thin wrappers for input-node creation, edge insertion, addition,
  concatenation, forward, backward, and prediction
- add one local multi-input function-edge helper for the loss node

Updated recommendation after inspection:
- stop short of claiming full recurrent training parity with the current
  `ComputationalGraph-C` execution model
- GRU/LSTM graph shells can reuse the existing bridge
- full GRU/LSTM `train(...)` should wait until one of these happens:
  - `ComputationalGraph-C` grows a true multi-input function-node execution
    path compatible with Java/C++
  - or `SequenceProcessing-C` implements a local alternative training path that
    intentionally bypasses graph-loss-node parity and is accepted as a design
    divergence
