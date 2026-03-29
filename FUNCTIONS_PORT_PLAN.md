# Functions Layer Port Plan

## Scope

This document plans only the `SequenceProcessing.Functions` layer.
It does not implement the functions.
It does not port model classes.
It does not modify `ComputationalGraph-C`.

## Java-Side Abstraction Pattern

### Function Interface

Java `ComputationalGraph.Function.Function` defines three responsibilities:

- `Tensor calculate(Tensor value)`
- `Tensor derivative(Tensor value, Tensor backward)`
- `ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased)`

### FunctionNode Pattern

Each `SequenceProcessing.Functions.*` class follows the same graph-construction pattern:

1. create a new `FunctionNode(isBiased, this)`
2. connect it to the first input node with `inputNodes.get(0).add(newNode)`
3. return the new node

### Evaluation / Backward Hooks

On the Java side:

- forward evaluation is delegated through the function object stored on the node
- backward evaluation is delegated through `derivative(value, backward)`

This means each function class mixes two concerns:

- tensor math
- graph-edge construction

## Relevant C-Side Abstractions In `ComputationalGraph-C`

### Already Present

- `Function` vtable in [Function.h](/home/cengiz/dev/ozu/ComputationalGraph-C/src/Function/Function.h)
  - `calculate(const void*, const Tensor*)`
  - `derivative(const void*, const Tensor*, const Tensor*)`
- `Computational_node` in [ComputationalNode.h](/home/cengiz/dev/ozu/ComputationalGraph-C/src/Node/ComputationalNode.h)
  - stores `void* function`
  - stores `value` and `backward`
- graph edge registration in [ComputationalGraph.h](/home/cengiz/dev/ozu/ComputationalGraph-C/src/ComputationalGraph.h)
  - `add_edge(...)`
  - `add_addition_edge(...)`
  - `add_edge_with_hadamard(...)`
- derivative dispatch in [ComputationalGraph.c](/home/cengiz/dev/ozu/ComputationalGraph-C/src/ComputationalGraph.c)
  - uses `function->derivative(...)`

### Missing Relative To Java

- no dedicated `FunctionNode` type
- no function-level `addEdge(inputNodes, isBiased)` equivalent
- no node-local `children` / `parents` arrays like Java
- graph topology is owned by `Computational_graph` hash maps, not by nodes

## Consequence

The tensor-math half of the Java functions can be ported locally in `SequenceProcessing-C`.

The Java `addEdge(...)` pattern cannot be ported 1:1 without introducing a local compatibility layer, because in C edge creation needs a `Computational_graph_ptr`, not just an input-node list.

This is an architectural mismatch, but it is not necessarily a blocker in `ComputationalGraph-C` itself. The existing C graph API is already strong enough to attach function nodes; it just uses a different entry point.

## Recommended Minimal Function-Layer Design

### Design Goal

Port the function math locally in `SequenceProcessing-C` while avoiding any fake `FunctionNode` clone.

### Minimal Viable C Design

1. Each SequenceProcessing function becomes a local C struct whose first field is `Function`.
2. Stateless functions still get their own concrete struct for consistency.
3. Stateful functions keep their Java state:
   - `AdditionByConstant.constant`
   - `MultiplyByConstant.constant`
   - `SquareRoot.epsilon`
   - `Switch.turn`
4. Each function exposes:
   - constructor
   - destructor
   - `calculate_*`
   - `derivative_*`
5. Graph attachment is handled by a local helper, not by the function structs themselves.

### Suggested Local Graph Helper

Add a small local compatibility helper later in `SequenceProcessing-C`, for example:

- `src/Functions/SequenceFunctionEdge.h`
- `src/Functions/SequenceFunctionEdge.c`

with a shape like:

- `Computational_node_ptr add_sequence_function_edge(Computational_graph_ptr graph, Computational_node_ptr input, Function* function, bool is_biased);`

Implementation would delegate to `ComputationalGraph-C:add_edge(...)`.

This keeps:

- function math local to `SequenceProcessing-C`
- graph ownership local to `ComputationalGraph-C`
- model code explicit about which graph it is mutating

## Exact Mapping Per Java Class

### `AdditionByConstant`

- proposed C files:
  - `src/Functions/AdditionByConstant.h`
  - `src/Functions/AdditionByConstant.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - later `ComputationalGraph.h` only through local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes, fully for tensor math
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)` parity
- notes:
  - easy stateful unary transform
  - derivative just returns backward unchanged

### `Inverse`

- proposed C files:
  - `src/Functions/Inverse.h`
  - `src/Functions/Inverse.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - later local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)`
- notes:
  - derivative in Java appears mathematically suspicious
  - Java source currently uses `-pow(x, 2)` rather than `-1 / x^2`
  - if ported, this should follow Java exactly unless the source of truth is corrected first

### `Mask`

- proposed C files:
  - `src/Functions/Mask.h`
  - `src/Functions/Mask.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - `Tensor.h`
  - later local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)`
- notes:
  - simple upper-triangular masking with `-inf`
  - likely useful early for transformer work

### `Mean`

- proposed C files:
  - `src/Functions/Mean.h`
  - `src/Functions/Mean.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - later local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)`
- notes:
  - row-wise reduction then broadcast back to original shape
  - straightforward derivative mask of `1 / width`

### `MultiplyByConstant`

- proposed C files:
  - `src/Functions/MultiplyByConstant.h`
  - `src/Functions/MultiplyByConstant.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - later local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)`
- notes:
  - easy stateful unary transform

### `RemoveBias`

- proposed C files:
  - `src/Functions/RemoveBias.h`
  - `src/Functions/RemoveBias.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - later local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)`
- notes:
  - shape-sensitive but still simple
  - trims the last element and re-expands gradient with a zero appended

### `SquareRoot`

- proposed C files:
  - `src/Functions/SquareRoot.h`
  - `src/Functions/SquareRoot.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - later local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)`
- notes:
  - stateful through `epsilon`
  - derivative in Java uses `1 / (2 * val)` where `val` is read from input tensor, not from `sqrt(epsilon + x)` output
  - if ported now, it should mirror Java as written

### `Switch`

- proposed C files:
  - `src/Functions/Switch.h`
  - `src/Functions/Switch.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - later local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)`
- notes:
  - mutable runtime state through `turn`
  - returns input unchanged or a same-shaped zero tensor
  - derivative mirrors the same switch behavior

### `Transpose`

- proposed C files:
  - `src/Functions/Transpose.h`
  - `src/Functions/Transpose.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - `Math-C: transpose_tensor(...)`
  - later local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)`
- notes:
  - easiest candidate because `Math-C` already has `transpose_tensor`

### `Variance`

- proposed C files:
  - `src/Functions/Variance.h`
  - `src/Functions/Variance.c`
- likely `ComputationalGraph-C` dependency:
  - `Function.h`
  - later local edge helper
- can be implemented locally in `SequenceProcessing-C`:
  - yes
- requires missing base abstraction in `ComputationalGraph-C`:
  - no for math
  - yes for Java-style `addEdge(...)`
- notes:
  - derivative in Java appears mathematically suspicious
  - highest review risk among current function classes

## What Is Actually Blocked?

### Not Blocked

The following are not blocked for pure tensor-math porting:

- `AdditionByConstant`
- `Mask`
- `Mean`
- `MultiplyByConstant`
- `RemoveBias`
- `Switch`
- `Transpose`
- `Inverse`
- `SquareRoot`
- `Variance`

Reason:

- `ComputationalGraph-C` already has a function-vtable model
- `Math-C` already has enough tensor operations for unary transforms, hadamard products, transpose, and shape-preserving tensor allocation

### Blocked

What is blocked is full Java-class parity for the graph-construction method:

- `addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased)`

Reason:

- Java function objects mutate node connectivity directly
- C graph construction currently requires `Computational_graph_ptr` and uses graph-owned edge maps
- there is no direct `FunctionNode` or node-local adjacency abstraction in `ComputationalGraph-C`

## Minimal Missing Abstraction

The missing abstraction is not “function math”.
The missing abstraction is “function-local graph attachment without an explicit graph handle”.

Minimal local fix in `SequenceProcessing-C`:

- add one small helper to bridge from
  - `(graph, input node, function, is_biased)`
  - to `ComputationalGraph-C:add_edge(...)`

This is enough to unblock the function layer without forking `ComputationalGraph-C`.

## Port Order

### Easiest First

1. `Transpose`
2. `AdditionByConstant`
3. `MultiplyByConstant`
4. `RemoveBias`
5. `Mask`
6. `Mean`
7. `Switch`

### Medium Risk

8. `Inverse`
9. `SquareRoot`

### Highest Risk Last

10. `Variance`

Risk rationale:

- `Transpose` is mostly a direct wrapper over existing tensor support
- constant transforms and `RemoveBias` are shape-simple
- `Mask` and `Mean` need more explicit shape loops but are still straightforward
- `Switch` adds mutable state
- `Inverse`, `SquareRoot`, and especially `Variance` have derivative formulas that should be reviewed carefully against the Java source before freezing C behavior

## Recommendation

### Recommended Minimal Function-Layer Design

- implement local SequenceProcessing function structs that embed `Function`
- do not try to recreate Java `FunctionNode`
- add a small local graph-edge helper in `SequenceProcessing-C` when model ports start needing these functions in graphs

### Which Functions Can Be Ported Now

As pure function math:

- `Transpose`
- `AdditionByConstant`
- `MultiplyByConstant`
- `RemoveBias`
- `Mask`
- `Mean`
- `Switch`

Possible now but with derivative-review caution:

- `Inverse`
- `SquareRoot`
- `Variance`

### Which Functions Are Blocked

Blocked only for full Java `addEdge(...)` parity:

- all ten functions

because the blocking issue is shared graph-attachment architecture, not per-function tensor math.

### Recommended Next Step

`B. patch architecture in SequenceProcessing-C`

Reason:

- `ComputationalGraph-C` already has enough low-level primitives for function execution and graph insertion
- the missing piece is a local compatibility layer for graph edge creation
- moving abstractions into `ComputationalGraph-C` can wait until there is a stronger need for Java-style API parity across multiple repos
