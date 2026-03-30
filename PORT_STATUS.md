# SequenceProcessing-C Port Status

## Ported Java Classes

Sequence layer:

- `Sequence/LabelledSentence.java`
- `Sequence/LabelledVectorizedWord.java`
- `Sequence/SequenceCorpus.java`

Parameter layer:

- `Parameters/RecurrentNeuralNetworkParameter.java`
- `Parameters/TransformerParameter.java`

Functions layer:

- `Functions/AdditionByConstant.java`
- `Functions/Inverse.java`
- `Functions/Mask.java`
- `Functions/Mean.java`
- `Functions/MultiplyByConstant.java`
- `Functions/RemoveBias.java`
- `Functions/SquareRoot.java`
- `Functions/Switch.java`
- `Functions/Transpose.java`
- `Functions/Variance.java`

Model layer:

- `Classification/RecurrentNeuralNetworkModel.java`
- `Classification/GatedRecurrentUnitModel.java`
- `Classification/LongShortTermMemoryModel.java`
- `Classification/Transformer.java`

## Staged Compromises

- Recurrent and Transformer training use the grounded class-index backprop path
  instead of claiming multi-input loss-node forward-value parity.
- `Transformer` is staged: shell, preprocessing helpers, graph-input shell,
  staged graph construction, and staged `train(...)` exist, but full Java
  `test(...)` parity is not claimed.
- `SequenceProcessing-C` provides a local compatibility declaration for
  `ClassificationPerformance` only to satisfy `ComputationalGraph-C` header
  dependencies. It does not implement `Classification-C`.

## Upstream Blockers

### ComputationalGraph-C

- `ComputationalGraph-C/src/ComputationalGraph.c:forward_calculation_with_dropout`
  contains a concatenation-path bug in the `CONCATENATED_NODE` assembly loop:
  it uses `new_list2[i]` inside a `for (int j = 1; ...)` loop.
- That corrupts forward execution for model graphs that depend on
  `concat_edges(...)`, which currently blocks real GRU/LSTM runtime and likely
  affects staged Transformer graph execution as well.

### Dictionary-C

- `TransformerTest` currently hits invalid cleanup through
  `free_vectorized_dictionary(...)` in the local test path.
- This is outside `SequenceProcessing-C` ownership and remains a sibling-library
  boundary issue.

### Classification-C Boundary

- Full Java `test(...)` parity for recurrent models and Transformer still needs
  real `ClassificationPerformance` integration from the sibling stack.

## Test Reality

Real local compile-and-run tests verified during this stabilization pass:

- `SequenceCorpusTest`
- `ParameterSliceTest`
- `FunctionSliceTest`
- `RecurrentNeuralNetworkModelTest`

Compile-only tests at the current checkpoint:

- `GatedRecurrentUnitModelTest`
- `LongShortTermMemoryModelTest`
- `TransformerTest`

## Build Reality

- Real local compilation and linking are possible by compiling
  `SequenceProcessing-C` together with the local sibling source trees.
- The checked-in CMake now targets that local-sibling build strategy.
- This environment does not have `cmake` installed, so CMake configure/test was
  not executed directly here.
