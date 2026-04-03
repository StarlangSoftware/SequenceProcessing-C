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
- `Transformer` is still staged overall: shell, preprocessing helpers,
  graph-input shell, staged graph construction, staged `train(...)`, and the
  current local `test(...)` path exist, but that is still not a claim of full
  Java model parity.
- The current Transformer token-lookup slice uses the local
  `TransformerTokenStore` helper rather than `Dictionary-C`, including the
  current local token-vector retrieval needed by `Transformer.test(...)`.
- `SequenceProcessing-C` still does not claim full `Classification-C`
  integration or full Java model/test parity.

## Current Dependency Decisions

- Current Transformer constructor/token-scan/token-vector behavior does not
  depend on `Dictionary-C`. It is served by the local `TransformerTokenStore`
  helper.
- `ComputationalGraph-C` now has an explicit local
  `src/Performance/ClassificationPerformance.*` test dependency so its
  standalone test runner links under the current local compile pattern.
- Local recurrent and Transformer `test(...)` paths now return local
  `ClassificationPerformance` support, but this is still not a claim of full
  sibling-stack integration parity.

## Verification Matrix

Standalone `SequenceProcessing-C` tests verified:

- `SequenceCorpusTest`
- `ParameterSliceTest`
- `FunctionSliceTest`
- `RecurrentNeuralNetworkModelTest`
- `GatedRecurrentUnitModelTest`
- `LongShortTermMemoryModelTest`
- `TransformerTest`

Standalone `ComputationalGraph-C` verification also passes for:

- `test/Test.c`
- `test/BackpropDerivativeOwnership.c`

Current verification modes:

- normal standalone local builds: pass for every target listed above
- ASAN standalone local builds with `ASAN_OPTIONS=detect_leaks=0`: pass for
  every target listed above

## Build Reality

- Real local compilation and linking are possible by compiling
  `SequenceProcessing-C` together with the local sibling source trees.
- The checked-in `SequenceProcessing-C` CMake targets that local-sibling build
  strategy.
- `ComputationalGraph-C` standalone local test compilation is also now explicit
  about its local classification-performance dependency.
- This environment does not have `cmake` installed, so CMake configure/test was
  not executed directly here.

## Intentionally Not Claimed

- full Java parity across all model behavior
- production parity with the full sibling stack
- production readiness
- leak-clean ASAN verification beyond the current `detect_leaks=0` sweep
