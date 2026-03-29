# SequenceProcessing -> SequenceProcessing-C Porting Plan

## Scope

This document is a planning artifact only.
No Java logic is translated in this step.
No model behavior, data flow, training logic, or numerical operations are implemented in this step.

## Source Repositories

- Java source of truth: `../SequenceProcessing`
- C style/reference implementation: `../ComputationalGraph-C`
- Target repository: `./`

## Planning Assumptions

- Each Java class maps to a dedicated C header and source file unless later consolidation is justified.
- The initial C layout mirrors the Java package split:
  - `src/Classification/`
  - `src/Functions/`
  - `src/Parameters/`
  - `src/Sequence/`
- Model classes will likely need explicit C structs plus function-pointer based dispatch where Java currently relies on inheritance and overriding.
- Dependencies listed below are based on current Java imports and the existing C style in `ComputationalGraph-C`.

## Per-File Port Map

### Classification Package

#### `Classification/GatedRecurrentUnitModel.java`

- package: `SequenceProcessing.Classification`
- proposed C target files:
  - `src/Classification/GatedRecurrentUnitModel.h`
  - `src/Classification/GatedRecurrentUnitModel.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: computational graph core, nodes, tensor operations, neural network parameter support, activation functions
  - `Classification-C`: likely shared model/performance abstractions if that repo exposes them
  - `WordToVec-C`: none directly visible from imports
- likely additional external repo dependencies:
  - `Math-C` for tensor math if not fully covered through `ComputationalGraph-C`
- complexity: high
- port priority: later
- notes:
  - depends on `AdditionByConstant`, `RemoveBias`, `Switch`
  - uses multiple graph node types and recurrent model state handling

#### `Classification/LongShortTermMemoryModel.java`

- package: `SequenceProcessing.Classification`
- proposed C target files:
  - `src/Classification/LongShortTermMemoryModel.h`
  - `src/Classification/LongShortTermMemoryModel.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: computational graph core, graph functions, node types, parameter support
  - `Classification-C`: likely shared model/performance abstractions if needed
  - `WordToVec-C`: none directly visible from imports
- likely additional external repo dependencies:
  - `Math-C`
- complexity: high
- port priority: later
- notes:
  - broad wildcard imports suggest heavy graph composition
  - likely one of the more stateful model ports

#### `Classification/RecurrentNeuralNetworkModel.java`

- package: `SequenceProcessing.Classification`
- proposed C target files:
  - `src/Classification/RecurrentNeuralNetworkModel.h`
  - `src/Classification/RecurrentNeuralNetworkModel.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: graph core, softmax, computational nodes, multiplication/concatenation nodes, neural network parameter support
  - `Classification-C`: `ClassificationPerformance` equivalent appears likely
  - `WordToVec-C`: none directly visible from imports
- likely additional external repo dependencies:
  - `Math-C`
- complexity: high
- port priority: later
- notes:
  - likely base recurrent model abstraction for GRU and LSTM-adjacent behavior
  - dependency on classification performance reporting makes API design relevant

#### `Classification/Transformer.java`

- package: `SequenceProcessing.Classification`
- proposed C target files:
  - `src/Classification/Transformer.h`
  - `src/Classification/Transformer.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: graph core, node types, activation and loss functions, initialization, optimizer-facing integration
  - `Classification-C`: `ClassificationPerformance` equivalent appears likely
  - `WordToVec-C`: likely indirect for vectorized vocabulary/data flow, but Java imports point more directly to dictionary/vector layers
- likely additional external repo dependencies:
  - `Dictionary-C`
  - `Math-C`
- complexity: high
- port priority: blocked
- notes:
  - highest architectural risk among current classes
  - depends on `TransformerParameter` and several custom function operators
  - likely needs attention/masking conventions defined before implementation

### Functions Package

#### `Functions/AdditionByConstant.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/AdditionByConstant.h`
  - `src/Functions/AdditionByConstant.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none directly visible
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - straightforward operator logic, but blocked on missing `FunctionNode` design in the current C graph reference

#### `Functions/Inverse.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/Inverse.h`
  - `src/Functions/Inverse.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - likely simple tensor transform once graph-function interface is defined

#### `Functions/Mask.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/Mask.h`
  - `src/Functions/Mask.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - likely important for transformer attention masking

#### `Functions/Mean.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/Mean.h`
  - `src/Functions/Mean.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - likely shares reduction conventions with `Variance`

#### `Functions/MultiplyByConstant.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/MultiplyByConstant.h`
  - `src/Functions/MultiplyByConstant.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - likely low algorithmic risk once function plumbing exists

#### `Functions/RemoveBias.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/RemoveBias.h`
  - `src/Functions/RemoveBias.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - referenced by multiple recurrent model classes

#### `Functions/SquareRoot.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/SquareRoot.h`
  - `src/Functions/SquareRoot.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - likely simple transform after tensor operation conventions are defined

#### `Functions/Switch.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/Switch.h`
  - `src/Functions/Switch.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - referenced by recurrent model classes
  - behavior may affect control-flow-like graph semantics

#### `Functions/Transpose.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/Transpose.h`
  - `src/Functions/Transpose.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - likely low algorithmic complexity, but still blocked on function integration design

#### `Functions/Variance.java`

- package: `SequenceProcessing.Functions`
- proposed C target files:
  - `src/Functions/Variance.h`
  - `src/Functions/Variance.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: function base type, `FunctionNode`-like abstraction, computational node integration, tensor math
  - `Classification-C`: none
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - `Math-C`
- complexity: medium
- port priority: blocked
- notes:
  - probably tied to normalization behavior together with `Mean` and `SquareRoot`

### Parameters Package

#### `Parameters/RecurrentNeuralNetworkParameter.java`

- package: `SequenceProcessing.Parameters`
- proposed C target files:
  - `src/Parameters/RecurrentNeuralNetworkParameter.h`
  - `src/Parameters/RecurrentNeuralNetworkParameter.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: neural network parameter base struct, initialization abstractions, function references
  - `Classification-C`: none directly visible
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - none expected beyond graph dependencies
- complexity: medium
- port priority: now
- notes:
  - appears primarily parameter-container oriented
  - good early candidate once base parameter embedding strategy is chosen

#### `Parameters/TransformerParameter.java`

- package: `SequenceProcessing.Parameters`
- proposed C target files:
  - `src/Parameters/TransformerParameter.h`
  - `src/Parameters/TransformerParameter.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: neural network parameter base struct, initialization abstractions, function references
  - `Classification-C`: none directly visible
  - `WordToVec-C`: none
- likely additional external repo dependencies:
  - none expected beyond graph dependencies
- complexity: medium
- port priority: now
- notes:
  - likely a data-configuration type and a reasonable early implementation target

### Sequence Package

#### `Sequence/LabelledSentence.java`

- package: `SequenceProcessing.Sequence`
- proposed C target files:
  - `src/Sequence/LabelledSentence.h`
  - `src/Sequence/LabelledSentence.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: none
  - `Classification-C`: none
  - `WordToVec-C`: none directly visible
- likely additional external repo dependencies:
  - `Corpus-C`
- complexity: low
- port priority: now
- notes:
  - likely simple wrapper or extension over sentence data with labels
  - good first structural port candidate

#### `Sequence/LabelledVectorizedWord.java`

- package: `SequenceProcessing.Sequence`
- proposed C target files:
  - `src/Sequence/LabelledVectorizedWord.h`
  - `src/Sequence/LabelledVectorizedWord.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: none
  - `Classification-C`: none
  - `WordToVec-C`: likely conceptual dependency, but direct Java imports point to dictionary/vector types rather than a `WordToVec` package
- likely additional external repo dependencies:
  - `Dictionary-C`
  - `Math-C`
- complexity: medium
- port priority: now
- notes:
  - data-structure oriented
  - likely needed before corpus loading and model input preparation

#### `Sequence/SequenceCorpus.java`

- package: `SequenceProcessing.Sequence`
- proposed C target files:
  - `src/Sequence/SequenceCorpus.h`
  - `src/Sequence/SequenceCorpus.c`
- likely direct dependencies from sibling repos:
  - `ComputationalGraph-C`: none
  - `Classification-C`: none
  - `WordToVec-C`: likely indirect only
- likely additional external repo dependencies:
  - `Corpus-C`
  - `Dictionary-C`
  - `Math-C`
  - `Util-C`
- complexity: medium
- port priority: later
- notes:
  - loader/parsing logic appears implementable before model code
  - blocked only if supporting corpus/file/vector APIs are not yet available in C

## Dependency View

### Likely Depend On `ComputationalGraph-C`

- all files under `src/Classification/`
- all files under `src/Functions/`
- `src/Parameters/RecurrentNeuralNetworkParameter.*`
- `src/Parameters/TransformerParameter.*`

Rationale:
- Java imports point to graph nodes, graph functions, neural network parameter classes, initialization classes, and tensor-oriented operations.
- The current C reference repo already defines graph core, nodes, optimizers, initialization, and activation functions, so it is the primary architectural style guide.

### Likely Depend On `Classification-C`

- `src/Classification/RecurrentNeuralNetworkModel.*`
- `src/Classification/Transformer.*`
- possibly `src/Classification/GatedRecurrentUnitModel.*`
- possibly `src/Classification/LongShortTermMemoryModel.*`

Rationale:
- Java imports explicitly reference `Classification.Performance.ClassificationPerformance` in some model classes.
- If classification metrics, training loops, or shared model interfaces live in `Classification-C`, these model ports should align with them rather than duplicating that layer locally.

### Likely Depend On `WordToVec-C`

- no source file shows a direct Java import from a `WordToVec` package
- most vectorized-token dependencies appear to come through `Dictionary.VectorizedWord` and `Math.Vector`
- practical integration candidates:
  - `src/Sequence/LabelledVectorizedWord.*`
  - `src/Sequence/SequenceCorpus.*`
  - `src/Classification/Transformer.*` indirectly through vectorized vocabulary/data preparation

Rationale:
- the Maven project depends on `WordToVec`, but the currently visible Java source uses dictionary/vector abstractions instead of directly importing it.
- this suggests either an indirect dependency chain or shared types exposed through sibling libraries.

### Appear Self-Contained Within This Repo

- none are fully self-contained in the strict sense
- lowest external coupling:
  - `src/Sequence/LabelledSentence.*`
  - `src/Parameters/RecurrentNeuralNetworkParameter.*`
  - `src/Parameters/TransformerParameter.*`

Rationale:
- even the simplest files still appear to rely on shared types from sibling libraries such as corpus, graph parameter, initialization, or vector types.

## Test Mapping

### `SequenceCorpusTest.java`

- proposed C test files:
  - `test/SequenceCorpusTest.c`
  - optionally `test/SequenceCorpusTest.h`
- production files under test:
  - `src/Sequence/SequenceCorpus.h`
  - `src/Sequence/SequenceCorpus.c`
- can be tested early:
  - yes, after the sequence data structures and corpus/file-loading dependencies are available
- early test focus:
  - corpus size and sentence count
  - label extraction
  - vectorized token loading/parsing
  - basic error handling for malformed resources

### `TransformerTest.java`

- proposed C test files:
  - `test/TransformerTest.c`
  - optionally `test/TransformerTest.h`
- production files under test:
  - `src/Classification/Transformer.h`
  - `src/Classification/Transformer.c`
  - `src/Parameters/TransformerParameter.h`
  - `src/Parameters/TransformerParameter.c`
  - dependent custom function files under `src/Functions/`
- can be tested early:
  - partially
- early test focus:
  - parameter construction and teardown
  - tensor shape expectations for helper utilities
  - deterministic setup wiring if initialization stubs or mocks exist
- deferred test focus:
  - end-to-end model training/inference
  - optimizer integration
  - classification performance assertions

## Explicit Porting Phases

### Phase 1: Files To Port Now

- `src/Sequence/LabelledSentence.h`
- `src/Sequence/LabelledSentence.c`
- `src/Sequence/LabelledVectorizedWord.h`
- `src/Sequence/LabelledVectorizedWord.c`
- `src/Parameters/RecurrentNeuralNetworkParameter.h`
- `src/Parameters/RecurrentNeuralNetworkParameter.c`
- `src/Parameters/TransformerParameter.h`
- `src/Parameters/TransformerParameter.c`

Reasoning:
- these files appear to be mostly data-structure and configuration oriented
- they establish shared types needed by later ports
- they have materially lower algorithmic risk than model classes

### Phase 2: Files To Port After Phase 1

- `src/Sequence/SequenceCorpus.h`
- `src/Sequence/SequenceCorpus.c`
- `test/SequenceCorpusTest.c`

Reasoning:
- corpus parsing is useful early, but it depends on Phase 1 sequence types and on confirming the C-side corpus/vector/file utility APIs
- this phase gives an early testable slice before graph-heavy model work begins

### Phase 3: Files Blocked On Deeper Design

- `src/Functions/AdditionByConstant.h`
- `src/Functions/AdditionByConstant.c`
- `src/Functions/Inverse.h`
- `src/Functions/Inverse.c`
- `src/Functions/Mask.h`
- `src/Functions/Mask.c`
- `src/Functions/Mean.h`
- `src/Functions/Mean.c`
- `src/Functions/MultiplyByConstant.h`
- `src/Functions/MultiplyByConstant.c`
- `src/Functions/RemoveBias.h`
- `src/Functions/RemoveBias.c`
- `src/Functions/SquareRoot.h`
- `src/Functions/SquareRoot.c`
- `src/Functions/Switch.h`
- `src/Functions/Switch.c`
- `src/Functions/Transpose.h`
- `src/Functions/Transpose.c`
- `src/Functions/Variance.h`
- `src/Functions/Variance.c`

Reasoning:
- the Java code assumes a graph-function layer with `FunctionNode`
- `ComputationalGraph-C` does not currently expose an obvious `FunctionNode` counterpart in the observed files
- these operators should wait until the C representation for custom graph functions is settled

### Phase 4: Model-Heavy Classes

- `src/Classification/RecurrentNeuralNetworkModel.h`
- `src/Classification/RecurrentNeuralNetworkModel.c`
- `src/Classification/GatedRecurrentUnitModel.h`
- `src/Classification/GatedRecurrentUnitModel.c`
- `src/Classification/LongShortTermMemoryModel.h`
- `src/Classification/LongShortTermMemoryModel.c`
- `src/Classification/Transformer.h`
- `src/Classification/Transformer.c`
- `test/TransformerTest.c`

Reasoning:
- these files sit on top of nearly every earlier phase
- they require settled parameter structs, function/operator plumbing, graph integration, and likely classification metric interfaces
- `Transformer` is the most design-sensitive class and should be last among the current set

## Current Blockers And Unknowns

- `ComputationalGraph-C` currently shows node and function support, but the Java-side `FunctionNode` abstraction is not obviously present in the observed C reference files.
- The exact C equivalents for Java dependencies from `Corpus`, `Dictionary`, `Util`, `Math`, and possibly `Classification` are not yet confirmed locally.
- The Maven dependency on `WordToVec` appears indirect from the current imports, so the exact integration surface still needs confirmation before implementation.
- Model classes will require an explicit ownership and lifecycle model for tensors, nodes, parameters, and temporary graph state.
- The final public API shape for `SequenceProcessing-C` should be decided before model implementation to avoid churn across all headers.
