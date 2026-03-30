# SequenceProcessing-C

C port of `SequenceProcessing`, staged against local sibling C repos in this
workspace.

## Local Build Strategy

`SequenceProcessing-C` now builds against the local sibling source trees rather
than requiring preinstalled CMake packages:

- `../ComputationalGraph-C`
- `../Corpus-C`
- `../Dictionary-C`
- `../Math-C`
- `../Util-C`
- `../DataStructure-C`
- `../Regular-C`

The CMake build defines private local libraries from those source trees and
links `SequenceProcessing` against them. This is the intended local workspace
configuration path.

## Local Compatibility Notes

- `src/Performance/ClassificationPerformance.h` is a minimal local compatibility
  declaration so `ComputationalGraph-C` public headers compile without
  `Classification-C` in this workspace.
- `SequenceProcessing-C` does not implement `ClassificationPerformance`
  behavior locally.
- Feature macros `_DEFAULT_SOURCE` and `_POSIX_C_SOURCE=200809L` are enabled so
  libc APIs such as `strtok_r`, `mkstemp`, `fdopen`, `random`, and `srandom`
  are declared during real builds.

## Test Status

Currently runnable in local manual builds:

- `SequenceCorpusTest`
- `ParameterSliceTest`
- `FunctionSliceTest`
- `RecurrentNeuralNetworkModelTest`

Currently compile-only in this repo:

- `GatedRecurrentUnitModelTest`
- `LongShortTermMemoryModelTest`
- `TransformerTest`

Those model-layer tests are intentionally not registered with `ctest` yet
because their runtime paths still cross upstream sibling-library issues outside
`SequenceProcessing-C`.

See [PORT_STATUS.md](/home/cengiz/dev/ozu/SequenceProcessing-C/PORT_STATUS.md)
for the current port and blocker summary.
