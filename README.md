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

- `src/Performance/ClassificationPerformance.h` remains a minimal local
  compatibility declaration for `SequenceProcessing-C`'s own local build path.
- Current `Transformer` token/vector lookup does not rely on `Dictionary-C`.
  It uses the local
  [TransformerTokenStore](/home/cengiz/dev/ozu/SequenceProcessing-C/src/Classification/TransformerTokenStore.h)
  helper for the current constructor, token-scan, and `test(...)` surface.
- `Dictionary-C` is still present in the broader workspace build because other
  ported files include it, but it is not required for the current local
  Transformer token/vector slice.
- Feature macros `_DEFAULT_SOURCE` and `_POSIX_C_SOURCE=200809L` are enabled so
  libc APIs such as `strtok_r`, `mkstemp`, `fdopen`, `random`, and `srandom`
  are declared during real builds.

## Test Status

Current standalone local verification:

- `SequenceCorpusTest`
- `ParameterSliceTest`
- `FunctionSliceTest`
- `RecurrentNeuralNetworkModelTest`
- `GatedRecurrentUnitModelTest`
- `LongShortTermMemoryModelTest`
- `TransformerTest`

All of the above currently pass:

- in normal standalone local builds
- in ASAN standalone local builds with `ASAN_OPTIONS=detect_leaks=0`

Qualified limits at this checkpoint:

- this is a verified staged checkpoint, not a claim of full Java parity
- recurrent and Transformer training currently use the grounded class-index
  backprop path
- `Transformer` remains a staged port even though the current local
  `test(...)` path is implemented
- full Java parity across all model behavior is not claimed
- full sibling-stack integration parity is not claimed
- ASAN coverage in this checkpoint used `detect_leaks=0`, so leak-cleanliness is
  not claimed here

See [PORT_STATUS.md](/home/cengiz/dev/ozu/SequenceProcessing-C/PORT_STATUS.md)
for the current port and blocker summary.
