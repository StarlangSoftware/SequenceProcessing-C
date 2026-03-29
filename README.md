# SequenceProcessing-C

Scaffold only.

This repository currently contains the initial project layout for a future C port of `SequenceProcessing`.
No translation of Java logic has been implemented yet.

## Local Dependency Wiring

The current `SequenceProcessing-C` build is wired to local sibling repo headers for the first sequence slice only:

- `../Corpus-C/src`
- `../Dictionary-C/src`
- `../Dictionary-C/src/Dictionary`
- `../Math-C/src`
- `../Util-C/src`
- `../DataStructure-C/src`

The current sequence implementation uses real sibling headers in `.c` files while keeping public `SequenceProcessing-C` headers on forward declarations.

## Current Sequence Slice Dependency Chain

The currently ported sequence files compile at header level through this chain:

- `LabelledSentence.c`
  - `Sentence.h` from `Corpus-C`
  - `Sentence.h` depends on `Dictionary/Word.h`
  - `Word.h` depends on `StringUtils.h` from `Util-C`
  - both `Sentence.h` and `StringUtils.h` depend on `ArrayList.h` from `DataStructure-C`

- `LabelledVectorizedWord.c`
  - `VectorizedWord.h` from `Dictionary-C`
  - `VectorizedWord.h` depends on `Vector.h` from `Math-C`
  - `VectorizedWord.h` also depends on `Word.h` from `Dictionary-C`
  - `Word.h` depends on `StringUtils.h` from `Util-C`
  - `Vector.h` and `StringUtils.h` depend on `ArrayList.h` from `DataStructure-C`

## Current Limitation

For the current sequence slice, adding `DataStructure-C` resolves the immediate missing-header problem and the two `.c` files now compile cleanly in syntax-only checks with the configured include roots.

The next likely external dependency for subsequent ports is `Regular-C`, because `Corpus-C` declares `find_package(regular_c REQUIRED)` at the project level. That package is not required by the currently ported sequence files, but it may become relevant once `SequenceProcessing-C` starts building against broader `Corpus-C` functionality such as `SequenceCorpus`.
