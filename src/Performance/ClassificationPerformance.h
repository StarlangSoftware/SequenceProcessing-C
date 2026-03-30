#ifndef SEQUENCE_PROCESSING_C_CLASSIFICATION_PERFORMANCE_COMPAT_H
#define SEQUENCE_PROCESSING_C_CLASSIFICATION_PERFORMANCE_COMPAT_H

/*
 * Minimal local compatibility declaration for ComputationalGraph-C's public
 * header dependency. SequenceProcessing-C does not implement or consume
 * ClassificationPerformance behavior locally; the staged ports only need the
 * opaque pointer type so ComputationalGraph-C headers compile.
 */

typedef struct classification_performance Classification_performance;
typedef Classification_performance* Classification_performance_ptr;

#endif
