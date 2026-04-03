#ifndef SEQUENCE_PROCESSING_C_CLASSIFICATION_PERFORMANCE_COMPAT_H
#define SEQUENCE_PROCESSING_C_CLASSIFICATION_PERFORMANCE_COMPAT_H

/*
 * Local classification-performance support for SequenceProcessing-C.
 *
 * This keeps the same struct/tag names used across the sibling stack so
 * `ComputationalGraph-C` public headers still compile, but the helper API is
 * prefixed locally to avoid symbol collisions with sibling repositories.
 */

typedef struct classification_performance {
    double accuracy;
} Classification_performance;

typedef Classification_performance* Classification_performance_ptr;

Classification_performance_ptr create_sequence_processing_classification_performance(double accuracy);

void free_sequence_processing_classification_performance(Classification_performance_ptr performance);

double sequence_processing_classification_performance_get_accuracy(const Classification_performance* performance);

#endif
