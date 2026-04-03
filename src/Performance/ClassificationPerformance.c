#include "ClassificationPerformance.h"

#include "Memory/Memory.h"

Classification_performance_ptr create_sequence_processing_classification_performance(double accuracy) {
    Classification_performance_ptr result = malloc_(sizeof(Classification_performance));
    if (result == NULL) {
        return NULL;
    }
    result->accuracy = accuracy;
    return result;
}

void free_sequence_processing_classification_performance(Classification_performance_ptr performance) {
    if (performance == NULL) {
        return;
    }
    free_(performance);
}

double sequence_processing_classification_performance_get_accuracy(const Classification_performance* performance) {
    if (performance == NULL) {
        return 0.0;
    }
    return performance->accuracy;
}
