#ifndef SEQUENCE_PROCESSING_MEAN_H
#define SEQUENCE_PROCESSING_MEAN_H

#include "Function/Function.h"

typedef struct mean_function Mean_function;
typedef Mean_function* Mean_function_ptr;

struct mean_function {
    Function function;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_mean_function() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Mean_function_ptr create_mean_function(void);

void free_mean_function(Mean_function_ptr mean_function);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Grounded to the Java implementation, this function currently supports only
 * 2D tensors and repeats each row mean across that row.
 */
Tensor_ptr calculate_mean_function(const void* function, const Tensor* tensor);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Grounded to the Java implementation, this function currently supports only
 * 2D tensors and multiplies `backward` by a row-wise `1 / columns` mask.
 */
Tensor_ptr derivative_mean_function(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
