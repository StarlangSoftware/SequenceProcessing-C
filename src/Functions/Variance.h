#ifndef SEQUENCE_PROCESSING_VARIANCE_H
#define SEQUENCE_PROCESSING_VARIANCE_H

#include "Function/Function.h"

typedef struct variance_function Variance_function;
typedef Variance_function* Variance_function_ptr;

struct variance_function {
    Function function;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_variance_function() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Variance_function_ptr create_variance_function(void);

void free_variance_function(Variance_function_ptr variance_function);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Grounded to the Java implementation, this currently supports only 2D
 * tensors and repeats each row's mean squared value across that row.
 */
Tensor_ptr calculate_variance_function(const void* function, const Tensor* tensor);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Grounded to the Java implementation, this currently supports only 2D
 * tensors and multiplies `backward` by
 * `2 * sqrt(columns * x) / columns` elementwise.
 */
Tensor_ptr derivative_variance_function(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
