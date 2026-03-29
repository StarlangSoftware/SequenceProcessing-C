#ifndef SEQUENCE_PROCESSING_INVERSE_H
#define SEQUENCE_PROCESSING_INVERSE_H

#include "Function/Function.h"

typedef struct inverse_function Inverse_function;
typedef Inverse_function* Inverse_function_ptr;

struct inverse_function {
    Function function;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_inverse_function() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Inverse_function_ptr create_inverse_function(void);

void free_inverse_function(Inverse_function_ptr inverse_function);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Grounded to the Java implementation, this currently supports only 2D
 * tensors and applies elementwise `1.0 / x`.
 */
Tensor_ptr calculate_inverse_function(const void* function, const Tensor* tensor);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Grounded to the Java implementation, this currently supports only 2D
 * tensors and multiplies `backward` by `-(x^2)` elementwise.
 */
Tensor_ptr derivative_inverse_function(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
