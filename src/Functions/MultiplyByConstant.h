#ifndef SEQUENCE_PROCESSING_MULTIPLY_BY_CONSTANT_H
#define SEQUENCE_PROCESSING_MULTIPLY_BY_CONSTANT_H

#include "Function/Function.h"

typedef struct multiply_by_constant Multiply_by_constant;
typedef Multiply_by_constant* Multiply_by_constant_ptr;

struct multiply_by_constant {
    Function function;
    double constant;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_multiply_by_constant() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Multiply_by_constant_ptr create_multiply_by_constant(double constant);

void free_multiply_by_constant(Multiply_by_constant_ptr multiply_by_constant);

/*
 * Returns a newly allocated tensor owned by the caller.
 */
Tensor_ptr calculate_multiply_by_constant(const void* function, const Tensor* tensor);

/*
 * Returns a newly allocated tensor owned by the caller.
 */
Tensor_ptr derivative_multiply_by_constant(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
