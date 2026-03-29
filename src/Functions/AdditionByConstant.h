#ifndef SEQUENCE_PROCESSING_ADDITION_BY_CONSTANT_H
#define SEQUENCE_PROCESSING_ADDITION_BY_CONSTANT_H

#include "Function/Function.h"

typedef struct addition_by_constant Addition_by_constant;
typedef Addition_by_constant* Addition_by_constant_ptr;

struct addition_by_constant {
    Function function;
    double constant;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_addition_by_constant() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Addition_by_constant_ptr create_addition_by_constant(double constant);

void free_addition_by_constant(Addition_by_constant_ptr addition_by_constant);

/*
 * Returns a newly allocated tensor owned by the caller.
 */
Tensor_ptr calculate_addition_by_constant(const void* function, const Tensor* tensor);

/*
 * Java returns the backward tensor reference unchanged. The C port returns a
 * cloned tensor so the caller receives an owned result.
 */
Tensor_ptr derivative_addition_by_constant(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
