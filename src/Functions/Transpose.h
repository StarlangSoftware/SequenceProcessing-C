#ifndef SEQUENCE_PROCESSING_TRANSPOSE_H
#define SEQUENCE_PROCESSING_TRANSPOSE_H

#include "Function/Function.h"

typedef struct transpose_function Transpose_function;
typedef Transpose_function* Transpose_function_ptr;

struct transpose_function {
    Function function;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_transpose_function() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Transpose_function_ptr create_transpose_function(void);

void free_transpose_function(Transpose_function_ptr transpose_function);

/*
 * Returns a newly allocated tensor owned by the caller.
 */
Tensor_ptr calculate_transpose_function(const void* function, const Tensor* tensor);

/*
 * Returns a newly allocated tensor owned by the caller.
 */
Tensor_ptr derivative_transpose_function(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
