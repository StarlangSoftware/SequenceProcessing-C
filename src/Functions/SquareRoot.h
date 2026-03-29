#ifndef SEQUENCE_PROCESSING_SQUARE_ROOT_H
#define SEQUENCE_PROCESSING_SQUARE_ROOT_H

#include "Function/Function.h"

typedef struct square_root_function Square_root_function;
typedef Square_root_function* Square_root_function_ptr;

struct square_root_function {
    Function function;
    double epsilon;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_square_root_function() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Square_root_function_ptr create_square_root_function(double epsilon);

void free_square_root_function(Square_root_function_ptr square_root_function);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Grounded to the Java implementation, this currently supports only 2D
 * tensors and applies elementwise `sqrt(epsilon + x)`.
 */
Tensor_ptr calculate_square_root_function(const void* function, const Tensor* tensor);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Grounded to the Java implementation, this currently supports only 2D
 * tensors and multiplies `backward` by `1 / (2 * x)` elementwise. This uses
 * the raw input tensor values, matching the Java source exactly.
 */
Tensor_ptr derivative_square_root_function(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
