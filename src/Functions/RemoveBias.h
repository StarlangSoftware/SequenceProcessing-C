#ifndef SEQUENCE_PROCESSING_REMOVE_BIAS_H
#define SEQUENCE_PROCESSING_REMOVE_BIAS_H

#include "Function/Function.h"

typedef struct remove_bias Remove_bias;
typedef Remove_bias* Remove_bias_ptr;

struct remove_bias {
    Function function;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_remove_bias() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Remove_bias_ptr create_remove_bias(void);

void free_remove_bias(Remove_bias_ptr remove_bias);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Java flattens the input data and returns a row tensor of shape
 * `1 x (total_elements - 1)`.
 */
Tensor_ptr calculate_remove_bias(const void* function, const Tensor* tensor);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Java appends a trailing `0.0` to the flattened backward tensor and returns
 * a row tensor of shape `1 x (total_elements + 1)`.
 */
Tensor_ptr derivative_remove_bias(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
