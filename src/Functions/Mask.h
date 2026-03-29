#ifndef SEQUENCE_PROCESSING_MASK_H
#define SEQUENCE_PROCESSING_MASK_H

#include "Function/Function.h"

typedef struct mask_function Mask_function;
typedef Mask_function* Mask_function_ptr;

struct mask_function {
    Function function;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_mask_function() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Mask_function_ptr create_mask_function(void);

void free_mask_function(Mask_function_ptr mask_function);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Grounded to the Java implementation, this function currently supports only
 * 2D tensors and sets entries above the main diagonal to negative infinity.
 */
Tensor_ptr calculate_mask_function(const void* function, const Tensor* tensor);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * Java multiplies `backward` by an all-ones tensor of the input shape. With
 * matching shapes this is equivalent to cloning `backward`.
 */
Tensor_ptr derivative_mask_function(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
