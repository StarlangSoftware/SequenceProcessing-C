#ifndef SEQUENCE_PROCESSING_SWITCH_H
#define SEQUENCE_PROCESSING_SWITCH_H

#include "Function/Function.h"

#include <stdbool.h>

typedef struct switch_function Switch_function;
typedef Switch_function* Switch_function_ptr;

struct switch_function {
    Function function;
    bool turn;
};

/*
 * Ownership:
 * - caller owns the returned function object until it is either freed with
 *   free_switch_function() or transferred into a graph node via
 *   add_sequence_function_edge()
 */
Switch_function_ptr create_switch_function(void);

void free_switch_function(Switch_function_ptr switch_function);

void set_switch_turn(Switch_function_ptr switch_function, bool turn);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * When `turn` is true, Java returns the original tensor reference. The C port
 * returns a cloned tensor so the caller receives an owned result.
 */
Tensor_ptr calculate_switch_function(const void* function, const Tensor* tensor);

/*
 * Returns a newly allocated tensor owned by the caller.
 *
 * When `turn` is false, Java delegates to `calculate(value)`, which produces a
 * zero tensor of the same shape as `value`.
 */
Tensor_ptr derivative_switch_function(const void* function, const Tensor* tensor, const Tensor* backward);

#endif
