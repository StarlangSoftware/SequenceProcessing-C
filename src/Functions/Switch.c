#include "Switch.h"

#include "Memory/Memory.h"

#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

static Tensor_ptr create_zero_tensor_like(const Tensor* tensor) {
    double* values;
    int i;
    if (tensor == NULL) {
        return NULL;
    }
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements; i++) {
        values[i] = 0.0;
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

Switch_function_ptr create_switch_function(void) {
    Switch_function_ptr result = malloc_(sizeof(Switch_function));
    if (result == NULL) {
        return NULL;
    }
    result->turn = true;
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_switch_function;
    result->function.derivative = derivative_switch_function;
    return result;
}

void free_switch_function(Switch_function_ptr switch_function) {
    if (switch_function == NULL) {
        return;
    }
    free_(switch_function);
}

void set_switch_turn(Switch_function_ptr switch_function, bool turn) {
    if (switch_function == NULL) {
        return;
    }
    switch_function->turn = turn;
}

Tensor_ptr calculate_switch_function(const void* function, const Tensor* tensor) {
    const Switch_function* switch_function = function;
    if (switch_function == NULL || tensor == NULL) {
        return NULL;
    }
    if (switch_function->turn) {
        return clone_tensor(tensor);
    }
    return create_zero_tensor_like(tensor);
}

Tensor_ptr derivative_switch_function(const void* function, const Tensor* tensor, const Tensor* backward) {
    const Switch_function* switch_function = function;
    if (switch_function == NULL) {
        return NULL;
    }
    if (switch_function->turn) {
        if (backward == NULL) {
            return NULL;
        }
        return clone_tensor(backward);
    }
    return calculate_switch_function(function, tensor);
}
