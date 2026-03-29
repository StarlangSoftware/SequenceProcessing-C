#include "Mask.h"

#include "Memory/Memory.h"

#include <math.h>

#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

Mask_function_ptr create_mask_function(void) {
    Mask_function_ptr result = malloc_(sizeof(Mask_function));
    if (result == NULL) {
        return NULL;
    }
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_mask_function;
    result->function.derivative = derivative_mask_function;
    return result;
}

void free_mask_function(Mask_function_ptr mask_function) {
    if (mask_function == NULL) {
        return;
    }
    free_(mask_function);
}

Tensor_ptr calculate_mask_function(const void* function, const Tensor* tensor) {
    double* values;
    int rows, cols;
    int i, j;
    (void) function;
    if (tensor == NULL || tensor->dimensions != 2) {
        return NULL;
    }
    rows = tensor->shape[0];
    cols = tensor->shape[1];
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            values[i * cols + j] = j > i ? -INFINITY : tensor->data[i * cols + j];
        }
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

Tensor_ptr derivative_mask_function(const void* function, const Tensor* tensor, const Tensor* backward) {
    (void) function;
    if (tensor == NULL || backward == NULL) {
        return NULL;
    }
    if (tensor->dimensions != backward->dimensions || tensor->total_elements != backward->total_elements) {
        return NULL;
    }
    return clone_tensor(backward);
}
