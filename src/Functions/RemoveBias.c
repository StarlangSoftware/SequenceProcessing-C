#include "RemoveBias.h"

#include "Memory/Memory.h"

#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

Remove_bias_ptr create_remove_bias(void) {
    Remove_bias_ptr result = malloc_(sizeof(Remove_bias));
    if (result == NULL) {
        return NULL;
    }
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_remove_bias;
    result->function.derivative = derivative_remove_bias;
    return result;
}

void free_remove_bias(Remove_bias_ptr remove_bias) {
    if (remove_bias == NULL) {
        return;
    }
    free_(remove_bias);
}

Tensor_ptr calculate_remove_bias(const void* function, const Tensor* tensor) {
    double* values;
    int shape[2] = {1, 0};
    int i;
    (void) function;
    if (tensor == NULL || tensor->total_elements < 1) {
        return NULL;
    }
    values = malloc_((tensor->total_elements - 1) * sizeof(double));
    if (values == NULL && tensor->total_elements > 1) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements - 1; i++) {
        values[i] = tensor->data[i];
    }
    shape[1] = tensor->total_elements - 1;
    return create_tensor3(values, shape, 2);
}

Tensor_ptr derivative_remove_bias(const void* function, const Tensor* tensor, const Tensor* backward) {
    double* values;
    int shape[2] = {1, 0};
    int i;
    (void) function;
    (void) tensor;
    if (backward == NULL) {
        return NULL;
    }
    values = malloc_((backward->total_elements + 1) * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < backward->total_elements; i++) {
        values[i] = backward->data[i];
    }
    values[backward->total_elements] = 0.0;
    shape[1] = backward->total_elements + 1;
    return create_tensor3(values, shape, 2);
}
