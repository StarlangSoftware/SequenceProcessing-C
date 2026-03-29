#include "Inverse.h"

#include "Memory/Memory.h"

#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

Inverse_function_ptr create_inverse_function(void) {
    Inverse_function_ptr result = malloc_(sizeof(Inverse_function));
    if (result == NULL) {
        return NULL;
    }
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_inverse_function;
    result->function.derivative = derivative_inverse_function;
    return result;
}

void free_inverse_function(Inverse_function_ptr inverse_function) {
    if (inverse_function == NULL) {
        return;
    }
    free_(inverse_function);
}

Tensor_ptr calculate_inverse_function(const void* function, const Tensor* tensor) {
    double* values;
    int i;
    (void) function;
    if (tensor == NULL || tensor->dimensions != 2) {
        return NULL;
    }
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements; i++) {
        values[i] = 1.0 / tensor->data[i];
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

Tensor_ptr derivative_inverse_function(const void* function, const Tensor* tensor, const Tensor* backward) {
    double* values;
    Tensor_ptr tmp;
    Tensor_ptr result;
    int i;
    (void) function;
    if (tensor == NULL || backward == NULL || tensor->dimensions != 2 || backward->dimensions != 2) {
        return NULL;
    }
    if (tensor->total_elements != backward->total_elements ||
        tensor->shape[0] != backward->shape[0] ||
        tensor->shape[1] != backward->shape[1]) {
        return NULL;
    }
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements; i++) {
        values[i] = -(tensor->data[i] * tensor->data[i]);
    }
    tmp = create_tensor3(values, tensor->shape, tensor->dimensions);
    if (tmp == NULL) {
        free_(values);
        return NULL;
    }
    result = hadamard_product(backward, tmp);
    free_tensor(tmp);
    return result;
}
