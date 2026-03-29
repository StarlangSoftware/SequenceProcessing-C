#include "Mean.h"

#include "Memory/Memory.h"

#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

Mean_function_ptr create_mean_function(void) {
    Mean_function_ptr result = malloc_(sizeof(Mean_function));
    if (result == NULL) {
        return NULL;
    }
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_mean_function;
    result->function.derivative = derivative_mean_function;
    return result;
}

void free_mean_function(Mean_function_ptr mean_function) {
    if (mean_function == NULL) {
        return;
    }
    free_(mean_function);
}

Tensor_ptr calculate_mean_function(const void* function, const Tensor* tensor) {
    double* values;
    int i, j;
    int rows, cols;
    double total;
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
        total = 0.0;
        for (j = 0; j < cols; j++) {
            total += tensor->data[i * cols + j];
        }
        total /= cols;
        for (j = 0; j < cols; j++) {
            values[i * cols + j] = total;
        }
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

Tensor_ptr derivative_mean_function(const void* function, const Tensor* tensor, const Tensor* backward) {
    double* values;
    Tensor_ptr tmp;
    Tensor_ptr result;
    int i;
    double coefficient;
    (void) function;
    if (tensor == NULL || backward == NULL || tensor->dimensions != 2 || backward->dimensions != 2) {
        return NULL;
    }
    if (tensor->total_elements != backward->total_elements ||
        tensor->shape[0] != backward->shape[0] ||
        tensor->shape[1] != backward->shape[1]) {
        return NULL;
    }
    coefficient = 1.0 / tensor->shape[1];
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements; i++) {
        values[i] = coefficient;
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
