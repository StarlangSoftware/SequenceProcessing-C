#include "Variance.h"

#include "Memory/Memory.h"

#include <math.h>

#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

Variance_function_ptr create_variance_function(void) {
    Variance_function_ptr result = malloc_(sizeof(Variance_function));
    if (result == NULL) {
        return NULL;
    }
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_variance_function;
    result->function.derivative = derivative_variance_function;
    return result;
}

void free_variance_function(Variance_function_ptr variance_function) {
    if (variance_function == NULL) {
        return;
    }
    free_(variance_function);
}

Tensor_ptr calculate_variance_function(const void* function, const Tensor* tensor) {
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
            total += tensor->data[i * cols + j] * tensor->data[i * cols + j];
        }
        total /= cols;
        for (j = 0; j < cols; j++) {
            values[i * cols + j] = total;
        }
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

Tensor_ptr derivative_variance_function(const void* function, const Tensor* tensor, const Tensor* backward) {
    double* values;
    Tensor_ptr tmp;
    Tensor_ptr result;
    int i;
    double cols;
    (void) function;
    if (tensor == NULL || backward == NULL || tensor->dimensions != 2 || backward->dimensions != 2) {
        return NULL;
    }
    if (tensor->total_elements != backward->total_elements ||
        tensor->shape[0] != backward->shape[0] ||
        tensor->shape[1] != backward->shape[1]) {
        return NULL;
    }
    cols = tensor->shape[1];
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements; i++) {
        values[i] = 2.0 * sqrt(cols * tensor->data[i]) / cols;
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
