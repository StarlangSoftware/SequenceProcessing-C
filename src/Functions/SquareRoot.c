#include "SquareRoot.h"

#include "Memory/Memory.h"

#include <math.h>

#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

Square_root_function_ptr create_square_root_function(double epsilon) {
    Square_root_function_ptr result = malloc_(sizeof(Square_root_function));
    if (result == NULL) {
        return NULL;
    }
    result->epsilon = epsilon;
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_square_root_function;
    result->function.derivative = derivative_square_root_function;
    return result;
}

void free_square_root_function(Square_root_function_ptr square_root_function) {
    if (square_root_function == NULL) {
        return;
    }
    free_(square_root_function);
}

Tensor_ptr calculate_square_root_function(const void* function, const Tensor* tensor) {
    const Square_root_function* square_root_function = function;
    double* values;
    int i;
    if (square_root_function == NULL || tensor == NULL || tensor->dimensions != 2) {
        return NULL;
    }
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements; i++) {
        values[i] = sqrt(square_root_function->epsilon + tensor->data[i]);
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

Tensor_ptr derivative_square_root_function(const void* function, const Tensor* tensor, const Tensor* backward) {
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
        values[i] = 1.0 / (2.0 * tensor->data[i]);
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
