#include "Transpose.h"

#include "Memory/Memory.h"

#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

Transpose_function_ptr create_transpose_function(void) {
    Transpose_function_ptr result = malloc_(sizeof(Transpose_function));
    if (result == NULL) {
        return NULL;
    }
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_transpose_function;
    result->function.derivative = derivative_transpose_function;
    return result;
}

void free_transpose_function(Transpose_function_ptr transpose_function) {
    if (transpose_function == NULL) {
        return;
    }
    free_(transpose_function);
}

Tensor_ptr calculate_transpose_function(const void* function, const Tensor* tensor) {
    int axes[2] = {1, 0};
    (void) function;
    if (tensor == NULL || tensor->dimensions != 2) {
        return NULL;
    }
    return transpose_tensor(tensor, axes);
}

Tensor_ptr derivative_transpose_function(const void* function, const Tensor* tensor, const Tensor* backward) {
    int axes[2] = {1, 0};
    (void) function;
    (void) tensor;
    if (backward == NULL || backward->dimensions != 2) {
        return NULL;
    }
    return transpose_tensor(backward, axes);
}
