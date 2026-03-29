#include "MultiplyByConstant.h"

#include "Memory/Memory.h"

#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

Multiply_by_constant_ptr create_multiply_by_constant(double constant) {
    Multiply_by_constant_ptr result = malloc_(sizeof(Multiply_by_constant));
    if (result == NULL) {
        return NULL;
    }
    result->constant = constant;
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_multiply_by_constant;
    result->function.derivative = derivative_multiply_by_constant;
    return result;
}

void free_multiply_by_constant(Multiply_by_constant_ptr multiply_by_constant) {
    if (multiply_by_constant == NULL) {
        return;
    }
    free_(multiply_by_constant);
}

Tensor_ptr calculate_multiply_by_constant(const void* function, const Tensor* tensor) {
    const Multiply_by_constant* multiply_by_constant = function;
    double* values;
    int i;
    if (multiply_by_constant == NULL || tensor == NULL) {
        return NULL;
    }
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements; i++) {
        values[i] = multiply_by_constant->constant * tensor->data[i];
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

Tensor_ptr derivative_multiply_by_constant(const void* function, const Tensor* tensor, const Tensor* backward) {
    const Multiply_by_constant* multiply_by_constant = function;
    double* values;
    int i;
    if (multiply_by_constant == NULL || tensor == NULL || backward == NULL) {
        return NULL;
    }
    if (tensor->dimensions != backward->dimensions || tensor->total_elements != backward->total_elements) {
        return NULL;
    }
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements; i++) {
        values[i] = multiply_by_constant->constant * backward->data[i];
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}
