#include "AdditionByConstant.h"

#include "Memory/Memory.h"

/*
 * ComputationalGraph-C does not currently expose a custom function-type enum
 * value. Any non-DROPOUT value is equivalent for execution; only DROPOUT is
 * treated specially in forward_calculation_with_dropout().
 */
#define SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE RELU

Addition_by_constant_ptr create_addition_by_constant(double constant) {
    Addition_by_constant_ptr result = malloc_(sizeof(Addition_by_constant));
    if (result == NULL) {
        return NULL;
    }
    result->constant = constant;
    result->function.function_type = SEQUENCE_PROCESSING_LOCAL_FUNCTION_TYPE;
    result->function.calculate = calculate_addition_by_constant;
    result->function.derivative = derivative_addition_by_constant;
    return result;
}

void free_addition_by_constant(Addition_by_constant_ptr addition_by_constant) {
    if (addition_by_constant == NULL) {
        return;
    }
    free_(addition_by_constant);
}

Tensor_ptr calculate_addition_by_constant(const void* function, const Tensor* tensor) {
    const Addition_by_constant* addition_by_constant = function;
    double* values;
    int i;
    if (addition_by_constant == NULL || tensor == NULL) {
        return NULL;
    }
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < tensor->total_elements; i++) {
        values[i] = tensor->data[i] + addition_by_constant->constant;
    }
    return create_tensor3(values, tensor->shape, tensor->dimensions);
}

Tensor_ptr derivative_addition_by_constant(const void* function, const Tensor* tensor, const Tensor* backward) {
    (void) function;
    (void) tensor;
    if (backward == NULL) {
        return NULL;
    }
    return clone_tensor(backward);
}
