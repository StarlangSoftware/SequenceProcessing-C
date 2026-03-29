#include "Functions/AdditionByConstant.h"
#include "Functions/MultiplyByConstant.h"
#include "Functions/SequenceFunctionEdge.h"
#include "Functions/Transpose.h"

#include "Memory/Memory.h"
#include "Node/ComputationalNode.h"

#include <stddef.h>

extern Computational_graph_ptr create_computational_graph(void);
extern void free_computational_graph(Computational_graph_ptr graph);

static Tensor_ptr create_test_tensor_2x2(double a, double b, double c, double d) {
    double* values = malloc_(4 * sizeof(double));
    int shape[2] = {2, 2};
    values[0] = a;
    values[1] = b;
    values[2] = c;
    values[3] = d;
    return create_tensor3(values, shape, 2);
}

static int test_addition_by_constant(void) {
    Addition_by_constant_ptr function = create_addition_by_constant(3.0);
    Tensor_ptr input = create_test_tensor_2x2(1.0, 2.0, 3.0, 4.0);
    Tensor_ptr backward = create_test_tensor_2x2(5.0, 6.0, 7.0, 8.0);
    Tensor_ptr output = calculate_addition_by_constant(function, input);
    Tensor_ptr derivative = derivative_addition_by_constant(function, input, backward);
    int success = output != NULL &&
                  derivative != NULL &&
                  output->data[0] == 4.0 &&
                  output->data[3] == 7.0 &&
                  derivative->data[0] == 5.0 &&
                  derivative->data[3] == 8.0;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output);
    free_tensor(derivative);
    free_addition_by_constant(function);
    return success;
}

static int test_multiply_by_constant(void) {
    Multiply_by_constant_ptr function = create_multiply_by_constant(2.5);
    Tensor_ptr input = create_test_tensor_2x2(1.0, 2.0, 3.0, 4.0);
    Tensor_ptr backward = create_test_tensor_2x2(5.0, 6.0, 7.0, 8.0);
    Tensor_ptr output = calculate_multiply_by_constant(function, input);
    Tensor_ptr derivative = derivative_multiply_by_constant(function, input, backward);
    int success = output != NULL &&
                  derivative != NULL &&
                  output->data[0] == 2.5 &&
                  output->data[3] == 10.0 &&
                  derivative->data[0] == 12.5 &&
                  derivative->data[3] == 20.0;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output);
    free_tensor(derivative);
    free_multiply_by_constant(function);
    return success;
}

static int test_transpose(void) {
    Transpose_function_ptr function = create_transpose_function();
    Tensor_ptr input = create_test_tensor_2x2(1.0, 2.0, 3.0, 4.0);
    Tensor_ptr backward = create_test_tensor_2x2(5.0, 6.0, 7.0, 8.0);
    Tensor_ptr output = calculate_transpose_function(function, input);
    Tensor_ptr derivative = derivative_transpose_function(function, input, backward);
    int success = output != NULL &&
                  derivative != NULL &&
                  output->shape[0] == 2 &&
                  output->shape[1] == 2 &&
                  output->data[0] == 1.0 &&
                  output->data[1] == 3.0 &&
                  output->data[2] == 2.0 &&
                  output->data[3] == 4.0 &&
                  derivative->data[0] == 5.0 &&
                  derivative->data[1] == 7.0 &&
                  derivative->data[2] == 6.0 &&
                  derivative->data[3] == 8.0;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output);
    free_tensor(derivative);
    free_transpose_function(function);
    return success;
}

static int test_sequence_function_edge(void) {
    Computational_graph_ptr graph = create_computational_graph();
    Computational_node_ptr input = create_computational_node3(false, false);
    Addition_by_constant_ptr function = create_addition_by_constant(1.0);
    Computational_node_ptr child = add_sequence_function_edge(graph, input, (Function*) function, true);
    /*
     * Ownership of `function` transfers on success, so the test only releases
     * the graph afterwards.
     */
    int success = child != NULL &&
                  child->function == function &&
                  child->is_biased;
    free_computational_graph(graph);
    return success;
}

int main(void) {
    if (!test_addition_by_constant()) {
        return 1;
    }
    if (!test_multiply_by_constant()) {
        return 1;
    }
    if (!test_transpose()) {
        return 1;
    }
    if (!test_sequence_function_edge()) {
        return 1;
    }
    return 0;
}
