#include "Functions/AdditionByConstant.h"
#include "Functions/Inverse.h"
#include "Functions/Mask.h"
#include "Functions/Mean.h"
#include "Functions/MultiplyByConstant.h"
#include "Functions/RemoveBias.h"
#include "Functions/SequenceFunctionEdge.h"
#include "Functions/SquareRoot.h"
#include "Functions/Switch.h"
#include "Functions/Transpose.h"
#include "Functions/Variance.h"

#include "Memory/Memory.h"
#include "Node/ComputationalNode.h"

#include <math.h>
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

static Tensor_ptr create_test_tensor_2x3(double a, double b, double c, double d, double e, double f) {
    double* values = malloc_(6 * sizeof(double));
    int shape[2] = {2, 3};
    values[0] = a;
    values[1] = b;
    values[2] = c;
    values[3] = d;
    values[4] = e;
    values[5] = f;
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

static int test_remove_bias(void) {
    Remove_bias_ptr function = create_remove_bias();
    Tensor_ptr input = create_test_tensor_2x2(1.0, 2.0, 3.0, 4.0);
    Tensor_ptr backward = create_test_tensor_2x2(5.0, 6.0, 7.0, 8.0);
    Tensor_ptr output = calculate_remove_bias(function, input);
    Tensor_ptr derivative = derivative_remove_bias(function, input, backward);
    int success = output != NULL &&
                  derivative != NULL &&
                  output->dimensions == 2 &&
                  output->shape[0] == 1 &&
                  output->shape[1] == 3 &&
                  output->data[0] == 1.0 &&
                  output->data[2] == 3.0 &&
                  derivative->shape[0] == 1 &&
                  derivative->shape[1] == 5 &&
                  derivative->data[0] == 5.0 &&
                  derivative->data[3] == 8.0 &&
                  derivative->data[4] == 0.0;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output);
    free_tensor(derivative);
    free_remove_bias(function);
    return success;
}

static int test_mean(void) {
    Mean_function_ptr function = create_mean_function();
    Tensor_ptr input = create_test_tensor_2x3(1.0, 2.0, 3.0, 4.0, 7.0, 10.0);
    Tensor_ptr backward = create_test_tensor_2x3(3.0, 6.0, 9.0, 12.0, 15.0, 18.0);
    Tensor_ptr output = calculate_mean_function(function, input);
    Tensor_ptr derivative = derivative_mean_function(function, input, backward);
    int success = output != NULL &&
                  derivative != NULL &&
                  output->data[0] == 2.0 &&
                  output->data[1] == 2.0 &&
                  output->data[2] == 2.0 &&
                  output->data[3] == 7.0 &&
                  output->data[4] == 7.0 &&
                  output->data[5] == 7.0 &&
                  derivative->data[0] == 1.0 &&
                  derivative->data[1] == 2.0 &&
                  derivative->data[2] == 3.0 &&
                  derivative->data[3] == 4.0 &&
                  derivative->data[4] == 5.0 &&
                  derivative->data[5] == 6.0;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output);
    free_tensor(derivative);
    free_mean_function(function);
    return success;
}

static int test_mask(void) {
    Mask_function_ptr function = create_mask_function();
    Tensor_ptr input = create_test_tensor_2x2(1.0, 2.0, 3.0, 4.0);
    Tensor_ptr backward = create_test_tensor_2x2(5.0, 6.0, 7.0, 8.0);
    Tensor_ptr output = calculate_mask_function(function, input);
    Tensor_ptr derivative = derivative_mask_function(function, input, backward);
    int success = output != NULL &&
                  derivative != NULL &&
                  output->data[0] == 1.0 &&
                  isinf(output->data[1]) &&
                  output->data[1] < 0.0 &&
                  output->data[2] == 3.0 &&
                  output->data[3] == 4.0 &&
                  derivative->data[0] == 5.0 &&
                  derivative->data[3] == 8.0;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output);
    free_tensor(derivative);
    free_mask_function(function);
    return success;
}

static int test_switch(void) {
    Switch_function_ptr function = create_switch_function();
    Tensor_ptr input = create_test_tensor_2x2(1.0, 2.0, 3.0, 4.0);
    Tensor_ptr backward = create_test_tensor_2x2(5.0, 6.0, 7.0, 8.0);
    Tensor_ptr output_on = calculate_switch_function(function, input);
    Tensor_ptr derivative_on = derivative_switch_function(function, input, backward);
    set_switch_turn(function, false);
    Tensor_ptr output_off = calculate_switch_function(function, input);
    Tensor_ptr derivative_off = derivative_switch_function(function, input, backward);
    int success = output_on != NULL &&
                  derivative_on != NULL &&
                  output_off != NULL &&
                  derivative_off != NULL &&
                  output_on->data[0] == 1.0 &&
                  output_on->data[3] == 4.0 &&
                  derivative_on->data[0] == 5.0 &&
                  derivative_on->data[3] == 8.0 &&
                  output_off->data[0] == 0.0 &&
                  output_off->data[3] == 0.0 &&
                  derivative_off->data[0] == 0.0 &&
                  derivative_off->data[3] == 0.0;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output_on);
    free_tensor(derivative_on);
    free_tensor(output_off);
    free_tensor(derivative_off);
    free_switch_function(function);
    return success;
}

static int test_inverse(void) {
    Inverse_function_ptr function = create_inverse_function();
    Tensor_ptr input = create_test_tensor_2x2(1.0, 2.0, 4.0, 5.0);
    Tensor_ptr backward = create_test_tensor_2x2(3.0, 6.0, 9.0, 12.0);
    Tensor_ptr output = calculate_inverse_function(function, input);
    Tensor_ptr derivative = derivative_inverse_function(function, input, backward);
    int success = output != NULL &&
                  derivative != NULL &&
                  output->data[0] == 1.0 &&
                  output->data[1] == 0.5 &&
                  output->data[2] == 0.25 &&
                  output->data[3] == 0.2 &&
                  derivative->data[0] == -3.0 &&
                  derivative->data[1] == -24.0 &&
                  derivative->data[2] == -144.0 &&
                  derivative->data[3] == -300.0;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output);
    free_tensor(derivative);
    free_inverse_function(function);
    return success;
}

static int test_square_root(void) {
    Square_root_function_ptr function = create_square_root_function(1.0);
    Tensor_ptr input = create_test_tensor_2x2(3.0, 8.0, 15.0, 24.0);
    Tensor_ptr backward = create_test_tensor_2x2(3.0, 6.0, 9.0, 12.0);
    Tensor_ptr output = calculate_square_root_function(function, input);
    Tensor_ptr derivative = derivative_square_root_function(function, input, backward);
    int success = output != NULL &&
                  derivative != NULL &&
                  output->data[0] == 2.0 &&
                  output->data[1] == 3.0 &&
                  output->data[2] == 4.0 &&
                  output->data[3] == 5.0 &&
                  derivative->data[0] == 0.5 &&
                  derivative->data[1] == 0.375 &&
                  derivative->data[2] == 0.3 &&
                  derivative->data[3] == 0.25;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output);
    free_tensor(derivative);
    free_square_root_function(function);
    return success;
}

static int test_variance(void) {
    Variance_function_ptr function = create_variance_function();
    Tensor_ptr input = create_test_tensor_2x2(1.0, 3.0, 4.0, 9.0);
    Tensor_ptr backward = create_test_tensor_2x2(3.0, 6.0, 9.0, 12.0);
    Tensor_ptr output = calculate_variance_function(function, input);
    Tensor_ptr derivative = derivative_variance_function(function, input, backward);
    int success = output != NULL &&
                  derivative != NULL &&
                  output->data[0] == 5.0 &&
                  output->data[1] == 5.0 &&
                  output->data[2] == 48.5 &&
                  output->data[3] == 48.5 &&
                  derivative->data[0] == 4.242640687119286 &&
                  derivative->data[1] == 14.696938456699069 &&
                  derivative->data[2] == 25.455844122715714 &&
                  derivative->data[3] == 50.91168824543143;
    free_tensor(input);
    free_tensor(backward);
    free_tensor(output);
    free_tensor(derivative);
    free_variance_function(function);
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
    if (!test_remove_bias()) {
        return 1;
    }
    if (!test_mean()) {
        return 1;
    }
    if (!test_mask()) {
        return 1;
    }
    if (!test_switch()) {
        return 1;
    }
    if (!test_inverse()) {
        return 1;
    }
    if (!test_square_root()) {
        return 1;
    }
    if (!test_variance()) {
        return 1;
    }
    if (!test_sequence_function_edge()) {
        return 1;
    }
    return 0;
}
