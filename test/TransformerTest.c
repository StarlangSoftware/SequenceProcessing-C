#include "Classification/Transformer.h"
#include "Classification/TransformerTokenStore.h"

#include "ArrayList.h"
#include "Initialization/Initialization.h"
#include "Memory/Memory.h"
#include "Node/ComputationalNode.h"
#include "Optimizer/Optimizer.h"
#include "Optimizer/StochasticGradientDescent.h"
#include "Performance/ClassificationPerformance.h"
#include "Functions/MultiplyByConstant.h"
#include "Vector.h"

#include <float.h>

static Transformer_parameter_ptr create_test_parameter(void) {
    Array_list_ptr empty_ints = create_array_list();
    Array_list_ptr empty_functions = create_array_list();
    Array_list_ptr empty_doubles = create_array_list();
    Optimizer_ptr optimizer = create_optimizer(0.1, 1.0);
    Transformer_parameter_ptr parameter = create_transformer_parameter(
            13,
            2,
            optimizer,
            Random,
            NULL,
            4,
            2,
            8,
            1e-8,
            empty_ints,
            empty_ints,
            empty_functions,
            empty_functions,
            empty_doubles,
            empty_doubles,
            empty_doubles,
            empty_doubles);
    free_array_list(empty_ints, free_);
    free_array_list(empty_functions, NULL);
    free_array_list(empty_doubles, free_);
    return parameter;
}

static Transformer_parameter_ptr create_trainable_test_parameter(Optimizer_ptr optimizer) {
    Array_list_ptr empty_ints = create_array_list();
    Array_list_ptr empty_functions = create_array_list();
    Array_list_ptr gamma_input = create_array_list();
    Array_list_ptr gamma_output = create_array_list();
    Array_list_ptr beta_input = create_array_list();
    Array_list_ptr beta_output = create_array_list();
    Transformer_parameter_ptr parameter;
    array_list_add_double(gamma_input, 1.0);
    array_list_add_double(gamma_input, 1.0);
    array_list_add_double(gamma_output, 1.0);
    array_list_add_double(gamma_output, 1.0);
    array_list_add_double(gamma_output, 1.0);
    array_list_add_double(beta_input, 0.0);
    array_list_add_double(beta_input, 0.0);
    array_list_add_double(beta_output, 0.0);
    array_list_add_double(beta_output, 0.0);
    array_list_add_double(beta_output, 0.0);
    parameter = create_transformer_parameter(
            13,
            2,
            optimizer,
            Random,
            NULL,
            1,
            1,
            3,
            1e-8,
            empty_ints,
            empty_ints,
            empty_functions,
            empty_functions,
            gamma_input,
            gamma_output,
            beta_input,
            beta_output);
    free_array_list(empty_ints, free_);
    free_array_list(empty_functions, NULL);
    free_array_list(gamma_input, free_);
    free_array_list(gamma_output, free_);
    free_array_list(beta_input, free_);
    free_array_list(beta_output, free_);
    return parameter;
}

static Transformer_token_store_ptr create_test_token_store(void) {
    Transformer_token_store_ptr token_store = create_transformer_token_store();
    double hello_values[2] = {0.1, 0.2};
    double start_values[2] = {0.3, 0.4};
    double end_values[2] = {0.5, 0.6};
    Vector_ptr hello_vector;
    Vector_ptr start_vector;
    Vector_ptr end_vector;
    if (token_store == NULL) {
        return NULL;
    }
    hello_vector = create_vector4(hello_values, 2);
    start_vector = create_vector4(start_values, 2);
    end_vector = create_vector4(end_values, 2);
    if (!transformer_token_store_add(token_store, "hello", hello_vector) ||
        !transformer_token_store_add(token_store, "<S>", start_vector) ||
        !transformer_token_store_add(token_store, "</S>", end_vector)) {
        free_vector(hello_vector);
        free_vector(start_vector);
        free_vector(end_vector);
        free_transformer_token_store(token_store);
        return NULL;
    }
    free_vector(hello_vector);
    free_vector(start_vector);
    free_vector(end_vector);
    transformer_token_store_sort(token_store);
    return token_store;
}

static Transformer_token_store_ptr create_feedback_token_store(void) {
    Transformer_token_store_ptr token_store = create_transformer_token_store();
    double hello_values[3] = {1.2, 0.1, 0.2};
    double start_values[3] = {0.1, 0.2, 1.4};
    double end_values[3] = {1.5, 0.1, 0.1};
    Vector_ptr hello_vector;
    Vector_ptr start_vector;
    Vector_ptr end_vector;
    if (token_store == NULL) {
        return NULL;
    }
    hello_vector = create_vector4(hello_values, 3);
    start_vector = create_vector4(start_values, 3);
    end_vector = create_vector4(end_values, 3);
    if (!transformer_token_store_add(token_store, "hello", hello_vector) ||
        !transformer_token_store_add(token_store, "<S>", start_vector) ||
        !transformer_token_store_add(token_store, "</S>", end_vector)) {
        free_vector(hello_vector);
        free_vector(start_vector);
        free_vector(end_vector);
        free_transformer_token_store(token_store);
        return NULL;
    }
    free_vector(hello_vector);
    free_vector(start_vector);
    free_vector(end_vector);
    transformer_token_store_sort(token_store);
    return token_store;
}

static int test_transformer_token_store_vectors(void) {
    Transformer_token_store_ptr token_store = create_test_token_store();
    int start_index = transformer_token_store_find_index(token_store, "<S>");
    const Vector* start_vector = transformer_token_store_get_vector(token_store, start_index);
    int success = token_store != NULL &&
                  transformer_token_store_get_embedding_size(token_store) == 2 &&
                  start_index >= 0 &&
                  start_vector != NULL &&
                  get_value(start_vector, 0) == 0.3 &&
                  get_value(start_vector, 1) == 0.4;
    free_transformer_token_store(token_store);
    return success;
}

static int test_transformer_constructor_scans_start_and_end_tokens(void) {
    Transformer_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Transformer_token_store_ptr token_store = create_test_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    int success = model != NULL &&
                  transformer_model_get_token_store(model) == token_store &&
                  transformer_model_get_start_index(model) == transformer_token_store_find_index(token_store, "<S>") &&
                  transformer_model_get_end_index(model) == transformer_token_store_find_index(token_store, "</S>");
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_(optimizer);
    return success;
}

static int test_transformer_constructor_handles_missing_tokens(void) {
    Array_list_ptr empty_ints = create_array_list();
    Array_list_ptr empty_functions = create_array_list();
    Array_list_ptr empty_doubles = create_array_list();
    Optimizer_ptr optimizer = create_optimizer(0.1, 1.0);
    Transformer_parameter_ptr parameter = create_transformer_parameter(
            13, 2, optimizer, Random, NULL, 4, 2, 8, 1e-8,
            empty_ints, empty_ints, empty_functions, empty_functions,
            empty_doubles, empty_doubles, empty_doubles, empty_doubles);
    Transformer_token_store_ptr token_store = create_transformer_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    int success = model != NULL &&
                  transformer_model_get_start_index(model) == -1 &&
                  transformer_model_get_end_index(model) == -1;
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_array_list(empty_ints, free_);
    free_array_list(empty_functions, NULL);
    free_array_list(empty_doubles, free_);
    free_(optimizer);
    return success;
}

static int test_transformer_positional_encoding_helper(void) {
    Transformer_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Transformer_token_store_ptr token_store = create_test_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    double* values = malloc_(4 * sizeof(double));
    int shape[2] = {2, 2};
    Tensor_ptr tensor;
    Tensor_ptr encoded;
    int success;
    values[0] = 1.0;
    values[1] = 2.0;
    values[2] = 3.0;
    values[3] = 4.0;
    tensor = create_tensor3(values, shape, 2);
    encoded = transformer_model_positional_encoding(model, tensor, 2);
    success = encoded != NULL &&
              encoded->shape[0] == 2 &&
              encoded->shape[1] == 2 &&
              encoded->data[0] > 1.0 &&
              encoded->data[1] > 2.0 &&
              encoded->data[2] > 3.0;
    free_tensor(encoded);
    free_tensor(tensor);
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_(optimizer);
    return success;
}

static int test_transformer_create_packed_inputs_helper(void) {
    Transformer_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Transformer_token_store_ptr token_store = create_test_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    double* values = malloc_(11 * sizeof(double));
    int shape[1] = {11};
    Tensor_ptr instance;
    Transformer_packed_inputs_ptr packed_inputs;
    Array_list_ptr class_labels;
    const Tensor* encoder_input;
    const Tensor* decoder_input;
    int success;
    values[0] = 1.0;
    values[1] = 2.0;
    values[2] = 3.0;
    values[3] = 4.0;
    values[4] = DBL_MAX;
    values[5] = 5.0;
    values[6] = 6.0;
    values[7] = 0.0;
    values[8] = 7.0;
    values[9] = 8.0;
    values[10] = 1.0;
    instance = create_tensor3(values, shape, 1);
    packed_inputs = transformer_model_create_packed_inputs(model, instance, 2);
    class_labels = transformer_packed_inputs_get_class_labels(packed_inputs);
    encoder_input = transformer_packed_inputs_get_encoder_input(packed_inputs);
    decoder_input = transformer_packed_inputs_get_decoder_input(packed_inputs);
    success = packed_inputs != NULL &&
              encoder_input != NULL &&
              decoder_input != NULL &&
              class_labels != NULL &&
              encoder_input->shape[0] == 2 &&
              encoder_input->shape[1] == 2 &&
              decoder_input->shape[0] == 2 &&
              decoder_input->shape[1] == 2 &&
              class_labels->size == 2 &&
              array_list_get_int(class_labels, 0) == 0 &&
              array_list_get_int(class_labels, 1) == 1;
    free_transformer_packed_inputs(packed_inputs);
    free_tensor(instance);
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_(optimizer);
    return success;
}

static int test_transformer_graph_input_initialization_and_apply(void) {
    Transformer_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Transformer_token_store_ptr token_store = create_test_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    double* values = malloc_(11 * sizeof(double));
    int shape[1] = {11};
    Tensor_ptr instance;
    Transformer_packed_inputs_ptr packed_inputs;
    Array_list_ptr input_nodes;
    Computational_node_ptr encoder_input;
    Computational_node_ptr decoder_input;
    int success;
    values[0] = 1.0;
    values[1] = 2.0;
    values[2] = 3.0;
    values[3] = 4.0;
    values[4] = DBL_MAX;
    values[5] = 5.0;
    values[6] = 6.0;
    values[7] = 0.0;
    values[8] = 7.0;
    values[9] = 8.0;
    values[10] = 1.0;
    instance = create_tensor3(values, shape, 1);
    packed_inputs = transformer_model_create_packed_inputs(model, instance, 2);
    success = transformer_model_initialize_graph_inputs(model) &&
              transformer_model_apply_packed_inputs(model, packed_inputs);
    input_nodes = transformer_model_get_input_nodes(model);
    encoder_input = array_list_get(input_nodes, 0);
    decoder_input = array_list_get(input_nodes, 1);
    success = success &&
              input_nodes != NULL &&
              input_nodes->size == 2 &&
              encoder_input != NULL &&
              decoder_input != NULL &&
              encoder_input->value != NULL &&
              decoder_input->value != NULL &&
              encoder_input->value->shape[0] == 2 &&
              encoder_input->value->shape[1] == 2 &&
              decoder_input->value->shape[0] == 2 &&
              decoder_input->value->shape[1] == 2;
    free_transformer_packed_inputs(packed_inputs);
    free_tensor(instance);
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_(optimizer);
    return success;
}

static int test_transformer_set_input_node_and_class_label_slot(void) {
    Transformer_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Transformer_token_store_ptr token_store = create_test_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    Array_list_ptr input_nodes;
    Computational_node_ptr decoder_input;
    Computational_node_ptr class_label_input;
    double first_values[2] = {0.25, 0.5};
    double second_values[2] = {0.75, 1.0};
    Vector_ptr first_vector = create_vector4(first_values, 2);
    Vector_ptr second_vector = create_vector4(second_values, 2);
    int success = transformer_model_initialize_graph_inputs(model);
    input_nodes = transformer_model_get_input_nodes(model);
    decoder_input = array_list_get(input_nodes, 1);
    success = success &&
              transformer_model_set_input_node(model, 1, first_vector, decoder_input) &&
              transformer_model_set_input_node(model, 2, second_vector, decoder_input);
    class_label_input = transformer_model_add_class_label_input(model);
    success = success &&
              decoder_input != NULL &&
              decoder_input->value != NULL &&
              decoder_input->value->shape[0] == 2 &&
              decoder_input->value->shape[1] == 2 &&
              decoder_input->value->total_elements == 4 &&
              class_label_input != NULL &&
              transformer_model_get_input_nodes(model)->size == 3;
    free_vector(first_vector);
    free_vector(second_vector);
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_(optimizer);
    return success;
}

static int test_transformer_build_graph_stage(void) {
    Optimizer_ptr optimizer = create_stochastic_gradient(0.1, 0.5);
    Transformer_parameter_ptr parameter = create_trainable_test_parameter(optimizer);
    Transformer_token_store_ptr token_store = create_test_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    Array_list_ptr input_nodes;
    /*
     * This validates only the staged local graph shell: graph claim plus the
     * three expected input slots. It does not assert Java loss-node forward
     * parity.
     */
    int success = transformer_model_build_graph(model);
    input_nodes = transformer_model_get_input_nodes(model);
    success = success &&
              transformer_model_is_graph_initialized(model) &&
              input_nodes != NULL &&
              input_nodes->size == 3;
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_(optimizer);
    return success;
}

static int test_transformer_train_stage(void) {
    Optimizer_ptr optimizer = create_stochastic_gradient(0.1, 0.5);
    Transformer_parameter_ptr parameter = create_trainable_test_parameter(optimizer);
    Transformer_token_store_ptr token_store = create_test_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    Array_list_ptr train_set = create_array_list();
    double* values = malloc_(9 * sizeof(double));
    int shape[1] = {9};
    Tensor_ptr instance = create_tensor3(values, shape, 1);
    int success;
    values[0] = 0.1;
    values[1] = 0.2;
    values[2] = DBL_MAX;
    values[3] = 0.3;
    values[4] = 0.0;
    values[5] = 0.4;
    values[6] = 1.0;
    values[7] = 0.5;
    values[8] = 2.0;
    array_list_add(train_set, instance);
    /*
     * This validates only the staged training compromise already implemented in
     * this repo: build-on-demand, packed inputs, class-index backprop, and
     * epoch-level learning-rate decay. It does not claim Java `test(...)`
     * parity or loss-node forward-value parity.
     */
    success = transformer_model_train(model, train_set) &&
              transformer_model_is_graph_initialized(model) &&
              transformer_model_get_input_nodes(model) != NULL &&
              transformer_model_get_input_nodes(model)->size == 3 &&
              optimizer->learning_rate == 0.025;
    free_array_list(train_set, (void (*)(void*)) free_tensor);
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_(optimizer);
    return success;
}

static int test_transformer_test_parity(void) {
    Transformer_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Transformer_token_store_ptr token_store = create_test_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    Array_list_ptr test_set = create_array_list();
    Array_list_ptr input_nodes;
    Computational_node_ptr decoder_input;
    Computational_node_ptr output_node;
    Classification_performance_ptr performance;
    double* values = malloc_(6 * sizeof(double));
    int shape[1] = {6};
    Tensor_ptr instance = create_tensor3(values, shape, 1);
    int success;
    values[0] = 1.0;
    values[1] = 2.0;
    values[2] = DBL_MAX;
    values[3] = 5.0;
    values[4] = 6.0;
    values[5] = 0.0;
    success = transformer_model_initialize_graph_inputs(model);
    input_nodes = transformer_model_get_input_nodes(model);
    decoder_input = success ? array_list_get(input_nodes, 1) : NULL;
    output_node = (decoder_input != NULL)
                  ? transformer_model_add_edge(model, decoder_input, (Function*) create_multiply_by_constant(1.0), false)
                  : NULL;
    transformer_model_set_output_node(model, output_node);
    array_list_add(test_set, instance);
    performance = transformer_model_test(model, test_set);
    success = success &&
              output_node != NULL &&
              performance != NULL &&
              sequence_processing_classification_performance_get_accuracy(performance) == 1.0;
    free_sequence_processing_classification_performance(performance);
    free_array_list(test_set, (void (*)(void*)) free_tensor);
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_(optimizer);
    return success;
}

static int test_transformer_test_decoder_feedback_loop(void) {
    Array_list_ptr empty_ints = create_array_list();
    Array_list_ptr empty_functions = create_array_list();
    Array_list_ptr empty_doubles = create_array_list();
    Optimizer_ptr optimizer = create_optimizer(0.1, 1.0);
    Transformer_parameter_ptr parameter = create_transformer_parameter(
            13, 2, optimizer, Random, NULL, 3, 1, 3, 1e-8,
            empty_ints, empty_ints, empty_functions, empty_functions,
            empty_doubles, empty_doubles, empty_doubles, empty_doubles);
    Transformer_token_store_ptr token_store = create_feedback_token_store();
    Transformer_model_ptr model = create_transformer_model(parameter, token_store);
    Array_list_ptr test_set = create_array_list();
    Array_list_ptr input_nodes;
    Computational_node_ptr decoder_input;
    Computational_node_ptr output_node;
    Classification_performance_ptr performance;
    double* values = malloc_(12 * sizeof(double));
    int shape[1] = {12};
    Tensor_ptr instance = create_tensor3(values, shape, 1);
    int success;

    values[0] = 0.2;
    values[1] = 0.3;
    values[2] = 0.4;
    values[3] = DBL_MAX;
    values[4] = 0.6;
    values[5] = 0.5;
    values[6] = 0.4;
    values[7] = 2.0;
    values[8] = 0.3;
    values[9] = 0.2;
    values[10] = 0.1;
    values[11] = 0.0;

    success = transformer_model_initialize_graph_inputs(model);
    input_nodes = transformer_model_get_input_nodes(model);
    decoder_input = success ? array_list_get(input_nodes, 1) : NULL;
    output_node = (decoder_input != NULL)
                  ? transformer_model_add_edge(model, decoder_input, (Function*) create_multiply_by_constant(1.0), false)
                  : NULL;
    transformer_model_set_output_node(model, output_node);
    array_list_add(test_set, instance);
    performance = transformer_model_test(model, test_set);
    success = success &&
              output_node != NULL &&
              performance != NULL &&
              sequence_processing_classification_performance_get_accuracy(performance) == 1.0;

    free_sequence_processing_classification_performance(performance);
    free_array_list(test_set, (void (*)(void*)) free_tensor);
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_transformer_token_store(token_store);
    free_array_list(empty_ints, free_);
    free_array_list(empty_functions, NULL);
    free_array_list(empty_doubles, free_);
    free_(optimizer);
    return success;
}

int main(void) {
    if (!test_transformer_token_store_vectors()) {
        return 1;
    }
    if (!test_transformer_constructor_scans_start_and_end_tokens()) {
        return 1;
    }
    if (!test_transformer_constructor_handles_missing_tokens()) {
        return 1;
    }
    if (!test_transformer_positional_encoding_helper()) {
        return 1;
    }
    if (!test_transformer_create_packed_inputs_helper()) {
        return 1;
    }
    if (!test_transformer_graph_input_initialization_and_apply()) {
        return 1;
    }
    if (!test_transformer_set_input_node_and_class_label_slot()) {
        return 1;
    }
    if (!test_transformer_build_graph_stage()) {
        return 1;
    }
    if (!test_transformer_train_stage()) {
        return 1;
    }
    if (!test_transformer_test_parity()) {
        return 1;
    }
    if (!test_transformer_test_decoder_feedback_loop()) {
        return 1;
    }
    return 0;
}
