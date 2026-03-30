#include "Classification/GatedRecurrentUnitModel.h"

#include "ArrayList.h"
#include "Function/Sigmoid.h"
#include "Initialization/Initialization.h"
#include "Memory/Memory.h"
#include "Optimizer/Optimizer.h"
#include "Optimizer/StochasticGradientDescent.h"

static Recurrent_neural_network_parameter_ptr create_test_parameter(void) {
    Array_list_ptr hidden_layers = create_array_list();
    Array_list_ptr functions = create_array_list();
    Optimizer_ptr optimizer = create_optimizer(0.1, 1.0);
    Recurrent_neural_network_parameter_ptr parameter =
            create_recurrent_neural_network_parameter(13, 2, optimizer, Random, NULL, hidden_layers, functions, 3);
    free_array_list(hidden_layers, free_);
    free_array_list(functions, NULL);
    return parameter;
}

static int test_gru_constructor_and_base_access(void) {
    Recurrent_neural_network_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Gated_recurrent_unit_model_ptr model = create_gated_recurrent_unit_model(parameter, 5);
    Recurrent_neural_network_model_ptr base = gated_recurrent_unit_model_get_base(model);
    int success = model != NULL &&
                  base != NULL &&
                  base->parameters == parameter &&
                  base->word_embedding_length == 5 &&
                  base->input_nodes != NULL &&
                  base->switches != NULL &&
                  base->input_nodes->size == 0 &&
                  base->switches->size == 0;
    free_gated_recurrent_unit_model(model);
    free_recurrent_neural_network_parameter(parameter);
    free_(optimizer);
    return success;
}

static int test_gru_train_builds_graph_and_runs_shared_training(void) {
    Array_list_ptr hidden_layers = create_array_list();
    Array_list_ptr functions = create_array_list();
    Sigmoid_ptr sigmoid1 = create_sigmoid();
    Sigmoid_ptr sigmoid2 = create_sigmoid();
    Optimizer_ptr optimizer = create_stochastic_gradient(0.1, 0.5);
    Recurrent_neural_network_parameter_ptr parameter;
    Gated_recurrent_unit_model_ptr model;
    Recurrent_neural_network_model_ptr base;
    Array_list_ptr train_set = create_array_list();
    double* values = malloc_(2 * sizeof(double));
    int shape[1] = {2};
    Tensor_ptr instance;
    int success;
    array_list_add_int(hidden_layers, 2);
    array_list_add(functions, sigmoid1);
    array_list_add(functions, sigmoid2);
    parameter = create_recurrent_neural_network_parameter(13, 2, optimizer, Random, NULL, hidden_layers, functions, 2);
    model = create_gated_recurrent_unit_model(parameter, 1);
    base = gated_recurrent_unit_model_get_base(model);
    values[0] = 0.5;
    values[1] = 1.0;
    instance = create_tensor3(values, shape, 1);
    array_list_add(train_set, instance);
    success = gated_recurrent_unit_model_train(model, train_set) &&
              base != NULL &&
              base->input_nodes != NULL &&
              base->switches != NULL &&
              base->input_nodes->size == 2 &&
              base->switches->size == 1 &&
              optimizer->learning_rate == 0.025;
    free_array_list(train_set, (void (*)(void*)) free_tensor);
    free_gated_recurrent_unit_model(model);
    free_recurrent_neural_network_parameter(parameter);
    free_sigmoid(sigmoid1);
    free_sigmoid(sigmoid2);
    free_array_list(hidden_layers, free_);
    free_array_list(functions, NULL);
    free_(optimizer);
    return success;
}

static int test_gru_repeated_train_is_explicitly_rejected(void) {
    Array_list_ptr hidden_layers = create_array_list();
    Array_list_ptr functions = create_array_list();
    Sigmoid_ptr sigmoid1 = create_sigmoid();
    Sigmoid_ptr sigmoid2 = create_sigmoid();
    Optimizer_ptr optimizer = create_stochastic_gradient(0.1, 0.5);
    Recurrent_neural_network_parameter_ptr parameter;
    Gated_recurrent_unit_model_ptr model;
    Recurrent_neural_network_model_ptr base;
    Array_list_ptr train_set = create_array_list();
    double* values = malloc_(2 * sizeof(double));
    int shape[1] = {2};
    Tensor_ptr instance;
    int success;
    array_list_add_int(hidden_layers, 2);
    array_list_add(functions, sigmoid1);
    array_list_add(functions, sigmoid2);
    parameter = create_recurrent_neural_network_parameter(13, 2, optimizer, Random, NULL, hidden_layers, functions, 2);
    model = create_gated_recurrent_unit_model(parameter, 1);
    base = gated_recurrent_unit_model_get_base(model);
    values[0] = 0.5;
    values[1] = 1.0;
    instance = create_tensor3(values, shape, 1);
    array_list_add(train_set, instance);
    success = gated_recurrent_unit_model_train(model, train_set) &&
              !gated_recurrent_unit_model_train(model, train_set) &&
              base != NULL &&
              base->graph_initialized &&
              base->input_nodes->size == 2 &&
              base->switches->size == 1 &&
              optimizer->learning_rate == 0.025;
    free_array_list(train_set, (void (*)(void*)) free_tensor);
    free_gated_recurrent_unit_model(model);
    free_recurrent_neural_network_parameter(parameter);
    free_sigmoid(sigmoid1);
    free_sigmoid(sigmoid2);
    free_array_list(hidden_layers, free_);
    free_array_list(functions, NULL);
    free_(optimizer);
    return success;
}

int main(void) {
    if (!test_gru_constructor_and_base_access()) {
        return 1;
    }
    if (!test_gru_train_builds_graph_and_runs_shared_training()) {
        return 1;
    }
    if (!test_gru_repeated_train_is_explicitly_rejected()) {
        return 1;
    }
    return 0;
}
