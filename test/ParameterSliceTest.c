#include "Parameters/RecurrentNeuralNetworkParameter.h"
#include "Parameters/TransformerParameter.h"
#include "ArrayList.h"
#include "Memory/Memory.h"

#include <stddef.h>

static int test_recurrent_parameter(void) {
    Recurrent_neural_network_parameter_ptr parameter;
    Array_list_ptr hidden_layers = create_array_list();
    Array_list_ptr functions = create_array_list();
    Function function1 = {.function_type = RELU, .calculate = NULL, .derivative = NULL};
    Function function2 = {.function_type = TANH, .calculate = NULL, .derivative = NULL};
    array_list_add_int(hidden_layers, 32);
    array_list_add_int(hidden_layers, 16);
    array_list_add(functions, &function1);
    array_list_add(functions, &function2);
    parameter = create_recurrent_neural_network_parameter(7, 11, NULL, Random, &function1, hidden_layers, functions, 5);
    free_array_list(hidden_layers, free_);
    free_array_list(functions, NULL);
    if (parameter == NULL) {
        return 0;
    }
    if (recurrent_neural_network_parameter_size(parameter) != 2) {
        free_recurrent_neural_network_parameter(parameter);
        return 0;
    }
    if (recurrent_neural_network_parameter_get_class_label_size(parameter) != 5) {
        free_recurrent_neural_network_parameter(parameter);
        return 0;
    }
    if (recurrent_neural_network_parameter_get_hidden_layer(parameter, 1) != 16) {
        free_recurrent_neural_network_parameter(parameter);
        return 0;
    }
    if (recurrent_neural_network_parameter_get_activation_function(parameter, 0) != &function1) {
        free_recurrent_neural_network_parameter(parameter);
        return 0;
    }
    free_recurrent_neural_network_parameter(parameter);
    return 1;
}

static int test_transformer_parameter(void) {
    Transformer_parameter_ptr parameter;
    Array_list_ptr input_hidden_layers = create_array_list();
    Array_list_ptr output_hidden_layers = create_array_list();
    Array_list_ptr input_functions = create_array_list();
    Array_list_ptr output_functions = create_array_list();
    Array_list_ptr gamma_input_values = create_array_list();
    Array_list_ptr gamma_output_values = create_array_list();
    Array_list_ptr beta_input_values = create_array_list();
    Array_list_ptr beta_output_values = create_array_list();
    Function function1 = {.function_type = RELU, .calculate = NULL, .derivative = NULL};
    Function function2 = {.function_type = TANH, .calculate = NULL, .derivative = NULL};
    array_list_add_int(input_hidden_layers, 64);
    array_list_add_int(input_hidden_layers, 32);
    array_list_add_int(output_hidden_layers, 128);
    array_list_add(input_functions, &function1);
    array_list_add(input_functions, &function2);
    array_list_add(output_functions, &function2);
    array_list_add_double(gamma_input_values, 1.5);
    array_list_add_double(gamma_output_values, 2.5);
    array_list_add_double(beta_input_values, 3.5);
    array_list_add_double(beta_output_values, 4.5);
    parameter = create_transformer_parameter(3,
                                             9,
                                             NULL,
                                             Random,
                                             &function1,
                                             15,
                                             4,
                                             99,
                                             1e-6,
                                             input_hidden_layers,
                                             output_hidden_layers,
                                             input_functions,
                                             output_functions,
                                             gamma_input_values,
                                             gamma_output_values,
                                             beta_input_values,
                                             beta_output_values);
    free_array_list(input_hidden_layers, free_);
    free_array_list(output_hidden_layers, free_);
    free_array_list(input_functions, NULL);
    free_array_list(output_functions, NULL);
    free_array_list(gamma_input_values, free_);
    free_array_list(gamma_output_values, free_);
    free_array_list(beta_input_values, free_);
    free_array_list(beta_output_values, free_);
    if (parameter == NULL) {
        return 0;
    }
    if (transformer_parameter_get_l(parameter) != 16 ||
        transformer_parameter_get_n(parameter) != 4 ||
        transformer_parameter_get_v(parameter) != 99 ||
        transformer_parameter_get_dk(parameter) != 4) {
        free_transformer_parameter(parameter);
        return 0;
    }
    if (transformer_parameter_get_input_size(parameter) != 2 ||
        transformer_parameter_get_output_size(parameter) != 1) {
        free_transformer_parameter(parameter);
        return 0;
    }
    if (transformer_parameter_get_input_hidden_layer(parameter, 0) != 64 ||
        transformer_parameter_get_output_hidden_layer(parameter, 0) != 128) {
        free_transformer_parameter(parameter);
        return 0;
    }
    if (transformer_parameter_get_gamma_input_value(parameter, 0) != 1.5 ||
        transformer_parameter_get_gamma_output_value(parameter, 0) != 2.5 ||
        transformer_parameter_get_beta_input_value(parameter, 0) != 3.5 ||
        transformer_parameter_get_beta_output_value(parameter, 0) != 4.5) {
        free_transformer_parameter(parameter);
        return 0;
    }
    if (transformer_parameter_get_input_activation_function(parameter, 1) != &function2 ||
        transformer_parameter_get_output_activation_function(parameter, 0) != &function2) {
        free_transformer_parameter(parameter);
        return 0;
    }
    free_transformer_parameter(parameter);
    return 1;
}

int main(void) {
    if (!test_recurrent_parameter()) {
        return 1;
    }
    if (!test_transformer_parameter()) {
        return 1;
    }
    return 0;
}
