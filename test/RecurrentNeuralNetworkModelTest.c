#include "Classification/RecurrentNeuralNetworkModel.h"

#include "ArrayList.h"
#include "Functions/Switch.h"
#include "Initialization/Initialization.h"
#include "Memory/Memory.h"
#include "Node/ComputationalNode.h"
#include "Optimizer/Optimizer.h"

static Tensor_ptr create_test_sequence_instance_1d(double a, double b, double c,
                                                   double d, double e, double f) {
    double* values = malloc_(6 * sizeof(double));
    int shape[1] = {6};
    values[0] = a;
    values[1] = b;
    values[2] = c;
    values[3] = d;
    values[4] = e;
    values[5] = f;
    return create_tensor3(values, shape, 1);
}

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

static int test_create_input_tensors(void) {
    Recurrent_neural_network_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Recurrent_neural_network_model_ptr model = create_recurrent_neural_network_model(parameter, 2);
    Tensor_ptr instance = create_test_sequence_instance_1d(1.0, 2.0, 0.0, 3.0, 4.0, 1.0);
    Array_list_ptr class_labels;
    Computational_node_ptr node0;
    Computational_node_ptr node1;
    Computational_node_ptr node2;
    int success;
    recurrent_neural_network_model_add_time_step_input(model);
    recurrent_neural_network_model_add_time_step_input(model);
    recurrent_neural_network_model_add_time_step_input(model);
    recurrent_neural_network_model_add_class_label_input(model);
    class_labels = recurrent_neural_network_model_create_input_tensors(model, instance);
    node0 = array_list_get(model->input_nodes, 0);
    node1 = array_list_get(model->input_nodes, 1);
    node2 = array_list_get(model->input_nodes, 2);
    success = class_labels != NULL &&
              class_labels->size == 3 &&
              array_list_get_int(class_labels, 0) == 0 &&
              array_list_get_int(class_labels, 1) == 1 &&
              array_list_get_int(class_labels, 2) == 0 &&
              ((Switch_function_ptr) array_list_get(model->switches, 0))->turn &&
              ((Switch_function_ptr) array_list_get(model->switches, 1))->turn &&
              !((Switch_function_ptr) array_list_get(model->switches, 2))->turn &&
              node0->value != NULL &&
              node1->value != NULL &&
              node2->value != NULL &&
              node0->value->shape[0] == 1 &&
              node0->value->shape[1] == 2 &&
              node0->value->data[0] == 1.0 &&
              node0->value->data[1] == 2.0 &&
              node1->value->data[0] == 3.0 &&
              node1->value->data[1] == 4.0 &&
              node2->value->data[0] == 0.0 &&
              node2->value->data[1] == 0.0;
    free_array_list(class_labels, free_);
    free_tensor(instance);
    free_recurrent_neural_network_model(model);
    free_recurrent_neural_network_parameter(parameter);
    free_(optimizer);
    return success;
}

static int test_find_time_step(void) {
    Recurrent_neural_network_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Recurrent_neural_network_model_ptr model = create_recurrent_neural_network_model(parameter, 2);
    Array_list_ptr train_set = create_array_list();
    Tensor_ptr instance1 = create_test_sequence_instance_1d(1.0, 2.0, 0.0, 3.0, 4.0, 1.0);
    double* values = malloc_(3 * sizeof(double));
    int shape[1] = {3};
    int success;
    values[0] = 5.0;
    values[1] = 6.0;
    values[2] = 2.0;
    array_list_add(train_set, create_tensor3(values, shape, 1));
    array_list_add(train_set, instance1);
    success = recurrent_neural_network_model_find_time_step(model, train_set) == 2;
    free_array_list(train_set, (void (*)(void*)) free_tensor);
    free_recurrent_neural_network_model(model);
    free_recurrent_neural_network_parameter(parameter);
    free_(optimizer);
    return success;
}

static int test_get_output_value(void) {
    double* values = malloc_(6 * sizeof(double));
    int shape[2] = {2, 3};
    Tensor_ptr output_tensor;
    Computational_node_ptr output_node;
    Array_list_ptr class_labels;
    int success;
    values[0] = 0.1;
    values[1] = 0.7;
    values[2] = 0.2;
    values[3] = 0.9;
    values[4] = 0.05;
    values[5] = 0.05;
    output_tensor = create_tensor3(values, shape, 2);
    output_node = create_computational_node(false, false, NULL, output_tensor);
    class_labels = recurrent_neural_network_model_get_output_value(output_node);
    success = class_labels != NULL &&
              class_labels->size == 2 &&
              array_list_get_double(class_labels, 0) == 1.0 &&
              array_list_get_double(class_labels, 1) == 0.0;
    free_array_list(class_labels, free_);
    free_computational_node(output_node);
    return success;
}

int main(void) {
    if (!test_create_input_tensors()) {
        return 1;
    }
    if (!test_find_time_step()) {
        return 1;
    }
    if (!test_get_output_value()) {
        return 1;
    }
    return 0;
}
