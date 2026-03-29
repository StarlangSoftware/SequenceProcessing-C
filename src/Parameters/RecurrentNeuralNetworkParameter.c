#include "RecurrentNeuralNetworkParameter.h"

#include "ArrayList.h"
#include "Memory/Memory.h"

static Array_list_ptr clone_int_list(const Array_list* source) {
    int i;
    Array_list_ptr result = create_array_list();
    if (result == NULL) {
        return NULL;
    }
    if (source == NULL) {
        return result;
    }
    for (i = 0; i < source->size; i++) {
        array_list_add_int(result, array_list_get_int(source, i));
    }
    return result;
}

static Array_list_ptr clone_pointer_list(const Array_list* source) {
    int i;
    Array_list_ptr result = create_array_list();
    if (result == NULL) {
        return NULL;
    }
    if (source == NULL) {
        return result;
    }
    for (i = 0; i < source->size; i++) {
        array_list_add(result, array_list_get(source, i));
    }
    return result;
}

Recurrent_neural_network_parameter_ptr create_recurrent_neural_network_parameter(int seed,
                                                                                 int epoch,
                                                                                 Optimizer_ptr optimizer,
                                                                                 Initialization initialization,
                                                                                 Function* loss,
                                                                                 const Array_list* hidden_layers,
                                                                                 const Array_list* functions,
                                                                                 int class_label_size) {
    Recurrent_neural_network_parameter_ptr result = malloc_(sizeof(Recurrent_neural_network_parameter));
    if (result == NULL) {
        return NULL;
    }
    result->neural_network_parameter.seed = seed;
    result->neural_network_parameter.optimizer = optimizer;
    result->neural_network_parameter.epoch = epoch;
    result->neural_network_parameter.initialization = initialization;
    result->neural_network_parameter.dropout = 0.0;
    result->loss_function = loss;
    result->batch_size = 1;
    result->hidden_layers = clone_int_list(hidden_layers);
    if (result->hidden_layers == NULL) {
        free_(result);
        return NULL;
    }
    result->functions = clone_pointer_list(functions);
    if (result->functions == NULL) {
        free_array_list(result->hidden_layers, free_);
        free_(result);
        return NULL;
    }
    result->class_label_size = class_label_size;
    return result;
}

void free_recurrent_neural_network_parameter(Recurrent_neural_network_parameter_ptr parameter) {
    if (parameter == NULL) {
        return;
    }
    free_array_list(parameter->hidden_layers, free_);
    free_array_list(parameter->functions, NULL);
    free_(parameter);
}

int recurrent_neural_network_parameter_size(const Recurrent_neural_network_parameter* parameter) {
    if (parameter == NULL || parameter->hidden_layers == NULL) {
        return 0;
    }
    return parameter->hidden_layers->size;
}

int recurrent_neural_network_parameter_get_class_label_size(const Recurrent_neural_network_parameter* parameter) {
    if (parameter == NULL) {
        return 0;
    }
    return parameter->class_label_size;
}

Function* recurrent_neural_network_parameter_get_activation_function(const Recurrent_neural_network_parameter* parameter,
                                                                    int index) {
    if (parameter == NULL || parameter->functions == NULL) {
        return NULL;
    }
    return array_list_get(parameter->functions, index);
}

int recurrent_neural_network_parameter_get_hidden_layer(const Recurrent_neural_network_parameter* parameter, int index) {
    if (parameter == NULL || parameter->hidden_layers == NULL) {
        return 0;
    }
    return array_list_get_int(parameter->hidden_layers, index);
}
