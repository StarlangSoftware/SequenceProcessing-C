#include "RecurrentNeuralNetworkModel.h"

#include "Functions/Switch.h"

#include "ArrayList.h"
#include "Memory/Memory.h"
#include "Node/ComputationalNode.h"

#include <float.h>
#include <stddef.h>

static void free_switch_entry(void* data) {
    free_switch_function((Switch_function_ptr) data);
}

Recurrent_neural_network_model_ptr create_recurrent_neural_network_model(Recurrent_neural_network_parameter_ptr parameters,
                                                                         int word_embedding_length) {
    Recurrent_neural_network_model_ptr result = malloc_(sizeof(Recurrent_neural_network_model));
    if (result == NULL) {
        return NULL;
    }
    result->parameters = parameters;
    result->word_embedding_length = word_embedding_length;
    result->input_nodes = create_array_list();
    result->switches = create_array_list();
    if (result->input_nodes == NULL || result->switches == NULL) {
        if (result->input_nodes != NULL) {
            free_array_list(result->input_nodes, NULL);
        }
        if (result->switches != NULL) {
            free_array_list(result->switches, NULL);
        }
        free_(result);
        return NULL;
    }
    return result;
}

void free_recurrent_neural_network_model(Recurrent_neural_network_model_ptr model) {
    if (model == NULL) {
        return;
    }
    if (model->input_nodes != NULL) {
        free_array_list(model->input_nodes, (void (*)(void*)) free_computational_node);
    }
    if (model->switches != NULL) {
        free_array_list(model->switches, free_switch_entry);
    }
    free_(model);
}

Computational_node_ptr recurrent_neural_network_model_add_time_step_input(Recurrent_neural_network_model_ptr model) {
    Computational_node_ptr input_node;
    Switch_function_ptr switch_function;
    if (model == NULL) {
        return NULL;
    }
    input_node = create_computational_node3(false, true);
    switch_function = create_switch_function();
    if (input_node == NULL || switch_function == NULL) {
        if (input_node != NULL) {
            free_computational_node(input_node);
        }
        if (switch_function != NULL) {
            free_switch_function(switch_function);
        }
        return NULL;
    }
    array_list_add(model->input_nodes, input_node);
    array_list_add(model->switches, switch_function);
    return input_node;
}

Computational_node_ptr recurrent_neural_network_model_add_class_label_input(Recurrent_neural_network_model_ptr model) {
    Computational_node_ptr class_label_node;
    if (model == NULL) {
        return NULL;
    }
    class_label_node = create_computational_node3(false, false);
    if (class_label_node == NULL) {
        return NULL;
    }
    array_list_add(model->input_nodes, class_label_node);
    return class_label_node;
}

Array_list_ptr recurrent_neural_network_model_create_input_tensors(Recurrent_neural_network_model_ptr model,
                                                                   const Tensor* instance) {
    Array_list_ptr class_labels;
    int time_step;
    int node_count;
    int i;
    int j = 0;
    if (model == NULL || instance == NULL || instance->dimensions != 1 || model->word_embedding_length < 0) {
        return NULL;
    }
    /*
     * Java assumes that the graph has already been built so that one trailing
     * class-label input node exists after all sequence input nodes.
     */
    node_count = model->input_nodes->size;
    if (node_count < 1 || model->switches->size != node_count - 1) {
        return NULL;
    }
    time_step = instance->shape[0] / (model->word_embedding_length + 1);
    class_labels = create_array_list();
    if (class_labels == NULL) {
        return NULL;
    }
    for (i = 0; i < node_count - 1; i++) {
        Tensor_ptr input_tensor;
        double* values = malloc_(model->word_embedding_length * sizeof(double));
        int shape[2] = {1, model->word_embedding_length};
        int k;
        if (values == NULL && model->word_embedding_length > 0) {
            free_array_list(class_labels, free_);
            return NULL;
        }
        if (i < time_step) {
            set_switch_turn((Switch_function_ptr) array_list_get(model->switches, i), true);
            for (k = 0; k < model->word_embedding_length; k++) {
                values[k] = instance->data[j];
                j++;
            }
            array_list_add_int(class_labels, (int) instance->data[j]);
            j++;
        } else {
            /*
             * Java still advances its flat index counters for padded time
             * steps, but does not read from the instance tensor in this path.
             */
            set_switch_turn((Switch_function_ptr) array_list_get(model->switches, i), false);
            for (k = 0; k < model->word_embedding_length; k++) {
                values[k] = 0.0;
                j++;
            }
            array_list_add_int(class_labels, 0);
            j++;
        }
        input_tensor = create_tensor3(values, shape, 2);
        if (input_tensor == NULL) {
            free_(values);
            free_array_list(class_labels, free_);
            return NULL;
        }
        set_node_value((Computational_node_ptr) array_list_get(model->input_nodes, i), input_tensor);
    }
    return class_labels;
}

int recurrent_neural_network_model_find_time_step(const Recurrent_neural_network_model* model,
                                                  const Array_list* train_set) {
    int time_step = -1;
    int i;
    if (model == NULL || train_set == NULL || model->word_embedding_length < 0) {
        return -1;
    }
    for (i = 0; i < train_set->size; i++) {
        const Tensor* tensor = array_list_get(train_set, i);
        int current_time_step;
        if (tensor == NULL || tensor->dimensions < 1) {
            return -1;
        }
        current_time_step = tensor->shape[0] / (model->word_embedding_length + 1);
        if (time_step < current_time_step) {
            time_step = current_time_step;
        }
    }
    return time_step;
}

Array_list_ptr recurrent_neural_network_model_get_output_value(const Computational_node* output_node) {
    Array_list_ptr class_labels;
    int i, j;
    if (output_node == NULL || output_node->value == NULL || output_node->value->dimensions != 2) {
        return NULL;
    }
    class_labels = create_array_list();
    if (class_labels == NULL) {
        return NULL;
    }
    for (i = 0; i < output_node->value->shape[0]; i++) {
        int index = -1;
        /*
         * Intentionally mirrors the Java source, which starts from
         * Double.MIN_VALUE rather than negative infinity.
         */
        double max = DBL_MIN;
        for (j = 0; j < output_node->value->shape[1]; j++) {
            double value = output_node->value->data[i * output_node->value->shape[1] + j];
            if (max < value) {
                max = value;
                index = j;
            }
        }
        array_list_add_double(class_labels, (double) index);
    }
    return class_labels;
}
