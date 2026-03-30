#include "RecurrentNeuralNetworkModel.h"

#include "JavaRandomCompat.h"
#include "RecurrentModelGraphBridge.h"
#include "Functions/Switch.h"

#include "ArrayList.h"
#include "Initialization/Initialization.h"
#include "Memory/Memory.h"
#include "Node/ComputationalNode.h"
#include "Node/MultiplicationNode.h"
#include "Optimizer/Optimizer.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

static void free_switch_entry(void* data) {
    free_switch_function((Switch_function_ptr) data);
}

static Array_list_ptr recurrent_output_extractor(const Computational_node* output_node) {
    return recurrent_neural_network_model_get_output_value(output_node);
}

static Tensor_ptr create_class_label_tensor(const Array_list* class_labels, int class_label_size) {
    double* values;
    int shape[2];
    int i, j;
    if (class_labels == NULL || class_label_size <= 0) {
        return NULL;
    }
    values = malloc_(class_labels->size * class_label_size * sizeof(double));
    if (values == NULL && class_labels->size * class_label_size > 0) {
        return NULL;
    }
    for (i = 0; i < class_labels->size; i++) {
        int class_label = array_list_get_int(class_labels, i);
        for (j = 0; j < class_label_size; j++) {
            values[(i * class_label_size) + j] = (j == class_label) ? 1.0 : 0.0;
        }
    }
    shape[0] = class_labels->size;
    shape[1] = class_label_size;
    return create_tensor3(values, shape, 2);
}

static int* create_class_label_index_array(const Array_list* class_labels) {
    int* result;
    int i;
    if (class_labels == NULL) {
        return NULL;
    }
    result = malloc_(class_labels->size * sizeof(int));
    if (result == NULL && class_labels->size > 0) {
        return NULL;
    }
    for (i = 0; i < class_labels->size; i++) {
        result[i] = array_list_get_int(class_labels, i);
    }
    return result;
}

static bool shuffle_train_set_like_java(Array_list_ptr train_set, Java_random_compat_ptr random) {
    int j;
    if (train_set == NULL || random == NULL) {
        return false;
    }
    for (j = 0; j < train_set->size; j++) {
        int i1;
        int i2;
        if (!java_random_compat_shuffle_pair_indices(random, train_set->size, &i1, &i2)) {
            return false;
        }
        array_list_swap(train_set, i1, i2);
    }
    return true;
}

Recurrent_neural_network_model_ptr create_recurrent_neural_network_model(Recurrent_neural_network_parameter_ptr parameters,
                                                                         int word_embedding_length) {
    Recurrent_neural_network_model_ptr result = malloc_(sizeof(Recurrent_neural_network_model));
    if (result == NULL) {
        return NULL;
    }
    result->parameters = parameters;
    result->word_embedding_length = word_embedding_length;
    result->graph_bridge = create_recurrent_model_graph_bridge(recurrent_output_extractor);
    result->input_nodes = recurrent_model_graph_bridge_get_input_nodes(result->graph_bridge);
    result->switches = create_array_list();
    result->graph_initialized = false;
    if (result->graph_bridge == NULL || result->input_nodes == NULL || result->switches == NULL) {
        if (result->graph_bridge != NULL) {
            free_recurrent_model_graph_bridge(result->graph_bridge);
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
    if (model->switches != NULL) {
        free_array_list(model->switches, free_switch_entry);
    }
    if (model->graph_bridge != NULL) {
        free_recurrent_model_graph_bridge(model->graph_bridge);
    }
    free_(model);
}

Computational_node_ptr recurrent_neural_network_model_add_time_step_input(Recurrent_neural_network_model_ptr model) {
    Computational_node_ptr input_node;
    Switch_function_ptr switch_function;
    if (model == NULL) {
        return NULL;
    }
    input_node = recurrent_model_graph_bridge_add_input_node(model->graph_bridge, false, true);
    switch_function = create_switch_function();
    if (input_node == NULL || switch_function == NULL) {
        if (switch_function != NULL) {
            free_switch_function(switch_function);
        }
        return NULL;
    }
    array_list_add(model->switches, switch_function);
    return input_node;
}

Computational_node_ptr recurrent_neural_network_model_add_class_label_input(Recurrent_neural_network_model_ptr model) {
    Computational_node_ptr class_label_node;
    if (model == NULL) {
        return NULL;
    }
    class_label_node = recurrent_model_graph_bridge_add_input_node(model->graph_bridge, false, false);
    return class_label_node;
}

Computational_node_ptr recurrent_neural_network_model_add_edge(Recurrent_neural_network_model_ptr model,
                                                               Computational_node_ptr first,
                                                               Function* function,
                                                               bool is_biased) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_add_function_edge(model->graph_bridge, first, function, is_biased);
}

Computational_node_ptr recurrent_neural_network_model_add_multiplication_edge(Recurrent_neural_network_model_ptr model,
                                                                              Computational_node_ptr first,
                                                                              Multiplication_node_ptr second,
                                                                              bool is_biased) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_add_multiplication_edge(model->graph_bridge, first, second, is_biased);
}

Computational_node_ptr recurrent_neural_network_model_add_hadamard_edge(Recurrent_neural_network_model_ptr model,
                                                                        Computational_node_ptr first,
                                                                        Computational_node_ptr second,
                                                                        bool is_biased) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_add_hadamard_edge(model->graph_bridge, first, second, is_biased);
}

Computational_node_ptr recurrent_neural_network_model_add_addition_edge(Recurrent_neural_network_model_ptr model,
                                                                        Computational_node_ptr first,
                                                                        Computational_node_ptr second,
                                                                        bool is_biased) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_add_addition_edge(model->graph_bridge, first, second, is_biased);
}

Concatenated_node_ptr recurrent_neural_network_model_concat_edges(Recurrent_neural_network_model_ptr model,
                                                                  Array_list_ptr nodes,
                                                                  int dimension) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_concat_edges(model->graph_bridge, nodes, dimension);
}

void recurrent_neural_network_model_set_output_node(Recurrent_neural_network_model_ptr model,
                                                    Computational_node_ptr output_node) {
    if (model == NULL) {
        return;
    }
    recurrent_model_graph_bridge_set_output_node(model->graph_bridge, output_node);
}

Array_list_ptr recurrent_neural_network_model_forward(Recurrent_neural_network_model_ptr model) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_forward(model->graph_bridge);
}

Array_list_ptr recurrent_neural_network_model_predict(Recurrent_neural_network_model_ptr model) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_predict(model->graph_bridge);
}

Multiplication_node_ptr recurrent_neural_network_model_create_weight_node(Recurrent_neural_network_model_ptr model,
                                                                          int row,
                                                                          int column,
                                                                          Java_random_compat_ptr random) {
    double* values;
    int shape[2];
    int i;
    if (model == NULL || model->parameters == NULL || random == NULL || row <= 0 || column <= 0) {
        return NULL;
    }
    values = malloc_(row * column * sizeof(double));
    if (values == NULL) {
        return NULL;
    }
    for (i = 0; i < row * column; i++) {
        double random_value = java_random_compat_next_double(random);
        switch (model->parameters->neural_network_parameter.initialization) {
            case He:
                values[i] = ((sqrt(6.0 / column) + sqrt(6.0 / row)) * random_value) - sqrt(6.0 / row);
                break;
            case Uniform:
                values[i] = (2.0 * random_value - 1.0) * sqrt(6.0 / (row + column));
                break;
            case Random:
            default:
                values[i] = -0.01 + (0.02 * random_value);
                break;
        }
    }
    shape[0] = row;
    shape[1] = column;
    return create_multiplication_node5(create_tensor3(values, shape, 2));
}

bool recurrent_neural_network_model_train_with_random(Recurrent_neural_network_model_ptr model,
                                                      Array_list_ptr train_set,
                                                      Java_random_compat_ptr random) {
    int epoch;
    Computational_node_ptr class_label_node;
    Optimizer_ptr optimizer;
    if (model == NULL || model->parameters == NULL || train_set == NULL || random == NULL) {
        return false;
    }
    if (model->input_nodes == NULL || model->input_nodes->size < 1) {
        return false;
    }
    class_label_node = array_list_get(model->input_nodes, model->input_nodes->size - 1);
    optimizer = model->parameters->neural_network_parameter.optimizer;
    if (class_label_node == NULL || optimizer == NULL) {
        return false;
    }
    for (epoch = 0; epoch < model->parameters->neural_network_parameter.epoch; epoch++) {
        int instance_index;
        if (!shuffle_train_set_like_java(train_set, random)) {
            return false;
        }
        for (instance_index = 0; instance_index < train_set->size; instance_index++) {
            Tensor_ptr instance = array_list_get(train_set, instance_index);
            Array_list_ptr class_labels = recurrent_neural_network_model_create_input_tensors(model, instance);
            Tensor_ptr class_label_tensor;
            int* class_label_index;
            Array_list_ptr predictions;
            if (class_labels == NULL) {
                return false;
            }
            class_label_tensor = create_class_label_tensor(
                    class_labels,
                    recurrent_neural_network_parameter_get_class_label_size(model->parameters));
            if (class_label_tensor == NULL) {
                free_array_list(class_labels, free_);
                return false;
            }
            set_node_value(class_label_node, class_label_tensor);
            predictions = recurrent_neural_network_model_forward(model);
            if (predictions == NULL) {
                free_array_list(class_labels, free_);
                return false;
            }
            free_array_list(predictions, free_);
            class_label_index = create_class_label_index_array(class_labels);
            if (class_label_index == NULL && class_labels->size > 0) {
                free_array_list(class_labels, free_);
                return false;
            }
            recurrent_model_graph_bridge_back_propagation(model->graph_bridge, optimizer, class_label_index);
            free_(class_label_index);
            free_array_list(class_labels, free_);
        }
        set_learning_rate(optimizer);
    }
    return true;
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
