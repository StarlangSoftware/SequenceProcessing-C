#include "Transformer.h"

#include "ArrayList.h"
#include "BorrowedFunctionProxy.h"
#include "Dictionary/Dictionary.h"
#include "Dictionary/VectorizedDictionary.h"
#include "Dictionary/VectorizedWord.h"
#include "Function/Negation.h"
#include "Function/SoftMax.h"
#include "JavaRandomCompat.h"
#include "Memory/Memory.h"
#include "Functions/MultiplyByConstant.h"
#include "Node/ComputationalNode.h"
#include "Node/MultiplicationNode.h"
#include "RecurrentModelGraphBridge.h"
#include "Functions/Inverse.h"
#include "Functions/Mask.h"
#include "Functions/Mean.h"
#include "Functions/SquareRoot.h"
#include "Functions/Transpose.h"
#include "Functions/Variance.h"
#include "Vector.h"
#include "Optimizer/Optimizer.h"

#include <float.h>
#include <math.h>
#include <string.h>

struct transformer_model {
    /*
     * Borrowed parameter object. The current shell does not own or free it.
     */
    Transformer_parameter_ptr parameters;

    /*
     * Borrowed dictionary reference, matching the Java constructor field.
     */
    Vectorized_dictionary_ptr dictionary;

    /*
     * Owned local graph bridge reused for staged Transformer input/output and
     * graph-body state.
     */
    Recurrent_model_graph_bridge_ptr graph_bridge;

    /*
     * Borrowed alias of bridge-managed input nodes.
     */
    Array_list_ptr input_nodes;

    int start_index;
    int end_index;

    /*
     * Lifecycle lock for the staged graph-backed path. This flag is set once
     * graph construction starts claiming this instance. It is not a claim of
     * full Java loss-node forward parity or `test(...)` parity.
     */
    bool graph_initialized;
};

struct transformer_packed_inputs {
    /*
     * Owned helper outputs. Getter functions return borrowed views into this
     * owned pack.
     */
    Tensor_ptr encoder_input;
    Tensor_ptr decoder_input;
    Array_list_ptr class_labels;
};

static int find_token_index(const Vectorized_dictionary* dictionary, const char* token) {
    int i;
    if (dictionary == NULL || token == NULL) {
        return -1;
    }
    for (i = 0; i < size((const Dictionary*) dictionary); i++) {
        const Vectorized_word* word = get_word_with_index((const Dictionary*) dictionary, i);
        if (word != NULL && word->word.name != NULL && strcmp(word->word.name, token) == 0) {
            return i;
        }
    }
    return -1;
}

static Array_list_ptr transformer_output_extractor(const Computational_node* output_node) {
    return transformer_model_get_output_value(output_node);
}

static Tensor_ptr create_tensor_from_double_list(const Array_list* values, int row_count, int column_count) {
    double* data;
    int shape[2];
    int i;
    if (values == NULL || row_count < 0 || column_count <= 0) {
        return NULL;
    }
    if (row_count * column_count != values->size) {
        return NULL;
    }
    data = malloc_(values->size * sizeof(double));
    if (data == NULL && values->size > 0) {
        return NULL;
    }
    for (i = 0; i < values->size; i++) {
        data[i] = array_list_get_double(values, i);
    }
    shape[0] = row_count;
    shape[1] = column_count;
    return create_tensor3(data, shape, 2);
}

static void free_weight_list(Array_list_ptr list) {
    if (list != NULL) {
        free_array_list(list, (void (*)(void*)) free_multiplication_node);
    }
}

static Computational_node_ptr add_owned_function_edge(Transformer_model_ptr model,
                                                      Computational_node_ptr first,
                                                      Function* function,
                                                      bool is_biased) {
    Computational_node_ptr result;
    if (model == NULL || function == NULL) {
        return NULL;
    }
    result = recurrent_model_graph_bridge_add_function_edge(model->graph_bridge, first, function, is_biased);
    if (result == NULL) {
        free_(function);
    }
    return result;
}

static Computational_node_ptr add_borrowed_function_edge(Transformer_model_ptr model,
                                                         Computational_node_ptr first,
                                                         const Function* function,
                                                         bool is_biased) {
    Function* proxy = create_borrowed_function_proxy(function);
    Computational_node_ptr result;
    if (model == NULL || proxy == NULL) {
        return NULL;
    }
    result = recurrent_model_graph_bridge_add_function_edge(model->graph_bridge, first, proxy, is_biased);
    if (result == NULL) {
        free_borrowed_function_proxy(proxy);
    }
    return result;
}

static Computational_node_ptr add_weight_edge(Transformer_model_ptr model,
                                              Computational_node_ptr first,
                                              Multiplication_node_ptr second,
                                              bool is_biased) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_add_multiplication_edge(model->graph_bridge, first, second, is_biased);
}

static Computational_node_ptr add_hadamard_edge(Transformer_model_ptr model,
                                                Computational_node_ptr first,
                                                Computational_node_ptr second,
                                                bool is_biased) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_add_hadamard_edge(model->graph_bridge, first, second, is_biased);
}

static Computational_node_ptr add_addition_edge(Transformer_model_ptr model,
                                                Computational_node_ptr first,
                                                Computational_node_ptr second,
                                                bool is_biased) {
    if (model == NULL) {
        return NULL;
    }
    return recurrent_model_graph_bridge_add_addition_edge(model->graph_bridge, first, second, is_biased);
}

extern Computational_node_ptr add_edge_with_hadamard(Computational_graph_ptr graph,
                                                     Computational_node_ptr first,
                                                     Computational_node_ptr second,
                                                     bool is_biased,
                                                     bool is_hadamard);

static Computational_node_ptr add_matrix_multiplication_edge(Transformer_model_ptr model,
                                                             Computational_node_ptr first,
                                                             Computational_node_ptr second,
                                                             bool is_biased) {
    if (model == NULL || model->graph_bridge == NULL || first == NULL || second == NULL) {
        return NULL;
    }
    return add_edge_with_hadamard(recurrent_model_graph_bridge_get_graph(model->graph_bridge),
                                  first,
                                  second,
                                  is_biased,
                                  false);
}

static Multiplication_node_ptr create_transformer_weight_node(Transformer_model_ptr model,
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

static Computational_node_ptr layer_normalization(Transformer_model_ptr model,
                                                  Computational_node_ptr input,
                                                  const Transformer_parameter* parameter,
                                                  bool is_input,
                                                  int ln_size[4]) {
    double* gamma_data;
    double* beta_data;
    int shape[2];
    int j;
    Computational_node_ptr input_c1_mean;
    Computational_node_ptr mean1_minus;
    Computational_node_ptr input_c1_mean1_minus;
    Computational_node_ptr variance1;
    Computational_node_ptr root_variance1;
    Computational_node_ptr inverse_root_variance1;
    Computational_node_ptr ln_value1;
    Computational_node_ptr gamma_node;
    Computational_node_ptr ln_value1_gamma;
    Computational_node_ptr beta_node;
    if (model == NULL || input == NULL || parameter == NULL) {
        return NULL;
    }
    input_c1_mean = add_owned_function_edge(model, input, (Function*) create_mean_function(), false);
    mean1_minus = add_owned_function_edge(model, input_c1_mean, (Function*) create_negation(), false);
    input_c1_mean1_minus = add_addition_edge(model, input, mean1_minus, false);
    variance1 = add_owned_function_edge(model, input_c1_mean1_minus, (Function*) create_variance_function(), false);
    root_variance1 = add_owned_function_edge(model, variance1,
                                             (Function*) create_square_root_function(transformer_parameter_get_epsilon(parameter)),
                                             false);
    inverse_root_variance1 = add_owned_function_edge(model, root_variance1, (Function*) create_inverse_function(), false);
    ln_value1 = add_hadamard_edge(model, input_c1_mean1_minus, inverse_root_variance1, false);
    if (input_c1_mean == NULL || mean1_minus == NULL || input_c1_mean1_minus == NULL ||
        variance1 == NULL || root_variance1 == NULL || inverse_root_variance1 == NULL || ln_value1 == NULL) {
        return NULL;
    }
    gamma_data = malloc_(transformer_parameter_get_l(parameter) * sizeof(double));
    beta_data = malloc_(transformer_parameter_get_l(parameter) * sizeof(double));
    if (gamma_data == NULL || beta_data == NULL) {
        if (gamma_data != NULL) {
            free_(gamma_data);
        }
        if (beta_data != NULL) {
            free_(beta_data);
        }
        return NULL;
    }
    for (j = 0; j < transformer_parameter_get_l(parameter); j++) {
        if (is_input) {
            gamma_data[j] = transformer_parameter_get_gamma_input_value(parameter, ln_size[0]);
            beta_data[j] = transformer_parameter_get_beta_input_value(parameter, ln_size[2]);
        } else {
            gamma_data[j] = transformer_parameter_get_gamma_output_value(parameter, ln_size[1]);
            beta_data[j] = transformer_parameter_get_beta_output_value(parameter, ln_size[3]);
        }
    }
    if (is_input) {
        ln_size[0]++;
        ln_size[2]++;
    } else {
        ln_size[1]++;
        ln_size[3]++;
    }
    shape[0] = 1;
    shape[1] = transformer_parameter_get_l(parameter);
    gamma_node = (Computational_node_ptr) create_multiplication_node3(true, false, create_tensor3(gamma_data, shape, 2), true);
    beta_node = create_computational_node(true, false, NULL, create_tensor3(beta_data, shape, 2));
    ln_value1_gamma = add_weight_edge(model, ln_value1, (Multiplication_node_ptr) gamma_node, false);
    if (gamma_node == NULL || beta_node == NULL || ln_value1_gamma == NULL) {
        if (gamma_node != NULL) {
            free_multiplication_node((Multiplication_node_ptr) gamma_node);
        }
        if (beta_node != NULL) {
            free_computational_node(beta_node);
        }
        return NULL;
    }
    return add_addition_edge(model, ln_value1_gamma, beta_node, false);
}

static Array_list_ptr multi_head_attention(Transformer_model_ptr model,
                                           Computational_node_ptr input,
                                           const Transformer_parameter* parameter,
                                           bool is_masked,
                                           Java_random_compat_ptr random) {
    Array_list_ptr nodes;
    int i;
    if (model == NULL || input == NULL || parameter == NULL || random == NULL) {
        return NULL;
    }
    nodes = create_array_list();
    if (nodes == NULL) {
        return NULL;
    }
    for (i = 0; i < transformer_parameter_get_n(parameter); i++) {
        Multiplication_node_ptr wk = create_transformer_weight_node(model, transformer_parameter_get_l(parameter),
                                                                    transformer_parameter_get_dk(parameter), random);
        Multiplication_node_ptr wq = create_transformer_weight_node(model, transformer_parameter_get_l(parameter),
                                                                    transformer_parameter_get_dk(parameter), random);
        Multiplication_node_ptr wv = create_transformer_weight_node(model, transformer_parameter_get_l(parameter),
                                                                    transformer_parameter_get_dk(parameter), random);
        Computational_node_ptr k;
        Computational_node_ptr q;
        Computational_node_ptr v;
        Computational_node_ptr k_transpose;
        Computational_node_ptr qk;
        Computational_node_ptr qk_dk;
        Computational_node_ptr s_qk_dk;
        Computational_node_ptr attention;
        if (wk == NULL || wq == NULL || wv == NULL) {
            if (wk != NULL) free_multiplication_node(wk);
            if (wq != NULL) free_multiplication_node(wq);
            if (wv != NULL) free_multiplication_node(wv);
            free_array_list(nodes, NULL);
            return NULL;
        }
        k = add_weight_edge(model, input, wk, false);
        q = add_weight_edge(model, input, wq, false);
        v = add_weight_edge(model, input, wv, false);
        k_transpose = add_owned_function_edge(model, k, (Function*) create_transpose_function(), false);
        qk = add_matrix_multiplication_edge(model, q, k_transpose, false);
        qk_dk = add_owned_function_edge(model, qk,
                                        (Function*) create_multiply_by_constant(1.0 / sqrt(transformer_parameter_get_dk(parameter))),
                                        false);
        if (is_masked) {
            Computational_node_ptr masked = add_owned_function_edge(model, qk_dk, (Function*) create_mask_function(), false);
            s_qk_dk = add_owned_function_edge(model, masked, (Function*) create_softmax(), false);
        } else {
            s_qk_dk = add_owned_function_edge(model, qk_dk, (Function*) create_softmax(), false);
        }
        attention = add_matrix_multiplication_edge(model, s_qk_dk, v, false);
        if (k == NULL || q == NULL || v == NULL || k_transpose == NULL || qk == NULL ||
            qk_dk == NULL || s_qk_dk == NULL || attention == NULL) {
            free_array_list(nodes, NULL);
            return NULL;
        }
        array_list_add(nodes, attention);
    }
    return nodes;
}

static Computational_node_ptr feedforward_neural_network(Transformer_model_ptr model,
                                                         Computational_node_ptr current,
                                                         int current_layer_size,
                                                         const Transformer_parameter* parameter,
                                                         Java_random_compat_ptr random,
                                                         bool is_input) {
    int size;
    int i;
    if (model == NULL || current == NULL || parameter == NULL || random == NULL) {
        return NULL;
    }
    size = is_input ? transformer_parameter_get_input_size(parameter) : transformer_parameter_get_output_size(parameter);
    for (i = 0; i < size; i++) {
        int hidden_size = is_input
                ? transformer_parameter_get_input_hidden_layer(parameter, i)
                : transformer_parameter_get_output_hidden_layer(parameter, i);
        Function* activation = is_input
                ? transformer_parameter_get_input_activation_function(parameter, i)
                : transformer_parameter_get_output_activation_function(parameter, i);
        Multiplication_node_ptr hidden_weight = create_transformer_weight_node(model, current_layer_size, hidden_size, random);
        Computational_node_ptr hidden_layer;
        if (hidden_weight == NULL) {
            return NULL;
        }
        hidden_layer = add_weight_edge(model, current, hidden_weight, false);
        current = add_borrowed_function_edge(model, hidden_layer, activation, true);
        if (hidden_layer == NULL || current == NULL) {
            return NULL;
        }
        current_layer_size = hidden_size + 1;
    }
    {
        Multiplication_node_ptr output_weight = create_transformer_weight_node(model,
                                                                               current_layer_size,
                                                                               transformer_parameter_get_l(parameter),
                                                                               random);
        Computational_node_ptr output_layer;
        if (output_weight == NULL) {
            return NULL;
        }
        output_layer = add_weight_edge(model, current, output_weight, false);
        if (output_layer == NULL) {
            return NULL;
        }
        return add_owned_function_edge(model, output_layer, (Function*) create_softmax(), false);
    }
}

static Tensor_ptr create_transformer_class_label_tensor(const Array_list* class_labels, int vocabulary_size) {
    double* values;
    int shape[2];
    int i, j;
    if (class_labels == NULL || vocabulary_size <= 0) {
        return NULL;
    }
    values = malloc_(class_labels->size * vocabulary_size * sizeof(double));
    if (values == NULL && class_labels->size * vocabulary_size > 0) {
        return NULL;
    }
    for (i = 0; i < class_labels->size; i++) {
        int class_label = array_list_get_int(class_labels, i);
        for (j = 0; j < vocabulary_size; j++) {
            values[(i * vocabulary_size) + j] = (j == class_label) ? 1.0 : 0.0;
        }
    }
    shape[0] = class_labels->size;
    shape[1] = vocabulary_size;
    return create_tensor3(values, shape, 2);
}

static int* create_transformer_class_label_index_array(const Array_list* class_labels) {
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

Transformer_model_ptr create_transformer_model(Transformer_parameter_ptr parameters,
                                               Vectorized_dictionary_ptr dictionary) {
    Transformer_model_ptr result = malloc_(sizeof(Transformer_model));
    if (result == NULL) {
        return NULL;
    }
    result->parameters = parameters;
    result->dictionary = dictionary;
    result->graph_bridge = create_recurrent_model_graph_bridge(transformer_output_extractor);
    result->input_nodes = recurrent_model_graph_bridge_get_input_nodes(result->graph_bridge);
    result->start_index = find_token_index(dictionary, "<S>");
    result->end_index = find_token_index(dictionary, "</S>");
    result->graph_initialized = false;
    if (result->graph_bridge == NULL || result->input_nodes == NULL) {
        if (result->graph_bridge != NULL) {
            free_recurrent_model_graph_bridge(result->graph_bridge);
        }
        free_(result);
        return NULL;
    }
    return result;
}

void free_transformer_model(Transformer_model_ptr model) {
    if (model == NULL) {
        return;
    }
    if (model->graph_bridge != NULL) {
        free_recurrent_model_graph_bridge(model->graph_bridge);
    }
    free_(model);
}

Vectorized_dictionary_ptr transformer_model_get_dictionary(const Transformer_model* model) {
    if (model == NULL) {
        return NULL;
    }
    return model->dictionary;
}

int transformer_model_get_start_index(const Transformer_model* model) {
    if (model == NULL) {
        return -1;
    }
    return model->start_index;
}

int transformer_model_get_end_index(const Transformer_model* model) {
    if (model == NULL) {
        return -1;
    }
    return model->end_index;
}

Array_list_ptr transformer_model_get_input_nodes(const Transformer_model* model) {
    if (model == NULL) {
        return NULL;
    }
    return model->input_nodes;
}

bool transformer_model_is_graph_initialized(const Transformer_model* model) {
    if (model == NULL) {
        return false;
    }
    return model->graph_initialized;
}

Tensor_ptr transformer_model_positional_encoding(const Transformer_model* model,
                                                 const Tensor* tensor,
                                                 int word_embedding_length) {
    double* values;
    int shape[2];
    int i;
    int j;
    (void) model;
    if (tensor == NULL || tensor->dimensions != 2 || word_embedding_length <= 0) {
        return NULL;
    }
    values = malloc_(tensor->total_elements * sizeof(double));
    if (values == NULL && tensor->total_elements > 0) {
        return NULL;
    }
    for (i = 0; i < tensor->shape[0]; i++) {
        for (j = 0; j < tensor->shape[1]; j++) {
            double value = tensor->data[(i * tensor->shape[1]) + j];
            if (j % 2 == 0) {
                values[(i * tensor->shape[1]) + j] =
                        value + sin((i + 1.0) / pow(10000.0, (j + 0.0) / word_embedding_length));
            } else {
                values[(i * tensor->shape[1]) + j] =
                        value + cos((i + 1.0) / pow(10000.0, (j - 1.0) / word_embedding_length));
            }
        }
    }
    shape[0] = tensor->shape[0];
    shape[1] = tensor->shape[1];
    return create_tensor3(values, shape, 2);
}

Transformer_packed_inputs_ptr transformer_model_create_packed_inputs(const Transformer_model* model,
                                                                    const Tensor* instance,
                                                                    int word_embedding_length) {
    Transformer_packed_inputs_ptr result;
    Array_list_ptr encoder_values;
    Array_list_ptr decoder_values;
    int i;
    bool is_output = false;
    int current_length = 0;
    if (model == NULL || instance == NULL || instance->dimensions != 1 || word_embedding_length <= 0) {
        return NULL;
    }
    result = malloc_(sizeof(Transformer_packed_inputs));
    encoder_values = create_array_list();
    decoder_values = create_array_list();
    if (result == NULL || encoder_values == NULL || decoder_values == NULL) {
        if (encoder_values != NULL) {
            free_array_list(encoder_values, free_);
        }
        if (decoder_values != NULL) {
            free_array_list(decoder_values, free_);
        }
        if (result != NULL) {
            free_(result);
        }
        return NULL;
    }
    result->encoder_input = NULL;
    result->decoder_input = NULL;
    result->class_labels = create_array_list();
    if (result->class_labels == NULL) {
        free_array_list(encoder_values, free_);
        free_array_list(decoder_values, free_);
        free_(result);
        return NULL;
    }
    for (i = 0; i < instance->shape[0]; i++) {
        double value = instance->data[i];
        if (value == DBL_MAX) {
            Tensor_ptr encoder_tensor = create_tensor_from_double_list(
                    encoder_values,
                    current_length / word_embedding_length,
                    word_embedding_length);
            if (encoder_tensor == NULL) {
                free_array_list(encoder_values, free_);
                free_array_list(decoder_values, free_);
                free_transformer_packed_inputs(result);
                return NULL;
            }
            result->encoder_input = transformer_model_positional_encoding(model, encoder_tensor, word_embedding_length);
            free_tensor(encoder_tensor);
            if (result->encoder_input == NULL) {
                free_array_list(encoder_values, free_);
                free_array_list(decoder_values, free_);
                free_transformer_packed_inputs(result);
                return NULL;
            }
            current_length = 0;
            free_array_list(encoder_values, free_);
            encoder_values = create_array_list();
            if (encoder_values == NULL) {
                free_array_list(decoder_values, free_);
                free_transformer_packed_inputs(result);
                return NULL;
            }
            is_output = true;
        } else if (is_output) {
            if ((current_length + 1) % (word_embedding_length + 1) == 0) {
                array_list_add_int(result->class_labels, (int) value);
            } else {
                array_list_add_double(decoder_values, value);
            }
            current_length++;
        } else {
            array_list_add_double(encoder_values, value);
            current_length++;
        }
    }
    if (result->encoder_input == NULL) {
        free_array_list(encoder_values, free_);
        free_array_list(decoder_values, free_);
        free_transformer_packed_inputs(result);
        return NULL;
    }
    {
        Tensor_ptr decoder_tensor = create_tensor_from_double_list(
                decoder_values,
                decoder_values->size / word_embedding_length,
                word_embedding_length);
        if (decoder_tensor == NULL) {
            free_array_list(encoder_values, free_);
            free_array_list(decoder_values, free_);
            free_transformer_packed_inputs(result);
            return NULL;
        }
        result->decoder_input = transformer_model_positional_encoding(model, decoder_tensor, word_embedding_length);
        free_tensor(decoder_tensor);
        if (result->decoder_input == NULL) {
            free_array_list(encoder_values, free_);
            free_array_list(decoder_values, free_);
            free_transformer_packed_inputs(result);
            return NULL;
        }
    }
    free_array_list(encoder_values, free_);
    free_array_list(decoder_values, free_);
    return result;
}

void free_transformer_packed_inputs(Transformer_packed_inputs_ptr packed_inputs) {
    if (packed_inputs == NULL) {
        return;
    }
    if (packed_inputs->encoder_input != NULL) {
        free_tensor(packed_inputs->encoder_input);
    }
    if (packed_inputs->decoder_input != NULL) {
        free_tensor(packed_inputs->decoder_input);
    }
    if (packed_inputs->class_labels != NULL) {
        free_array_list(packed_inputs->class_labels, free_);
    }
    free_(packed_inputs);
}

const Tensor* transformer_packed_inputs_get_encoder_input(const Transformer_packed_inputs* packed_inputs) {
    if (packed_inputs == NULL) {
        return NULL;
    }
    return packed_inputs->encoder_input;
}

const Tensor* transformer_packed_inputs_get_decoder_input(const Transformer_packed_inputs* packed_inputs) {
    if (packed_inputs == NULL) {
        return NULL;
    }
    return packed_inputs->decoder_input;
}

Array_list_ptr transformer_packed_inputs_get_class_labels(const Transformer_packed_inputs* packed_inputs) {
    if (packed_inputs == NULL) {
        return NULL;
    }
    return packed_inputs->class_labels;
}

bool transformer_model_set_input_node(const Transformer_model* model,
                                      int bound,
                                      const Vector* vector,
                                      Computational_node_ptr node) {
    double* data;
    int shape[2];
    int old_size = 0;
    int i;
    if (model == NULL || vector == NULL || node == NULL || bound <= 0 || vector->size <= 0) {
        return false;
    }
    if (node->value != NULL) {
        old_size = node->value->total_elements;
    }
    data = malloc_((old_size + vector->size) * sizeof(double));
    if (data == NULL) {
        return false;
    }
    for (i = 0; i < old_size; i++) {
        data[i] = node->value->data[i];
    }
    for (i = 0; i < vector->size; i++) {
        if (i % 2 == 0) {
            data[old_size + i] =
                    get_value(vector, i) + sin((bound + 0.0) / pow(10000.0, (i + 0.0) / vector->size));
        } else {
            data[old_size + i] =
                    get_value(vector, i) + cos((bound + 0.0) / pow(10000.0, (i - 1.0) / vector->size));
        }
    }
    shape[0] = bound;
    shape[1] = vector->size;
    set_node_value(node, create_tensor3(data, shape, 2));
    return true;
}

bool transformer_model_initialize_graph_inputs(Transformer_model_ptr model) {
    if (model == NULL || model->graph_bridge == NULL || model->input_nodes == NULL) {
        return false;
    }
    if (model->input_nodes->size == 0) {
        Computational_node_ptr encoder_input =
                recurrent_model_graph_bridge_add_input_node(model->graph_bridge, false, true);
        Computational_node_ptr decoder_input =
                recurrent_model_graph_bridge_add_input_node(model->graph_bridge, false, true);
        return encoder_input != NULL && decoder_input != NULL;
    }
    return model->input_nodes->size == 2 || model->input_nodes->size == 3;
}

bool transformer_model_apply_packed_inputs(Transformer_model_ptr model,
                                           const Transformer_packed_inputs* packed_inputs) {
    Computational_node_ptr encoder_input;
    Computational_node_ptr decoder_input;
    if (model == NULL || packed_inputs == NULL) {
        return false;
    }
    if (!transformer_model_initialize_graph_inputs(model)) {
        return false;
    }
    encoder_input = array_list_get(model->input_nodes, 0);
    decoder_input = array_list_get(model->input_nodes, 1);
    if (encoder_input == NULL || decoder_input == NULL ||
        packed_inputs->encoder_input == NULL || packed_inputs->decoder_input == NULL) {
        return false;
    }
    set_node_value(encoder_input, clone_tensor(packed_inputs->encoder_input));
    set_node_value(decoder_input, clone_tensor(packed_inputs->decoder_input));
    return true;
}

Computational_node_ptr transformer_model_add_class_label_input(Transformer_model_ptr model) {
    if (model == NULL || model->graph_bridge == NULL || model->input_nodes == NULL) {
        return NULL;
    }
    if (!transformer_model_initialize_graph_inputs(model)) {
        return NULL;
    }
    if (model->input_nodes->size >= 3) {
        return array_list_get(model->input_nodes, 2);
    }
    return recurrent_model_graph_bridge_add_input_node(model->graph_bridge, false, false);
}

Array_list_ptr transformer_model_get_output_value(const Computational_node* output_node) {
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
        double max = DBL_MIN;
        double index = -1.0;
        for (j = 0; j < output_node->value->shape[1]; j++) {
            double value = output_node->value->data[(i * output_node->value->shape[1]) + j];
            if (value > max) {
                max = value;
                index = (double) j;
            }
        }
        array_list_add_double(class_labels, index);
    }
    return class_labels;
}

bool transformer_model_build_graph(Transformer_model_ptr model) {
    Transformer_parameter_ptr parameter;
    Java_random_compat_ptr random;
    int ln_size[4] = {0, 0, 0, 0};
    Computational_node_ptr input1;
    Computational_node_ptr input2;
    Computational_node_ptr c1;
    Computational_node_ptr input_c1;
    Computational_node_ptr y1;
    Computational_node_ptr oe;
    Computational_node_ptr encoder;
    Computational_node_ptr c2;
    Computational_node_ptr input_c2;
    Computational_node_ptr cd2;
    Computational_node_ptr cd3;
    Computational_node_ptr cd3cd2;
    Computational_node_ptr yd1;
    Computational_node_ptr od;
    Computational_node_ptr oy;
    Computational_node_ptr d;
    Computational_node_ptr decoder;
    Array_list_ptr attention1 = NULL;
    Array_list_ptr attention2 = NULL;
    Array_list_ptr nodes = NULL;
    bool success = false;
    if (model == NULL || model->parameters == NULL) {
        return false;
    }
    if (model->graph_initialized) {
        return false;
    }
    /*
     * This staged slice treats graph construction as a one-way lifecycle step
     * for the instance, even if a later graph-building substep fails.
     */
    if (!transformer_model_initialize_graph_inputs(model) || model->input_nodes->size != 2) {
        model->graph_initialized = true;
        return false;
    }
    model->graph_initialized = true;
    parameter = model->parameters;
    random = create_java_random_compat(parameter->neural_network_parameter.seed);
    if (random == NULL) {
        return false;
    }
    input1 = array_list_get(model->input_nodes, 0);
    input2 = array_list_get(model->input_nodes, 1);
    attention1 = multi_head_attention(model, input1, parameter, false, random);
    if (attention1 == NULL) {
        goto cleanup;
    }
    {
        Concatenated_node_ptr concatenated_node1 =
                recurrent_model_graph_bridge_concat_edges(model->graph_bridge, attention1, 1);
        Multiplication_node_ptr we =
                create_transformer_weight_node(model, transformer_parameter_get_l(parameter),
                                               transformer_parameter_get_l(parameter), random);
        if (concatenated_node1 == NULL || we == NULL) {
            if (we != NULL) free_multiplication_node(we);
            goto cleanup;
        }
        c1 = add_weight_edge(model, (Computational_node_ptr) concatenated_node1, we, false);
    }
    input_c1 = add_addition_edge(model, input1, c1, false);
    y1 = layer_normalization(model, input_c1, parameter, true, ln_size);
    oe = add_addition_edge(model, feedforward_neural_network(model, y1, transformer_parameter_get_l(parameter),
                                                             parameter, random, true), y1, false);
    encoder = layer_normalization(model, oe, parameter, true, ln_size);
    if (c1 == NULL || input_c1 == NULL || y1 == NULL || oe == NULL || encoder == NULL) {
        goto cleanup;
    }
    attention2 = multi_head_attention(model, input2, parameter, true, random);
    if (attention2 == NULL) {
        goto cleanup;
    }
    {
        Concatenated_node_ptr concatenated_node2 =
                recurrent_model_graph_bridge_concat_edges(model->graph_bridge, attention2, 1);
        Multiplication_node_ptr wd1 =
                create_transformer_weight_node(model, transformer_parameter_get_l(parameter),
                                               transformer_parameter_get_l(parameter), random);
        if (concatenated_node2 == NULL || wd1 == NULL) {
            if (wd1 != NULL) free_multiplication_node(wd1);
            goto cleanup;
        }
        c2 = add_weight_edge(model, (Computational_node_ptr) concatenated_node2, wd1, false);
    }
    input_c2 = add_addition_edge(model, input2, c2, false);
    cd2 = layer_normalization(model, input_c2, parameter, false, ln_size);
    if (c2 == NULL || input_c2 == NULL || cd2 == NULL) {
        goto cleanup;
    }
    nodes = create_array_list();
    if (nodes == NULL) {
        goto cleanup;
    }
    {
        int i;
        for (i = 0; i < transformer_parameter_get_n(parameter); i++) {
            Multiplication_node_ptr wk = create_transformer_weight_node(model, transformer_parameter_get_l(parameter),
                                                                        transformer_parameter_get_dk(parameter), random);
            Multiplication_node_ptr wq = create_transformer_weight_node(model, transformer_parameter_get_l(parameter),
                                                                        transformer_parameter_get_dk(parameter), random);
            Multiplication_node_ptr wv = create_transformer_weight_node(model, transformer_parameter_get_l(parameter),
                                                                        transformer_parameter_get_dk(parameter), random);
            Computational_node_ptr k;
            Computational_node_ptr q;
            Computational_node_ptr v;
            Computational_node_ptr k_transpose;
            Computational_node_ptr qk;
            Computational_node_ptr qk_dk;
            Computational_node_ptr s_qk_dk;
            Computational_node_ptr attention;
            if (wk == NULL || wq == NULL || wv == NULL) {
                if (wk != NULL) free_multiplication_node(wk);
                if (wq != NULL) free_multiplication_node(wq);
                if (wv != NULL) free_multiplication_node(wv);
                goto cleanup;
            }
            k = add_weight_edge(model, encoder, wk, false);
            q = add_weight_edge(model, cd2, wq, false);
            v = add_weight_edge(model, encoder, wv, false);
            k_transpose = add_owned_function_edge(model, k, (Function*) create_transpose_function(), false);
            qk = add_matrix_multiplication_edge(model, q, k_transpose, false);
            qk_dk = add_owned_function_edge(model, qk,
                                            (Function*) create_multiply_by_constant(1.0 / sqrt(transformer_parameter_get_dk(parameter))),
                                            false);
            s_qk_dk = add_owned_function_edge(model, qk_dk, (Function*) create_softmax(), false);
            attention = add_matrix_multiplication_edge(model, s_qk_dk, v, false);
            if (k == NULL || q == NULL || v == NULL || k_transpose == NULL || qk == NULL ||
                qk_dk == NULL || s_qk_dk == NULL || attention == NULL) {
                goto cleanup;
            }
            array_list_add(nodes, attention);
        }
    }
    {
        Concatenated_node_ptr concatenated_node3 =
                recurrent_model_graph_bridge_concat_edges(model->graph_bridge, nodes, 1);
        Multiplication_node_ptr wd2 =
                create_transformer_weight_node(model, transformer_parameter_get_l(parameter),
                                               transformer_parameter_get_l(parameter), random);
        if (concatenated_node3 == NULL || wd2 == NULL) {
            if (wd2 != NULL) free_multiplication_node(wd2);
            goto cleanup;
        }
        cd3 = add_weight_edge(model, (Computational_node_ptr) concatenated_node3, wd2, false);
    }
    cd3cd2 = add_addition_edge(model, cd2, cd3, false);
    yd1 = layer_normalization(model, cd3cd2, parameter, false, ln_size);
    od = feedforward_neural_network(model, yd1, transformer_parameter_get_l(parameter), parameter, random, false);
    oy = add_addition_edge(model, od, yd1, false);
    d = layer_normalization(model, oy, parameter, false, ln_size);
    if (cd3 == NULL || cd3cd2 == NULL || yd1 == NULL || od == NULL || oy == NULL || d == NULL) {
        goto cleanup;
    }
    {
        Multiplication_node_ptr wdo = create_transformer_weight_node(model,
                                                                     transformer_parameter_get_l(parameter),
                                                                     transformer_parameter_get_v(parameter),
                                                                     random);
        Computational_node_ptr output_node;
        if (wdo == NULL) {
            goto cleanup;
        }
        decoder = add_weight_edge(model, d, wdo, false);
        if (decoder == NULL) {
            goto cleanup;
        }
        output_node = add_owned_function_edge(model, decoder, (Function*) create_softmax(), false);
        if (output_node == NULL) {
            goto cleanup;
        }
        recurrent_model_graph_bridge_set_output_node(model->graph_bridge, output_node);
    }
    if (transformer_model_add_class_label_input(model) == NULL) {
        goto cleanup;
    }
    success = true;

cleanup:
    if (attention1 != NULL) free_array_list(attention1, NULL);
    if (attention2 != NULL) free_array_list(attention2, NULL);
    if (nodes != NULL) free_array_list(nodes, NULL);
    free_java_random_compat(random);
    return success;
}

bool transformer_model_train(Transformer_model_ptr model, Array_list_ptr train_set) {
    Java_random_compat_ptr random;
    Transformer_parameter_ptr parameter;
    Computational_node_ptr class_label_node;
    Optimizer_ptr optimizer;
    int epoch;
    if (model == NULL || train_set == NULL || model->parameters == NULL) {
        return false;
    }
    /*
     * The staged train path reuses an already-claimed graph instance and does
     * not attempt to provide Java loss-node forward-value parity.
     */
    if (!model->graph_initialized) {
        if (!transformer_model_build_graph(model)) {
            return false;
        }
    } else if (model->input_nodes == NULL || model->input_nodes->size < 3) {
        return false;
    }
    parameter = model->parameters;
    class_label_node = array_list_get(model->input_nodes, 2);
    optimizer = parameter->neural_network_parameter.optimizer;
    if (class_label_node == NULL || optimizer == NULL) {
        return false;
    }
    random = create_java_random_compat(parameter->neural_network_parameter.seed);
    if (random == NULL) {
        return false;
    }
    for (epoch = 0; epoch < parameter->neural_network_parameter.epoch; epoch++) {
        int instance_index;
        if (!shuffle_train_set_like_java(train_set, random)) {
            free_java_random_compat(random);
            return false;
        }
        for (instance_index = 0; instance_index < train_set->size; instance_index++) {
            Tensor_ptr instance = array_list_get(train_set, instance_index);
            Transformer_packed_inputs_ptr packed_inputs =
                    transformer_model_create_packed_inputs(model, instance, transformer_parameter_get_l(parameter) - 1);
            Tensor_ptr class_label_tensor;
            int* class_label_index;
            Array_list_ptr predictions;
            if (packed_inputs == NULL) {
                free_java_random_compat(random);
                return false;
            }
            if (!transformer_model_apply_packed_inputs(model, packed_inputs)) {
                free_transformer_packed_inputs(packed_inputs);
                free_java_random_compat(random);
                return false;
            }
            class_label_tensor = create_transformer_class_label_tensor(
                    transformer_packed_inputs_get_class_labels(packed_inputs),
                    transformer_parameter_get_v(parameter));
            if (class_label_tensor == NULL) {
                free_transformer_packed_inputs(packed_inputs);
                free_java_random_compat(random);
                return false;
            }
            set_node_value(class_label_node, class_label_tensor);
            predictions = recurrent_model_graph_bridge_forward(model->graph_bridge);
            if (predictions == NULL) {
                free_transformer_packed_inputs(packed_inputs);
                free_java_random_compat(random);
                return false;
            }
            free_array_list(predictions, free_);
            class_label_index = create_transformer_class_label_index_array(
                    transformer_packed_inputs_get_class_labels(packed_inputs));
            if (class_label_index == NULL &&
                transformer_packed_inputs_get_class_labels(packed_inputs)->size > 0) {
                free_transformer_packed_inputs(packed_inputs);
                free_java_random_compat(random);
                return false;
            }
            recurrent_model_graph_bridge_back_propagation(model->graph_bridge, optimizer, class_label_index);
            free_(class_label_index);
            free_transformer_packed_inputs(packed_inputs);
        }
        set_learning_rate(optimizer);
    }
    free_java_random_compat(random);
    return true;
}
