#include "TransformerParameter.h"

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

static Array_list_ptr clone_double_list(const Array_list* source) {
    int i;
    Array_list_ptr result = create_array_list();
    if (result == NULL) {
        return NULL;
    }
    if (source == NULL) {
        return result;
    }
    for (i = 0; i < source->size; i++) {
        array_list_add_double(result, array_list_get_double(source, i));
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

static void free_transformer_parameter_lists(Transformer_parameter_ptr parameter) {
    if (parameter->input_hidden_layers != NULL) {
        free_array_list(parameter->input_hidden_layers, free_);
    }
    if (parameter->output_hidden_layers != NULL) {
        free_array_list(parameter->output_hidden_layers, free_);
    }
    if (parameter->input_functions != NULL) {
        free_array_list(parameter->input_functions, NULL);
    }
    if (parameter->output_functions != NULL) {
        free_array_list(parameter->output_functions, NULL);
    }
    if (parameter->gamma_input_values != NULL) {
        free_array_list(parameter->gamma_input_values, free_);
    }
    if (parameter->gamma_output_values != NULL) {
        free_array_list(parameter->gamma_output_values, free_);
    }
    if (parameter->beta_input_values != NULL) {
        free_array_list(parameter->beta_input_values, free_);
    }
    if (parameter->beta_output_values != NULL) {
        free_array_list(parameter->beta_output_values, free_);
    }
}

Transformer_parameter_ptr create_transformer_parameter(int seed,
                                                       int epoch,
                                                       Optimizer_ptr optimizer,
                                                       Initialization initialization,
                                                       Function* loss,
                                                       int word_embedding_length,
                                                       int multi_head_attention_length,
                                                       int vocabulary_length,
                                                       double epsilon,
                                                       const Array_list* input_hidden_layers,
                                                       const Array_list* output_hidden_layers,
                                                       const Array_list* input_activation_functions,
                                                       const Array_list* output_activation_functions,
                                                       const Array_list* gamma_input_values,
                                                       const Array_list* gamma_output_values,
                                                       const Array_list* beta_input_values,
                                                       const Array_list* beta_output_values) {
    Transformer_parameter_ptr result = malloc_(sizeof(Transformer_parameter));
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
    result->l = word_embedding_length + 1;
    result->n = multi_head_attention_length;
    result->v = vocabulary_length;
    result->epsilon = epsilon;
    result->input_hidden_layers = NULL;
    result->output_hidden_layers = NULL;
    result->input_functions = NULL;
    result->output_functions = NULL;
    result->gamma_input_values = NULL;
    result->gamma_output_values = NULL;
    result->beta_input_values = NULL;
    result->beta_output_values = NULL;
    result->input_hidden_layers = clone_int_list(input_hidden_layers);
    result->output_hidden_layers = clone_int_list(output_hidden_layers);
    result->input_functions = clone_pointer_list(input_activation_functions);
    result->output_functions = clone_pointer_list(output_activation_functions);
    result->gamma_input_values = clone_double_list(gamma_input_values);
    result->gamma_output_values = clone_double_list(gamma_output_values);
    result->beta_input_values = clone_double_list(beta_input_values);
    result->beta_output_values = clone_double_list(beta_output_values);
    if (result->input_hidden_layers == NULL ||
        result->output_hidden_layers == NULL ||
        result->input_functions == NULL ||
        result->output_functions == NULL ||
        result->gamma_input_values == NULL ||
        result->gamma_output_values == NULL ||
        result->beta_input_values == NULL ||
        result->beta_output_values == NULL) {
        free_transformer_parameter_lists(result);
        free_(result);
        return NULL;
    }
    return result;
}

void free_transformer_parameter(Transformer_parameter_ptr parameter) {
    if (parameter == NULL) {
        return;
    }
    free_transformer_parameter_lists(parameter);
    free_(parameter);
}

double transformer_parameter_get_gamma_input_value(const Transformer_parameter* parameter, int index) {
    if (parameter == NULL || parameter->gamma_input_values == NULL) {
        return 0.0;
    }
    return array_list_get_double(parameter->gamma_input_values, index);
}

double transformer_parameter_get_gamma_output_value(const Transformer_parameter* parameter, int index) {
    if (parameter == NULL || parameter->gamma_output_values == NULL) {
        return 0.0;
    }
    return array_list_get_double(parameter->gamma_output_values, index);
}

double transformer_parameter_get_beta_input_value(const Transformer_parameter* parameter, int index) {
    if (parameter == NULL || parameter->beta_input_values == NULL) {
        return 0.0;
    }
    return array_list_get_double(parameter->beta_input_values, index);
}

double transformer_parameter_get_beta_output_value(const Transformer_parameter* parameter, int index) {
    if (parameter == NULL || parameter->beta_output_values == NULL) {
        return 0.0;
    }
    return array_list_get_double(parameter->beta_output_values, index);
}

double transformer_parameter_get_epsilon(const Transformer_parameter* parameter) {
    if (parameter == NULL) {
        return 0.0;
    }
    return parameter->epsilon;
}

int transformer_parameter_get_dk(const Transformer_parameter* parameter) {
    if (parameter == NULL || parameter->n == 0) {
        return 0;
    }
    return parameter->l / parameter->n;
}

int transformer_parameter_get_l(const Transformer_parameter* parameter) {
    if (parameter == NULL) {
        return 0;
    }
    return parameter->l;
}

int transformer_parameter_get_n(const Transformer_parameter* parameter) {
    if (parameter == NULL) {
        return 0;
    }
    return parameter->n;
}

int transformer_parameter_get_v(const Transformer_parameter* parameter) {
    if (parameter == NULL) {
        return 0;
    }
    return parameter->v;
}

int transformer_parameter_get_input_hidden_layer(const Transformer_parameter* parameter, int index) {
    if (parameter == NULL || parameter->input_hidden_layers == NULL) {
        return 0;
    }
    return array_list_get_int(parameter->input_hidden_layers, index);
}

int transformer_parameter_get_output_hidden_layer(const Transformer_parameter* parameter, int index) {
    if (parameter == NULL || parameter->output_hidden_layers == NULL) {
        return 0;
    }
    return array_list_get_int(parameter->output_hidden_layers, index);
}

Function* transformer_parameter_get_input_activation_function(const Transformer_parameter* parameter, int index) {
    if (parameter == NULL || parameter->input_functions == NULL) {
        return NULL;
    }
    return array_list_get(parameter->input_functions, index);
}

Function* transformer_parameter_get_output_activation_function(const Transformer_parameter* parameter, int index) {
    if (parameter == NULL || parameter->output_functions == NULL) {
        return NULL;
    }
    return array_list_get(parameter->output_functions, index);
}

int transformer_parameter_get_input_size(const Transformer_parameter* parameter) {
    if (parameter == NULL || parameter->input_hidden_layers == NULL) {
        return 0;
    }
    return parameter->input_hidden_layers->size;
}

int transformer_parameter_get_output_size(const Transformer_parameter* parameter) {
    if (parameter == NULL || parameter->output_hidden_layers == NULL) {
        return 0;
    }
    return parameter->output_hidden_layers->size;
}
