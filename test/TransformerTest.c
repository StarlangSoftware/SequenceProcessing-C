#include "Classification/Transformer.h"

#include "ArrayList.h"
#include "Dictionary/Dictionary.h"
#include "Dictionary/VectorizedDictionary.h"
#include "Dictionary/VectorizedWord.h"
#include "Initialization/Initialization.h"
#include "Memory/Memory.h"
#include "Optimizer/Optimizer.h"

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

static Vectorized_dictionary_ptr create_test_dictionary(void) {
    Vectorized_dictionary_ptr dictionary = create_vectorized_dictionary();
    add_word((Dictionary_ptr) dictionary, (Word_ptr) create_vectorized_word("hello", create_vector2(2, 0.0)));
    add_word((Dictionary_ptr) dictionary, (Word_ptr) create_vectorized_word("<S>", create_vector2(2, 0.0)));
    add_word((Dictionary_ptr) dictionary, (Word_ptr) create_vectorized_word("</S>", create_vector2(2, 0.0)));
    sort((Dictionary_ptr) dictionary);
    return dictionary;
}

static int test_transformer_constructor_scans_start_and_end_tokens(void) {
    Transformer_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Vectorized_dictionary_ptr dictionary = create_test_dictionary();
    Transformer_model_ptr model = create_transformer_model(parameter, dictionary);
    int success = model != NULL &&
                  transformer_model_get_dictionary(model) == dictionary &&
                  transformer_model_get_start_index(model) == get_word_index((Dictionary_ptr) dictionary, "<S>") &&
                  transformer_model_get_end_index(model) == get_word_index((Dictionary_ptr) dictionary, "</S>");
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_vectorized_dictionary(dictionary);
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
    Vectorized_dictionary_ptr dictionary = create_vectorized_dictionary();
    Transformer_model_ptr model = create_transformer_model(parameter, dictionary);
    int success = model != NULL &&
                  transformer_model_get_start_index(model) == -1 &&
                  transformer_model_get_end_index(model) == -1;
    free_transformer_model(model);
    free_transformer_parameter(parameter);
    free_vectorized_dictionary(dictionary);
    free_array_list(empty_ints, free_);
    free_array_list(empty_functions, NULL);
    free_array_list(empty_doubles, free_);
    free_(optimizer);
    return success;
}

static int test_transformer_positional_encoding_helper(void) {
    Transformer_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Vectorized_dictionary_ptr dictionary = create_test_dictionary();
    Transformer_model_ptr model = create_transformer_model(parameter, dictionary);
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
    free_vectorized_dictionary(dictionary);
    free_(optimizer);
    return success;
}

static int test_transformer_create_packed_inputs_helper(void) {
    Transformer_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Vectorized_dictionary_ptr dictionary = create_test_dictionary();
    Transformer_model_ptr model = create_transformer_model(parameter, dictionary);
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
    free_vectorized_dictionary(dictionary);
    free_(optimizer);
    return success;
}

int main(void) {
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
    return 0;
}
