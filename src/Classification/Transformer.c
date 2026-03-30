#include "Transformer.h"

#include "ArrayList.h"
#include "Dictionary/Dictionary.h"
#include "Dictionary/VectorizedDictionary.h"
#include "Dictionary/VectorizedWord.h"
#include "Memory/Memory.h"

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

    int start_index;
    int end_index;
};

struct transformer_packed_inputs {
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

Transformer_model_ptr create_transformer_model(Transformer_parameter_ptr parameters,
                                               Vectorized_dictionary_ptr dictionary) {
    Transformer_model_ptr result = malloc_(sizeof(Transformer_model));
    if (result == NULL) {
        return NULL;
    }
    result->parameters = parameters;
    result->dictionary = dictionary;
    result->start_index = find_token_index(dictionary, "<S>");
    result->end_index = find_token_index(dictionary, "</S>");
    return result;
}

void free_transformer_model(Transformer_model_ptr model) {
    if (model == NULL) {
        return;
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
