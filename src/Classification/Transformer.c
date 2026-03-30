#include "Transformer.h"

#include "Dictionary/Dictionary.h"
#include "Dictionary/VectorizedDictionary.h"
#include "Dictionary/VectorizedWord.h"
#include "Memory/Memory.h"

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
