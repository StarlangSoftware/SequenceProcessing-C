#include "LabelledVectorizedWord.h"

#include "VectorizedWord.h"
#include "Vector.h"

#include <stdlib.h>
#include <string.h>

#define DEFAULT_LABELLED_VECTORIZED_WORD_EMBEDDING_SIZE 300

static char* duplicate_string(const char* value) {
    size_t length;
    char* copy;
    if (value == NULL) {
        return NULL;
    }
    length = strlen(value) + 1;
    copy = malloc(length);
    if (copy != NULL) {
        memcpy(copy, value, length);
    }
    return copy;
}

Labelled_vectorized_word_ptr create_labelled_vectorized_word(const char* word,
                                                             Vector_ptr embedding,
                                                             const char* class_label) {
    Labelled_vectorized_word_ptr labelled_vectorized_word = malloc(sizeof(Labelled_vectorized_word));
    if (labelled_vectorized_word == NULL) {
        if (embedding != NULL) {
            free_vector(embedding);
        }
        return NULL;
    }
    labelled_vectorized_word->vectorized_word = create_vectorized_word(word, embedding);
    if (labelled_vectorized_word->vectorized_word == NULL) {
        if (embedding != NULL) {
            free_vector(embedding);
        }
        free(labelled_vectorized_word);
        return NULL;
    }
    labelled_vectorized_word->class_label = duplicate_string(class_label);
    if (class_label != NULL && labelled_vectorized_word->class_label == NULL) {
        free_vectorized_word(labelled_vectorized_word->vectorized_word);
        free(labelled_vectorized_word);
        return NULL;
    }
    return labelled_vectorized_word;
}

Labelled_vectorized_word_ptr create_labelled_vectorized_word2(const char* word, const char* class_label) {
    Vector_ptr embedding = create_vector2(DEFAULT_LABELLED_VECTORIZED_WORD_EMBEDDING_SIZE, 0.0);
    if (embedding == NULL) {
        return NULL;
    }
    return create_labelled_vectorized_word(word,
                                           embedding,
                                           class_label);
}

const char* get_labelled_vectorized_word_class_label(const Labelled_vectorized_word_ptr labelled_vectorized_word) {
    if (labelled_vectorized_word == NULL) {
        return NULL;
    }
    return labelled_vectorized_word->class_label;
}

Vectorized_word_ptr get_labelled_vectorized_word(const Labelled_vectorized_word_ptr labelled_vectorized_word) {
    if (labelled_vectorized_word == NULL) {
        return NULL;
    }
    return labelled_vectorized_word->vectorized_word;
}

void free_labelled_vectorized_word(Labelled_vectorized_word_ptr labelled_vectorized_word) {
    if (labelled_vectorized_word == NULL) {
        return;
    }
    if (labelled_vectorized_word->vectorized_word != NULL) {
        free_vectorized_word(labelled_vectorized_word->vectorized_word);
    }
    free(labelled_vectorized_word->class_label);
    free(labelled_vectorized_word);
}
