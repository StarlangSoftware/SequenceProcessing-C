#include "LabelledSentence.h"

#include "Sentence.h"

#include <stdlib.h>
#include <string.h>

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

Labelled_sentence_ptr create_labelled_sentence(const char* class_label) {
    Labelled_sentence_ptr labelled_sentence = malloc(sizeof(Labelled_sentence));
    if (labelled_sentence == NULL) {
        return NULL;
    }
    labelled_sentence->sentence = create_sentence();
    if (labelled_sentence->sentence == NULL) {
        free(labelled_sentence);
        return NULL;
    }
    labelled_sentence->class_label = duplicate_string(class_label);
    if (class_label != NULL && labelled_sentence->class_label == NULL) {
        free_sentence(labelled_sentence->sentence);
        free(labelled_sentence);
        return NULL;
    }
    return labelled_sentence;
}

const char* get_labelled_sentence_class_label(const Labelled_sentence_ptr labelled_sentence) {
    if (labelled_sentence == NULL) {
        return NULL;
    }
    return labelled_sentence->class_label;
}

Sentence_ptr get_labelled_sentence_sentence(const Labelled_sentence_ptr labelled_sentence) {
    if (labelled_sentence == NULL) {
        return NULL;
    }
    return labelled_sentence->sentence;
}

void free_labelled_sentence(Labelled_sentence_ptr labelled_sentence) {
    if (labelled_sentence == NULL) {
        return;
    }
    if (labelled_sentence->sentence != NULL) {
        free_sentence(labelled_sentence->sentence);
    }
    free(labelled_sentence->class_label);
    free(labelled_sentence);
}
