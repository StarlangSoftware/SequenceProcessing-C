#include "SequenceCorpus.h"

#include "LabelledSentence.h"
#include "LabelledVectorizedWord.h"
#include "ArrayList.h"
#include "Sentence.h"
#include "Vector.h"
#include "VectorizedWord.h"
#include "FileUtils.h"
#include "Memory/Memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct sequence_word_entry {
    bool labelled;
    Vectorized_word_ptr vectorized_word;
    Labelled_vectorized_word_ptr labelled_vectorized_word;
} Sequence_word_entry;

typedef Sequence_word_entry* Sequence_word_entry_ptr;

typedef struct sequence_sentence_entry {
    bool sentence_labelled;
    Sentence_ptr sentence;
    Labelled_sentence_ptr labelled_sentence;
} Sequence_sentence_entry;

typedef Sequence_sentence_entry* Sequence_sentence_entry_ptr;

struct sequence_corpus {
    char* file_name;
    Array_list_ptr sentence_entries;
    /*
     * Follows the Java getClassLabels() assumption:
     * sentence-level vs word-level labelling is decided from the first sentence.
     */
    bool sentence_labelled;
    bool sentence_labelled_initialized;
};

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

static void trim_newline(char* line) {
    size_t length;
    if (line == NULL) {
        return;
    }
    length = strlen(line);
    while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) {
        line[length - 1] = '\0';
        length--;
    }
}

static int parse_line(char* line, char** first, char** second) {
    char* save_ptr = NULL;
    char* third;
    *first = strtok_r(line, " ", &save_ptr);
    if (*first == NULL) {
        *second = NULL;
        return 0;
    }
    *second = strtok_r(NULL, " ", &save_ptr);
    if (*second == NULL) {
        return 1;
    }
    third = strtok_r(NULL, " ", &save_ptr);
    if (third == NULL) {
        return 2;
    }
    return 3;
}

static Sequence_word_entry_ptr create_unlabelled_word_entry(const char* word) {
    Sequence_word_entry_ptr word_entry;
    Vector_ptr vector;
    Vectorized_word_ptr vectorized_word;
    vector = create_vector2(300, 0.0);
    if (vector == NULL) {
        return NULL;
    }
    vectorized_word = create_vectorized_word(word, vector);
    if (vectorized_word == NULL) {
        free_vector(vector);
        return NULL;
    }
    word_entry = malloc(sizeof(Sequence_word_entry));
    if (word_entry == NULL) {
        free_vectorized_word(vectorized_word);
        return NULL;
    }
    word_entry->labelled = false;
    word_entry->vectorized_word = vectorized_word;
    word_entry->labelled_vectorized_word = NULL;
    return word_entry;
}

static Sequence_word_entry_ptr create_labelled_word_entry(const char* word, const char* class_label) {
    Sequence_word_entry_ptr word_entry;
    Labelled_vectorized_word_ptr labelled_vectorized_word = create_labelled_vectorized_word2(word, class_label);
    if (labelled_vectorized_word == NULL) {
        return NULL;
    }
    word_entry = malloc(sizeof(Sequence_word_entry));
    if (word_entry == NULL) {
        free_labelled_vectorized_word(labelled_vectorized_word);
        return NULL;
    }
    word_entry->labelled = true;
    word_entry->vectorized_word = NULL;
    word_entry->labelled_vectorized_word = labelled_vectorized_word;
    return word_entry;
}

static void free_sequence_word_entry(Sequence_word_entry_ptr word_entry) {
    if (word_entry == NULL) {
        return;
    }
    if (word_entry->labelled) {
        free_labelled_vectorized_word(word_entry->labelled_vectorized_word);
    } else if (word_entry->vectorized_word != NULL) {
        free_vectorized_word(word_entry->vectorized_word);
    }
    free(word_entry);
}

static void free_sequence_sentence_storage(Sentence_ptr sentence) {
    if (sentence == NULL) {
        return;
    }
    free_array_list(sentence->words, (void (*)(void*)) free_sequence_word_entry);
    free_(sentence);
}

static void free_sequence_sentence_entry(Sequence_sentence_entry_ptr sentence_entry) {
    if (sentence_entry == NULL) {
        return;
    }
    if (sentence_entry->sentence_labelled) {
        if (sentence_entry->labelled_sentence != NULL) {
            free_sequence_sentence_storage(sentence_entry->labelled_sentence->sentence);
            free(sentence_entry->labelled_sentence->class_label);
            free(sentence_entry->labelled_sentence);
        }
    } else if (sentence_entry->sentence != NULL) {
        free_sequence_sentence_storage(sentence_entry->sentence);
    }
    free(sentence_entry);
}

static Sentence_ptr get_sentence_from_entry(const Sequence_sentence_entry_ptr sentence_entry) {
    if (sentence_entry == NULL) {
        return NULL;
    }
    if (sentence_entry->sentence_labelled) {
        return get_labelled_sentence_sentence(sentence_entry->labelled_sentence);
    }
    return sentence_entry->sentence;
}

static Sequence_sentence_entry_ptr create_sentence_entry(bool sentence_labelled, const char* class_label) {
    Sequence_sentence_entry_ptr sentence_entry = malloc(sizeof(Sequence_sentence_entry));
    if (sentence_entry == NULL) {
        return NULL;
    }
    sentence_entry->sentence_labelled = sentence_labelled;
    sentence_entry->sentence = NULL;
    sentence_entry->labelled_sentence = NULL;
    if (sentence_labelled) {
        sentence_entry->labelled_sentence = create_labelled_sentence(class_label);
        if (sentence_entry->labelled_sentence == NULL) {
            free(sentence_entry);
            return NULL;
        }
    } else {
        sentence_entry->sentence = create_sentence();
        if (sentence_entry->sentence == NULL) {
            free(sentence_entry);
            return NULL;
        }
    }
    return sentence_entry;
}

static void add_unique_label(Array_list_ptr class_labels, const char* label) {
    int i;
    if (label == NULL) {
        return;
    }
    for (i = 0; i < class_labels->size; i++) {
        char* existing = array_list_get(class_labels, i);
        if (strcmp(existing, label) == 0) {
            return;
        }
    }
    array_list_add(class_labels, duplicate_string(label));
}

Sequence_corpus_ptr create_sequence_corpus(const char* file_name) {
    FILE* input_file;
    char line[MAX_LINE_LENGTH];
    Sequence_corpus_ptr sequence_corpus = malloc(sizeof(Sequence_corpus));
    Sequence_sentence_entry_ptr current_sentence_entry = NULL;
    if (sequence_corpus == NULL) {
        return NULL;
    }
    sequence_corpus->file_name = duplicate_string(file_name);
    sequence_corpus->sentence_entries = create_array_list();
    sequence_corpus->sentence_labelled = false;
    sequence_corpus->sentence_labelled_initialized = false;
    if (sequence_corpus->sentence_entries == NULL) {
        free(sequence_corpus->file_name);
        free(sequence_corpus);
        return NULL;
    }
    input_file = fopen(file_name, "r");
    if (input_file == NULL) {
        return sequence_corpus;
    }
    while (fgets(line, MAX_LINE_LENGTH, input_file) != NULL) {
        char* first;
        char* second;
        int item_count;
        trim_newline(line);
        item_count = parse_line(line, &first, &second);
        if (item_count == 0) {
            continue;
        }
        if (strcmp(first, "<S>") == 0) {
            bool sentence_labelled = item_count == 2;
            if (current_sentence_entry != NULL) {
                free_sequence_sentence_entry(current_sentence_entry);
                current_sentence_entry = NULL;
            }
            current_sentence_entry = create_sentence_entry(sentence_labelled, second);
            if (current_sentence_entry == NULL) {
                break;
            }
            if (!sequence_corpus->sentence_labelled_initialized) {
                sequence_corpus->sentence_labelled = sentence_labelled;
                sequence_corpus->sentence_labelled_initialized = true;
            }
            continue;
        }
        if (strcmp(first, "</S>") == 0) {
            if (current_sentence_entry != NULL) {
                array_list_add(sequence_corpus->sentence_entries, current_sentence_entry);
                current_sentence_entry = NULL;
            }
            continue;
        }
        if (current_sentence_entry != NULL) {
            Sequence_word_entry_ptr word_entry = item_count == 2
                                                ? create_labelled_word_entry(first, second)
                                                : create_unlabelled_word_entry(first);
            if (word_entry == NULL) {
                break;
            }
            array_list_add(get_sentence_from_entry(current_sentence_entry)->words, word_entry);
        }
    }
    if (current_sentence_entry != NULL) {
        free_sequence_sentence_entry(current_sentence_entry);
    }
    fclose(input_file);
    return sequence_corpus;
}

void free_sequence_corpus(Sequence_corpus_ptr sequence_corpus) {
    if (sequence_corpus == NULL) {
        return;
    }
    free(sequence_corpus->file_name);
    free_array_list(sequence_corpus->sentence_entries, (void (*)(void*)) free_sequence_sentence_entry);
    free(sequence_corpus);
}

int sequence_corpus_sentence_count(const Sequence_corpus* sequence_corpus) {
    if (sequence_corpus == NULL || sequence_corpus->sentence_entries == NULL) {
        return 0;
    }
    return sequence_corpus->sentence_entries->size;
}

int sequence_corpus_number_of_words(const Sequence_corpus* sequence_corpus) {
    int total = 0;
    int i;
    if (sequence_corpus == NULL || sequence_corpus->sentence_entries == NULL) {
        return 0;
    }
    for (i = 0; i < sequence_corpus->sentence_entries->size; i++) {
        Sequence_sentence_entry_ptr sentence_entry = array_list_get(sequence_corpus->sentence_entries, i);
        Sentence_ptr sentence = get_sentence_from_entry(sentence_entry);
        total += sentence->words->size;
    }
    return total;
}

Array_list_ptr sequence_corpus_get_class_labels(const Sequence_corpus* sequence_corpus) {
    Array_list_ptr class_labels = create_array_list();
    int i;
    if (class_labels == NULL || sequence_corpus == NULL || sequence_corpus->sentence_entries == NULL) {
        return class_labels;
    }
    if (!sequence_corpus->sentence_labelled_initialized) {
        return class_labels;
    }
    for (i = 0; i < sequence_corpus->sentence_entries->size; i++) {
        Sequence_sentence_entry_ptr sentence_entry = array_list_get(sequence_corpus->sentence_entries, i);
        Sentence_ptr sentence = get_sentence_from_entry(sentence_entry);
        if (sequence_corpus->sentence_labelled) {
            add_unique_label(class_labels, get_labelled_sentence_class_label(sentence_entry->labelled_sentence));
        } else {
            int j;
            for (j = 0; j < sentence->words->size; j++) {
                Sequence_word_entry_ptr word_entry = array_list_get(sentence->words, j);
                if (word_entry->labelled) {
                    add_unique_label(class_labels,
                                     get_labelled_vectorized_word_class_label(word_entry->labelled_vectorized_word));
                }
            }
        }
    }
    return class_labels;
}

bool sequence_corpus_is_sentence_labelled(const Sequence_corpus* sequence_corpus) {
    if (sequence_corpus == NULL || !sequence_corpus->sentence_labelled_initialized) {
        return false;
    }
    return sequence_corpus->sentence_labelled;
}
