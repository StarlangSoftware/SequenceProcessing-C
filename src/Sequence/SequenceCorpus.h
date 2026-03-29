#ifndef SEQUENCE_PROCESSING_SEQUENCE_CORPUS_H
#define SEQUENCE_PROCESSING_SEQUENCE_CORPUS_H

#include <stdbool.h>

typedef struct array_list Array_list;
typedef Array_list* Array_list_ptr;

typedef struct sequence_corpus Sequence_corpus;
typedef Sequence_corpus* Sequence_corpus_ptr;

/*
 * Mirrors the Java constructor SequenceCorpus(String fileName).
 * If the file cannot be opened, an empty corpus is returned.
 *
 * Ownership:
 * - the returned Sequence_corpus owns all parsed sentence and word objects
 * - free_sequence_corpus() releases the full parsed structure
 *
 * Design note:
 * - this is not a drop-in subclass of Corpus-C
 * - Corpus-C sentence storage assumes plain string words, while SequenceCorpus
 *   stores vectorized-word objects as in the Java source
 */
Sequence_corpus_ptr create_sequence_corpus(const char* file_name);

void free_sequence_corpus(Sequence_corpus_ptr sequence_corpus);

int sequence_corpus_sentence_count(const Sequence_corpus* sequence_corpus);

int sequence_corpus_number_of_words(const Sequence_corpus* sequence_corpus);

/*
 * Returns a newly allocated ArrayList of newly allocated class-label strings.
 * The caller owns both the returned list and its string contents.
 *
 * Java-compatibility note:
 * - Java decides whether labels live on sentences or words by inspecting only
 *   the first sentence
 * - this C port follows the same corpus-level assumption
 */
Array_list_ptr sequence_corpus_get_class_labels(const Sequence_corpus* sequence_corpus);

/*
 * Returns true when the corpus uses sentence-level labels, false when it uses word-level labels.
 * For an empty corpus this returns false.
 */
bool sequence_corpus_is_sentence_labelled(const Sequence_corpus* sequence_corpus);

#endif
