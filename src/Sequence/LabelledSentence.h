#ifndef SEQUENCE_PROCESSING_LABELLED_SENTENCE_H
#define SEQUENCE_PROCESSING_LABELLED_SENTENCE_H

typedef struct sentence Sentence;
typedef Sentence* Sentence_ptr;

typedef struct labelled_sentence Labelled_sentence;
typedef Labelled_sentence* Labelled_sentence_ptr;

struct labelled_sentence {
    /*
     * Owned by Labelled_sentence.
     * This wrapper constructs the underlying Sentence via create_sentence()
     * and frees it in free_labelled_sentence().
     */
    Sentence_ptr sentence;

    /*
     * Owned by Labelled_sentence.
     * The constructor copies the incoming label string and the destructor frees it.
     */
    char* class_label;
};

/*
 * Mirrors the Java constructor LabelledSentence(String classLabel).
 * The returned wrapper owns both the created Sentence_ptr and the copied class label.
 */
Labelled_sentence_ptr create_labelled_sentence(const char* class_label);

const char* get_labelled_sentence_class_label(const Labelled_sentence_ptr labelled_sentence);

/*
 * Returns the wrapped Sentence_ptr.
 * The returned pointer is borrowed; ownership stays with Labelled_sentence.
 */
Sentence_ptr get_labelled_sentence_sentence(const Labelled_sentence_ptr labelled_sentence);

void free_labelled_sentence(Labelled_sentence_ptr labelled_sentence);

#endif
