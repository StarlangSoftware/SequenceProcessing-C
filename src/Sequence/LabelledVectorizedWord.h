#ifndef SEQUENCE_PROCESSING_LABELLED_VECTORIZED_WORD_H
#define SEQUENCE_PROCESSING_LABELLED_VECTORIZED_WORD_H

typedef struct vector Vector;
typedef Vector* Vector_ptr;

typedef struct vectorized_word Vectorized_word;
typedef Vectorized_word* Vectorized_word_ptr;

typedef struct labelled_vectorized_word Labelled_vectorized_word;
typedef Labelled_vectorized_word* Labelled_vectorized_word_ptr;

struct labelled_vectorized_word {
    /*
     * Owned by Labelled_vectorized_word.
     * The constructors create the underlying Vectorized_word and
     * free_labelled_vectorized_word() releases it.
     */
    Vectorized_word_ptr vectorized_word;

    /*
     * Owned by Labelled_vectorized_word.
     * The constructor copies the incoming label string and the destructor frees it.
     */
    char* class_label;
};

/*
 * Mirrors the Java constructor LabelledVectorizedWord(String word, Vector embedding, String classLabel).
 * Ownership of the provided embedding transfers to the created Vectorized_word and then to this wrapper.
 */
Labelled_vectorized_word_ptr create_labelled_vectorized_word(const char* word,
                                                             Vector_ptr embedding,
                                                             const char* class_label);

/*
 * Mirrors the Java constructor LabelledVectorizedWord(String word, String classLabel).
 * Creates and owns a zero-filled Vector of length 300.
 */
Labelled_vectorized_word_ptr create_labelled_vectorized_word2(const char* word, const char* class_label);

const char* get_labelled_vectorized_word_class_label(const Labelled_vectorized_word_ptr labelled_vectorized_word);

/*
 * Returns the wrapped Vectorized_word_ptr.
 * The returned pointer is borrowed; ownership stays with Labelled_vectorized_word.
 */
Vectorized_word_ptr get_labelled_vectorized_word(const Labelled_vectorized_word_ptr labelled_vectorized_word);

void free_labelled_vectorized_word(Labelled_vectorized_word_ptr labelled_vectorized_word);

#endif
