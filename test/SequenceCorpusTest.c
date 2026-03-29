#include "Sequence/SequenceCorpus.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int write_temp_file(const char* contents, char* path_template) {
    FILE* output;
    int fd = mkstemp(path_template);
    if (fd < 0) {
        return 0;
    }
    output = fdopen(fd, "w");
    if (output == NULL) {
        close(fd);
        return 0;
    }
    fputs(contents, output);
    fclose(output);
    return 1;
}

int main(void) {
    char sentence_labelled_path[] = "/tmp/sequence_corpus_sentence_XXXXXX";
    char word_labelled_path[] = "/tmp/sequence_corpus_word_XXXXXX";
    Sequence_corpus_ptr corpus;

    if (!write_temp_file("<S> positive\nhello\nworld\n</S>\n<S> negative\nbye\n</S>\n",
                         sentence_labelled_path)) {
        return 1;
    }
    corpus = create_sequence_corpus(sentence_labelled_path);
    if (corpus == NULL) {
        unlink(sentence_labelled_path);
        return 1;
    }
    if (sequence_corpus_sentence_count(corpus) != 2 || sequence_corpus_number_of_words(corpus) != 3) {
        free_sequence_corpus(corpus);
        unlink(sentence_labelled_path);
        return 1;
    }
    free_sequence_corpus(corpus);
    unlink(sentence_labelled_path);

    if (!write_temp_file("<S>\nhello NN\nworld VB\n</S>\n<S>\nbye NN\n</S>\n", word_labelled_path)) {
        return 1;
    }
    corpus = create_sequence_corpus(word_labelled_path);
    if (corpus == NULL) {
        unlink(word_labelled_path);
        return 1;
    }
    if (sequence_corpus_sentence_count(corpus) != 2 || sequence_corpus_number_of_words(corpus) != 3) {
        free_sequence_corpus(corpus);
        unlink(word_labelled_path);
        return 1;
    }
    free_sequence_corpus(corpus);
    unlink(word_labelled_path);
    return 0;
}
