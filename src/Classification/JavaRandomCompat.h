#ifndef SEQUENCE_PROCESSING_JAVA_RANDOM_COMPAT_H
#define SEQUENCE_PROCESSING_JAVA_RANDOM_COMPAT_H

#include <stdbool.h>

typedef struct java_random_compat Java_random_compat;
typedef Java_random_compat* Java_random_compat_ptr;

/*
 * Local java.util.Random-compatible 48-bit LCG for grounded Java recurrent
 * training parity.
 *
 * Ownership:
 * - caller owns the returned RNG object
 */
Java_random_compat_ptr create_java_random_compat(int seed);

void free_java_random_compat(Java_random_compat_ptr random);

double java_random_compat_next_double(Java_random_compat_ptr random);

int java_random_compat_next_int(Java_random_compat_ptr random, int bound);

bool java_random_compat_shuffle_pair_indices(Java_random_compat_ptr random, int size, int* first, int* second);

#endif
