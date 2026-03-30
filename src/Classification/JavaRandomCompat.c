#include "JavaRandomCompat.h"

#include "Memory/Memory.h"

#include <stdint.h>

struct java_random_compat {
    uint64_t seed;
};

#define JAVA_RANDOM_MULTIPLIER UINT64_C(25214903917)
#define JAVA_RANDOM_ADDEND UINT64_C(11)
#define JAVA_RANDOM_MASK ((UINT64_C(1) << 48) - 1)

static int java_random_compat_next_bits(Java_random_compat_ptr random, int bits) {
    if (random == NULL || bits <= 0 || bits > 32) {
        return 0;
    }
    random->seed = (random->seed * JAVA_RANDOM_MULTIPLIER + JAVA_RANDOM_ADDEND) & JAVA_RANDOM_MASK;
    return (int) (random->seed >> (48 - bits));
}

Java_random_compat_ptr create_java_random_compat(int seed) {
    Java_random_compat_ptr result = malloc_(sizeof(Java_random_compat));
    if (result == NULL) {
        return NULL;
    }
    result->seed = (((uint64_t) (int64_t) seed) ^ JAVA_RANDOM_MULTIPLIER) & JAVA_RANDOM_MASK;
    return result;
}

void free_java_random_compat(Java_random_compat_ptr random) {
    if (random == NULL) {
        return;
    }
    free_(random);
}

double java_random_compat_next_double(Java_random_compat_ptr random) {
    uint64_t high;
    uint64_t low;
    if (random == NULL) {
        return 0.0;
    }
    high = (uint64_t) java_random_compat_next_bits(random, 26);
    low = (uint64_t) java_random_compat_next_bits(random, 27);
    return ((high << 27) + low) / (double) (UINT64_C(1) << 53);
}

int java_random_compat_next_int(Java_random_compat_ptr random, int bound) {
    int bits;
    int value;
    if (random == NULL || bound <= 0) {
        return 0;
    }
    if ((bound & -bound) == bound) {
        return (int) ((bound * (int64_t) java_random_compat_next_bits(random, 31)) >> 31);
    }
    do {
        bits = java_random_compat_next_bits(random, 31);
        value = bits % bound;
    } while (bits - value + (bound - 1) < 0);
    return value;
}

bool java_random_compat_shuffle_pair_indices(Java_random_compat_ptr random, int size, int* first, int* second) {
    if (random == NULL || size <= 0 || first == NULL || second == NULL) {
        return false;
    }
    *first = java_random_compat_next_int(random, size);
    *second = java_random_compat_next_int(random, size);
    return true;
}
