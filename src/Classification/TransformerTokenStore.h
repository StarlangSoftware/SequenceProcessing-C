#ifndef SEQUENCE_PROCESSING_TRANSFORMER_TOKEN_STORE_H
#define SEQUENCE_PROCESSING_TRANSFORMER_TOKEN_STORE_H

#include <stdbool.h>

typedef struct vector Vector;
typedef Vector* Vector_ptr;

typedef struct transformer_token_store Transformer_token_store;
typedef Transformer_token_store* Transformer_token_store_ptr;

Transformer_token_store_ptr create_transformer_token_store(void);

void free_transformer_token_store(Transformer_token_store_ptr store);

/*
 * Adds one owned token entry.
 *
 * Ownership:
 * - borrowed: `token`, `vector`
 * - owned by store after success: copied token string, cloned vector payload
 */
bool transformer_token_store_add(Transformer_token_store_ptr store, const char* token, const Vector* vector);

void transformer_token_store_sort(Transformer_token_store_ptr store);

int transformer_token_store_find_index(const Transformer_token_store* store, const char* token);

const Vector* transformer_token_store_get_vector(const Transformer_token_store* store, int index);

int transformer_token_store_get_embedding_size(const Transformer_token_store* store);

#endif
