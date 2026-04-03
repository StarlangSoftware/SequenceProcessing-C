#include "TransformerTokenStore.h"

#include "ArrayList.h"
#include "Memory/Memory.h"
#include "StringUtils.h"
#include "Vector.h"

#include <string.h>

typedef struct transformer_token_entry {
    char* token;
    Vector_ptr vector;
} Transformer_token_entry;

struct transformer_token_store {
    Array_list_ptr entries;
};

static int compare_token_entries(const Transformer_token_entry* first, const Transformer_token_entry* second) {
    return strcmp(first->token, second->token);
}

static void free_token_entry(void* data) {
    Transformer_token_entry* entry = data;
    if (entry == NULL) {
        return;
    }
    free_(entry->token);
    if (entry->vector != NULL) {
        free_vector(entry->vector);
    }
    free_(entry);
}

static Vector_ptr clone_token_vector(const Vector* vector) {
    double* values;
    Vector_ptr result;
    int i;
    if (vector == NULL) {
        return NULL;
    }
    values = malloc_(vector->size * sizeof(double));
    if (values == NULL && vector->size > 0) {
        return NULL;
    }
    for (i = 0; i < vector->size; i++) {
        values[i] = get_value(vector, i);
    }
    result = create_vector4(values, vector->size);
    free_(values);
    return result;
}

Transformer_token_store_ptr create_transformer_token_store(void) {
    Transformer_token_store_ptr store = malloc_(sizeof(Transformer_token_store));
    if (store == NULL) {
        return NULL;
    }
    store->entries = create_array_list();
    if (store->entries == NULL) {
        free_(store);
        return NULL;
    }
    return store;
}

void free_transformer_token_store(Transformer_token_store_ptr store) {
    if (store == NULL) {
        return;
    }
    free_array_list(store->entries, free_token_entry);
    free_(store);
}

bool transformer_token_store_add(Transformer_token_store_ptr store, const char* token, const Vector* vector) {
    char* copy;
    Vector_ptr vector_copy;
    Transformer_token_entry* entry;
    if (store == NULL || store->entries == NULL || token == NULL || vector == NULL) {
        return false;
    }
    copy = str_copy(NULL, token);
    if (copy == NULL) {
        return false;
    }
    vector_copy = clone_token_vector(vector);
    if (vector_copy == NULL) {
        free_(copy);
        return false;
    }
    entry = malloc_(sizeof(Transformer_token_entry));
    if (entry == NULL) {
        free_(copy);
        free_vector(vector_copy);
        return false;
    }
    entry->token = copy;
    entry->vector = vector_copy;
    array_list_add(store->entries, entry);
    return true;
}

void transformer_token_store_sort(Transformer_token_store_ptr store) {
    if (store == NULL || store->entries == NULL) {
        return;
    }
    array_list_sort(store->entries, (int (*)(const void*, const void*)) compare_token_entries);
}

int transformer_token_store_find_index(const Transformer_token_store* store, const char* token) {
    int i;
    if (store == NULL || store->entries == NULL || token == NULL) {
        return -1;
    }
    for (i = 0; i < store->entries->size; i++) {
        const Transformer_token_entry* current = array_list_get(store->entries, i);
        if (current != NULL && current->token != NULL && strcmp(current->token, token) == 0) {
            return i;
        }
    }
    return -1;
}

const Vector* transformer_token_store_get_vector(const Transformer_token_store* store, int index) {
    const Transformer_token_entry* entry;
    if (store == NULL || store->entries == NULL || index < 0 || index >= store->entries->size) {
        return NULL;
    }
    entry = array_list_get(store->entries, index);
    return (entry != NULL) ? entry->vector : NULL;
}

int transformer_token_store_get_embedding_size(const Transformer_token_store* store) {
    const Vector* vector;
    if (store == NULL || store->entries == NULL || store->entries->size == 0) {
        return -1;
    }
    vector = transformer_token_store_get_vector(store, 0);
    return (vector != NULL) ? vector->size : -1;
}
