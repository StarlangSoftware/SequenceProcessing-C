#ifndef SEQUENCE_PROCESSING_TRANSFORMER_H
#define SEQUENCE_PROCESSING_TRANSFORMER_H

#include "Parameters/TransformerParameter.h"

typedef struct vectorized_dictionary Vectorized_dictionary;
typedef Vectorized_dictionary* Vectorized_dictionary_ptr;

typedef struct transformer_model Transformer_model;
typedef Transformer_model* Transformer_model_ptr;

/*
 * Constructor shell matching the Java Transformer constructor at the minimal
 * currently grounded level.
 *
 * Ownership:
 * - borrowed: `parameters`, `dictionary`
 * - owned: Transformer shell struct
 */
Transformer_model_ptr create_transformer_model(Transformer_parameter_ptr parameters,
                                               Vectorized_dictionary_ptr dictionary);

void free_transformer_model(Transformer_model_ptr model);

Vectorized_dictionary_ptr transformer_model_get_dictionary(const Transformer_model* model);

int transformer_model_get_start_index(const Transformer_model* model);

int transformer_model_get_end_index(const Transformer_model* model);

#endif
