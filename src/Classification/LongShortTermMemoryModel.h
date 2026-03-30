#ifndef SEQUENCE_PROCESSING_LONG_SHORT_TERM_MEMORY_MODEL_H
#define SEQUENCE_PROCESSING_LONG_SHORT_TERM_MEMORY_MODEL_H

#include "Classification/RecurrentNeuralNetworkModel.h"

#include <stdbool.h>

typedef struct long_short_term_memory_model Long_short_term_memory_model;
typedef Long_short_term_memory_model* Long_short_term_memory_model_ptr;

struct long_short_term_memory_model {
    /*
     * Owned recurrent base model. This subclass adds no extra persistent state
     * beyond the base in the current local slice.
     */
    Recurrent_neural_network_model_ptr base_model;
};

/*
 * Ownership:
 * - borrowed: `parameters`
 * - owned: LSTM model struct, owned base recurrent model
 */
Long_short_term_memory_model_ptr create_long_short_term_memory_model(Recurrent_neural_network_parameter_ptr parameters,
                                                                     int word_embedding_length);

void free_long_short_term_memory_model(Long_short_term_memory_model_ptr model);

/*
 * Returns a borrowed pointer to the owned recurrent base model.
 */
Recurrent_neural_network_model_ptr long_short_term_memory_model_get_base(Long_short_term_memory_model_ptr model);

/*
 * Local LSTM train port using the shared recurrent training bridge and
 * class-label-index backprop. Multi-input loss-node forward parity remains
 * deferred.
 *
 * Repeated `train(...)` calls on the same LSTM instance are intentionally
 * rejected in this slice. The recurrent graph is built once on a fresh model
 * and then treated as owned model state until destruction.
 *
 * Ownership:
 * - borrowed: `train_set`
 */
bool long_short_term_memory_model_train(Long_short_term_memory_model_ptr model, Array_list_ptr train_set);

#endif
