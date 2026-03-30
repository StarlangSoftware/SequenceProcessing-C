#ifndef SEQUENCE_PROCESSING_GATED_RECURRENT_UNIT_MODEL_H
#define SEQUENCE_PROCESSING_GATED_RECURRENT_UNIT_MODEL_H

#include "Classification/RecurrentNeuralNetworkModel.h"

#include <stdbool.h>

typedef struct gated_recurrent_unit_model Gated_recurrent_unit_model;
typedef Gated_recurrent_unit_model* Gated_recurrent_unit_model_ptr;

struct gated_recurrent_unit_model {
    /*
     * Owned recurrent base model. This subclass adds no extra persistent state
     * beyond the base in the current local slice.
     */
    Recurrent_neural_network_model_ptr base_model;
};

/*
 * Ownership:
 * - borrowed: `parameters`
 * - owned: GRU model struct, owned base recurrent model
 */
Gated_recurrent_unit_model_ptr create_gated_recurrent_unit_model(Recurrent_neural_network_parameter_ptr parameters,
                                                                 int word_embedding_length);

void free_gated_recurrent_unit_model(Gated_recurrent_unit_model_ptr model);

/*
 * Returns a borrowed pointer to the owned recurrent base model so later ports
 * can reuse the shared recurrent/model bridge.
 */
Recurrent_neural_network_model_ptr gated_recurrent_unit_model_get_base(Gated_recurrent_unit_model_ptr model);

/*
 * Local GRU train port using the shared recurrent training bridge and
 * class-label-index backprop. Multi-input loss-node forward parity remains
 * deferred.
 *
 * Repeated `train(...)` calls on the same GRU instance are intentionally
 * rejected in this slice. The recurrent graph is built once on a fresh model
 * and then treated as owned model state until destruction.
 *
 * Ownership:
 * - borrowed: `train_set`
 */
bool gated_recurrent_unit_model_train(Gated_recurrent_unit_model_ptr model, Array_list_ptr train_set);

#endif
