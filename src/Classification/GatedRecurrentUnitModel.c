#include "GatedRecurrentUnitModel.h"

#include "Memory/Memory.h"

Gated_recurrent_unit_model_ptr create_gated_recurrent_unit_model(Recurrent_neural_network_parameter_ptr parameters,
                                                                 int word_embedding_length) {
    Gated_recurrent_unit_model_ptr result = malloc_(sizeof(Gated_recurrent_unit_model));
    if (result == NULL) {
        return NULL;
    }
    result->base_model = create_recurrent_neural_network_model(parameters, word_embedding_length);
    if (result->base_model == NULL) {
        free_(result);
        return NULL;
    }
    /*
     * The Java constructor explicitly resets `switches` after calling `super`.
     * The current C base constructor already starts with an empty owned switch
     * list, so no extra reset work is required here.
     */
    return result;
}

void free_gated_recurrent_unit_model(Gated_recurrent_unit_model_ptr model) {
    if (model == NULL) {
        return;
    }
    free_recurrent_neural_network_model(model->base_model);
    free_(model);
}

Recurrent_neural_network_model_ptr gated_recurrent_unit_model_get_base(Gated_recurrent_unit_model_ptr model) {
    if (model == NULL) {
        return NULL;
    }
    return model->base_model;
}
