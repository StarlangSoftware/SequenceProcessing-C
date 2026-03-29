#include "Classification/GatedRecurrentUnitModel.h"

#include "ArrayList.h"
#include "Initialization/Initialization.h"
#include "Memory/Memory.h"
#include "Optimizer/Optimizer.h"

static Recurrent_neural_network_parameter_ptr create_test_parameter(void) {
    Array_list_ptr hidden_layers = create_array_list();
    Array_list_ptr functions = create_array_list();
    Optimizer_ptr optimizer = create_optimizer(0.1, 1.0);
    Recurrent_neural_network_parameter_ptr parameter =
            create_recurrent_neural_network_parameter(13, 2, optimizer, Random, NULL, hidden_layers, functions, 3);
    free_array_list(hidden_layers, free_);
    free_array_list(functions, NULL);
    return parameter;
}

static int test_gru_constructor_and_base_access(void) {
    Recurrent_neural_network_parameter_ptr parameter = create_test_parameter();
    Optimizer_ptr optimizer = parameter->neural_network_parameter.optimizer;
    Gated_recurrent_unit_model_ptr model = create_gated_recurrent_unit_model(parameter, 5);
    Recurrent_neural_network_model_ptr base = gated_recurrent_unit_model_get_base(model);
    int success = model != NULL &&
                  base != NULL &&
                  base->parameters == parameter &&
                  base->word_embedding_length == 5 &&
                  base->input_nodes != NULL &&
                  base->switches != NULL &&
                  base->input_nodes->size == 0 &&
                  base->switches->size == 0;
    free_gated_recurrent_unit_model(model);
    free_recurrent_neural_network_parameter(parameter);
    free_(optimizer);
    return success;
}

int main(void) {
    if (!test_gru_constructor_and_base_access()) {
        return 1;
    }
    return 0;
}
