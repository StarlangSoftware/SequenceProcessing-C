#include "GatedRecurrentUnitModel.h"

#include "BorrowedFunctionProxy.h"
#include "JavaRandomCompat.h"

#include "Function/Negation.h"
#include "Function/SoftMax.h"
#include "Function/Tanh.h"
#include "Functions/AdditionByConstant.h"
#include "Functions/RemoveBias.h"
#include "Functions/Switch.h"
#include "Memory/Memory.h"
#include "Node/MultiplicationNode.h"

static void free_weight_list(Array_list_ptr list) {
    if (list != NULL) {
        free_array_list(list, (void (*)(void*)) free_multiplication_node);
    }
}

static Computational_node_ptr add_owned_function_edge(Recurrent_neural_network_model_ptr base_model,
                                                      Computational_node_ptr first,
                                                      Function* function,
                                                      bool is_biased) {
    Computational_node_ptr result;
    result = recurrent_neural_network_model_add_edge(base_model, first, function, is_biased);
    if (result == NULL) {
        free_(function);
    }
    return result;
}

static Computational_node_ptr add_borrowed_function_edge(Recurrent_neural_network_model_ptr base_model,
                                                         Computational_node_ptr first,
                                                         const Function* function,
                                                         bool is_biased) {
    Function* proxy = create_borrowed_function_proxy(function);
    Computational_node_ptr result;
    if (proxy == NULL) {
        return NULL;
    }
    result = recurrent_neural_network_model_add_edge(base_model, first, proxy, is_biased);
    if (result == NULL) {
        free_borrowed_function_proxy(proxy);
    }
    return result;
}

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

bool gated_recurrent_unit_model_train(Gated_recurrent_unit_model_ptr model, Array_list_ptr train_set) {
    Recurrent_neural_network_model_ptr base_model;
    Recurrent_neural_network_parameter_ptr parameters;
    Java_random_compat_ptr random;
    Array_list_ptr weights = NULL;
    Array_list_ptr recurrent_weights = NULL;
    Array_list_ptr current_old_layers = NULL;
    Array_list_ptr output_nodes = NULL;
    int time_step;
    int current_length;
    int i;
    int k;
    bool graph_started = false;
    bool success = false;
    if (model == NULL || train_set == NULL) {
        return false;
    }
    base_model = model->base_model;
    parameters = (base_model != NULL) ? base_model->parameters : NULL;
    if (base_model == NULL || parameters == NULL) {
        return false;
    }
    if (base_model->input_nodes == NULL || base_model->switches == NULL ||
        base_model->input_nodes->size != 0 || base_model->switches->size != 0) {
        /*
         * This minimal slice supports building the recurrent graph once on a
         * fresh model instance, mirroring the current constructor->train flow.
         */
        return false;
    }
    random = create_java_random_compat(parameters->neural_network_parameter.seed);
    if (random == NULL) {
        return false;
    }
    time_step = recurrent_neural_network_model_find_time_step(base_model, train_set);
    if (time_step < 0) {
        free_java_random_compat(random);
        return false;
    }
    weights = create_array_list();
    recurrent_weights = create_array_list();
    current_old_layers = create_array_list();
    output_nodes = create_array_list();
    if (weights == NULL || recurrent_weights == NULL || current_old_layers == NULL || output_nodes == NULL) {
        if (!graph_started) {
            free_weight_list(weights);
            free_weight_list(recurrent_weights);
        } else {
            free_array_list(weights, NULL);
            free_array_list(recurrent_weights, NULL);
        }
        if (current_old_layers != NULL) {
            free_array_list(current_old_layers, NULL);
        }
        if (output_nodes != NULL) {
            free_array_list(output_nodes, NULL);
        }
        free_java_random_compat(random);
        return false;
    }
    current_length = base_model->word_embedding_length + 1;
    for (i = 0; i < recurrent_neural_network_parameter_size(parameters); i++) {
        int hidden_layer = recurrent_neural_network_parameter_get_hidden_layer(parameters, i);
        int j;
        for (j = 0; j < 3; j++) {
            Multiplication_node_ptr weight = recurrent_neural_network_model_create_weight_node(base_model, current_length, hidden_layer, random);
            Multiplication_node_ptr recurrent_weight = recurrent_neural_network_model_create_weight_node(base_model, hidden_layer, hidden_layer, random);
            if (weight == NULL || recurrent_weight == NULL) {
                if (weight != NULL) {
                    free_multiplication_node(weight);
                }
                if (recurrent_weight != NULL) {
                    free_multiplication_node(recurrent_weight);
                }
                goto cleanup;
            }
            array_list_add(weights, weight);
            array_list_add(recurrent_weights, recurrent_weight);
        }
        current_length = hidden_layer + 1;
    }
    {
        Multiplication_node_ptr output_weight = recurrent_neural_network_model_create_weight_node(
                base_model,
                current_length,
                recurrent_neural_network_parameter_get_class_label_size(parameters),
                random);
        if (output_weight == NULL) {
            goto cleanup;
        }
        array_list_add(weights, output_weight);
    }
    for (k = 0; k < time_step; k++) {
        Array_list_ptr new_old_layers = create_array_list();
        Computational_node_ptr input = recurrent_neural_network_model_add_time_step_input(base_model);
        Computational_node_ptr current = input;
        Switch_function_ptr switch_function;
        if (new_old_layers == NULL || input == NULL) {
            if (new_old_layers != NULL) {
                free_array_list(new_old_layers, NULL);
            }
            goto cleanup;
        }
        switch_function = array_list_get(base_model->switches, k);
        for (i = 0; i < recurrent_neural_network_parameter_size(parameters); i++) {
            Computational_node_ptr aw;
            Computational_node_ptr a_function;
            graph_started = true;
            if (!is_array_list_empty(current_old_layers)) {
                Computational_node_ptr o_without_bias;
                Computational_node_ptr ou;
                Computational_node_ptr aw_ou;
                Computational_node_ptr zt;
                Computational_node_ptr rt;
                Computational_node_ptr rt_ht1;
                Computational_node_ptr h_temp;
                Computational_node_ptr minus_zt;
                Computational_node_ptr one_minus_zt;
                aw = recurrent_neural_network_model_add_multiplication_edge(
                        base_model, current, array_list_get(weights, (i * 3)), false);
                o_without_bias = add_owned_function_edge(base_model, array_list_get(current_old_layers, i),
                                                         (Function*) create_remove_bias(), false);
                ou = recurrent_neural_network_model_add_multiplication_edge(
                        base_model, o_without_bias, array_list_get(recurrent_weights, (i * 3)), false);
                aw_ou = recurrent_neural_network_model_add_addition_edge(base_model, aw, ou, false);
                zt = add_borrowed_function_edge(base_model, aw_ou,
                                                recurrent_neural_network_parameter_get_activation_function(parameters, (i * 2)),
                                                false);
                aw = recurrent_neural_network_model_add_multiplication_edge(
                        base_model, current, array_list_get(weights, (i * 3) + 1), false);
                ou = recurrent_neural_network_model_add_multiplication_edge(
                        base_model, o_without_bias, array_list_get(recurrent_weights, (i * 3) + 1), false);
                aw_ou = recurrent_neural_network_model_add_addition_edge(base_model, aw, ou, false);
                rt = add_borrowed_function_edge(base_model, aw_ou,
                                                recurrent_neural_network_parameter_get_activation_function(parameters, (i * 2) + 1),
                                                false);
                aw = recurrent_neural_network_model_add_multiplication_edge(
                        base_model, current, array_list_get(weights, (i * 3) + 2), false);
                rt_ht1 = recurrent_neural_network_model_add_hadamard_edge(base_model, rt, o_without_bias, false);
                ou = recurrent_neural_network_model_add_multiplication_edge(
                        base_model, rt_ht1, array_list_get(recurrent_weights, (i * 3) + 2), false);
                aw_ou = recurrent_neural_network_model_add_addition_edge(base_model, aw, ou, false);
                h_temp = add_owned_function_edge(base_model, aw_ou, (Function*) create_tanh(), false);
                minus_zt = add_owned_function_edge(base_model, zt, (Function*) create_negation(), false);
                one_minus_zt = add_owned_function_edge(base_model, minus_zt, (Function*) create_addition_by_constant(1.0), false);
                aw = recurrent_neural_network_model_add_hadamard_edge(base_model, one_minus_zt, o_without_bias, false);
                ou = recurrent_neural_network_model_add_hadamard_edge(base_model, h_temp, zt, false);
                a_function = recurrent_neural_network_model_add_addition_edge(base_model, aw, ou, true);
            } else {
                Computational_node_ptr zt;
                Computational_node_ptr h_temp;
                aw = recurrent_neural_network_model_add_multiplication_edge(
                        base_model, current, array_list_get(weights, (i * 3)), false);
                zt = add_borrowed_function_edge(base_model, aw,
                                                recurrent_neural_network_parameter_get_activation_function(parameters, (i * 2)),
                                                false);
                aw = recurrent_neural_network_model_add_multiplication_edge(
                        base_model, current, array_list_get(weights, (i * 3) + 2), false);
                h_temp = add_owned_function_edge(base_model, aw, (Function*) create_tanh(), false);
                a_function = recurrent_neural_network_model_add_hadamard_edge(base_model, zt, h_temp, true);
            }
            if (a_function == NULL) {
                free_array_list(new_old_layers, NULL);
                goto cleanup;
            }
            current = a_function;
            array_list_add(new_old_layers, a_function);
        }
        if (current_old_layers != NULL) {
            free_array_list(current_old_layers, NULL);
        }
        current_old_layers = new_old_layers;
        {
            Computational_node_ptr node = recurrent_neural_network_model_add_multiplication_edge(
                    base_model, current, array_list_get(weights, weights->size - 1), false);
            Computational_node_ptr switched_output;
            Function* switch_proxy = create_borrowed_function_proxy((const Function*) switch_function);
            if (switch_proxy == NULL) {
                goto cleanup;
            }
            switched_output = recurrent_neural_network_model_add_edge(base_model, node, switch_proxy, false);
            if (switched_output == NULL) {
                free_borrowed_function_proxy(switch_proxy);
                goto cleanup;
            }
            array_list_add(output_nodes, switched_output);
        }
    }
    {
        Concatenated_node_ptr concatenated_node = recurrent_neural_network_model_concat_edges(base_model, output_nodes, 0);
        Computational_node_ptr output_node;
        if (concatenated_node == NULL) {
            goto cleanup;
        }
        output_node = add_owned_function_edge(base_model, (Computational_node_ptr) concatenated_node, (Function*) create_softmax(), false);
        if (output_node == NULL) {
            goto cleanup;
        }
        recurrent_neural_network_model_set_output_node(base_model, output_node);
    }
    if (recurrent_neural_network_model_add_class_label_input(base_model) == NULL) {
        goto cleanup;
    }
    success = recurrent_neural_network_model_train_with_random(base_model, train_set, random);

cleanup:
    if (!graph_started) {
        free_weight_list(weights);
        free_weight_list(recurrent_weights);
    } else {
        if (weights != NULL) {
            free_array_list(weights, NULL);
        }
        if (recurrent_weights != NULL) {
            free_array_list(recurrent_weights, NULL);
        }
    }
    if (current_old_layers != NULL) {
        free_array_list(current_old_layers, NULL);
    }
    if (output_nodes != NULL) {
        free_array_list(output_nodes, NULL);
    }
    free_java_random_compat(random);
    return success;
}
