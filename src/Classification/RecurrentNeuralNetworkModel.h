#ifndef SEQUENCE_PROCESSING_RECURRENT_NEURAL_NETWORK_MODEL_H
#define SEQUENCE_PROCESSING_RECURRENT_NEURAL_NETWORK_MODEL_H

#include "Parameters/RecurrentNeuralNetworkParameter.h"
#include "Tensor.h"

typedef struct array_list Array_list;
typedef Array_list* Array_list_ptr;

typedef struct computational_node Computational_node;
typedef Computational_node* Computational_node_ptr;

typedef struct recurrent_neural_network_model Recurrent_neural_network_model;
typedef Recurrent_neural_network_model* Recurrent_neural_network_model_ptr;

struct recurrent_neural_network_model {
    /*
     * Borrowed parameter object. The model does not own or free this pointer.
     *
     * Full Java train/test parity is deferred, but the parameter reference is
     * kept now because the Java model stores it as inherited state.
     */
    Recurrent_neural_network_parameter_ptr parameters;

    /*
     * Borrowed primitive copied from the Java constructor argument.
     */
    int word_embedding_length;

    /*
     * Owned input-node list and node objects for the current helper-layer
     * slice. This mirrors the Java `inputNodes` role used by
     * createInputTensors(), but does not yet embed the full inherited
     * ComputationalGraph-C object.
     *
     * Layout assumption for this slice:
     * - entries [0, size - 2] are sequence-step input nodes
     * - entry   [size - 1] is the class-label node
     */
    Array_list_ptr input_nodes;

    /*
     * Owned Switch_function list and objects, aligned one-to-one with the
     * non-class-label input nodes only.
     */
    Array_list_ptr switches;
};

/*
 * Ownership:
 * - borrowed: `parameters`
 * - owned: model struct, input-node list and node objects, switch list and
 *          switch objects created through this slice's setup helpers
 */
Recurrent_neural_network_model_ptr create_recurrent_neural_network_model(Recurrent_neural_network_parameter_ptr parameters,
                                                                         int word_embedding_length);

void free_recurrent_neural_network_model(Recurrent_neural_network_model_ptr model);

/*
 * Appends one sequence input slot and its matching Switch. The returned node is
 * owned by the model and is borrowed by the caller.
 */
Computational_node_ptr recurrent_neural_network_model_add_time_step_input(Recurrent_neural_network_model_ptr model);

/*
 * Appends the final class-label input slot. The returned node is owned by the
 * model and is borrowed by the caller.
 */
Computational_node_ptr recurrent_neural_network_model_add_class_label_input(Recurrent_neural_network_model_ptr model);

/*
 * Java helper parity for createInputTensors(Tensor instance).
 *
 * Preconditions for the current local slice:
 * - `input_nodes` must already contain at least one class-label node
 * - `switches->size == input_nodes->size - 1`
 *
 * Returns an owned ArrayList of owned integer entries. The caller must free the
 * list with `free_array_list(result, free_)`.
 */
Array_list_ptr recurrent_neural_network_model_create_input_tensors(Recurrent_neural_network_model_ptr model,
                                                                   const Tensor* instance);

/*
 * Java helper parity for findTimeStep(ArrayList<Tensor> trainSet).
 */
int recurrent_neural_network_model_find_time_step(const Recurrent_neural_network_model* model,
                                                  const Array_list* train_set);

/*
 * Java helper parity for getOutputValue(ComputationalNode outputNode).
 *
 * This mirrors the Java implementation exactly, including its `Double.MIN_VALUE`
 * initialization behavior when scanning each row for the argmax.
 *
 * Returns an owned ArrayList of owned double entries. The caller must free the
 * list with `free_array_list(result, free_)`.
 */
Array_list_ptr recurrent_neural_network_model_get_output_value(const Computational_node* output_node);

#endif
