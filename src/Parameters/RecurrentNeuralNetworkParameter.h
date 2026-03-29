#ifndef SEQUENCE_PROCESSING_RECURRENT_NEURAL_NETWORK_PARAMETER_H
#define SEQUENCE_PROCESSING_RECURRENT_NEURAL_NETWORK_PARAMETER_H

#include "NeuralNetworkParameter.h"
#include "Function/Function.h"

typedef struct array_list Array_list;
typedef Array_list* Array_list_ptr;

typedef struct recurrent_neural_network_parameter Recurrent_neural_network_parameter;
typedef Recurrent_neural_network_parameter* Recurrent_neural_network_parameter_ptr;

struct recurrent_neural_network_parameter {
    /*
     * Value copy of the common neural-network parameter fields provided by
     * ComputationalGraph-C.
     */
    Neural_network_parameter neural_network_parameter;

    /*
     * Borrowed function pointer matching the Java constructor argument.
     * Not freed by this struct.
     */
    Function* loss_function;

    /*
     * Stored for Java-constructor parity. Java passes batch size 1 through the
     * parent constructor for this class.
     */
    int batch_size;

    /*
     * Owned copies of the incoming hidden-layer sizes.
     */
    Array_list_ptr hidden_layers;

    /*
     * Owned shallow copy of incoming activation-function pointers.
     * The list storage is owned, the Function objects are borrowed.
     */
    Array_list_ptr functions;

    int class_label_size;
};

/*
 * Mirrors the Java constructor
 * RecurrentNeuralNetworkParameter(seed, epoch, optimizer, initialization,
 *                                 loss, hiddenLayers, functions, classLabelSize).
 *
 * Ownership:
 * - borrowed: optimizer, loss, Function objects referenced by functions
 * - owned: copied hidden-layer values, copied function-pointer list container,
 *          this parameter struct
 */
Recurrent_neural_network_parameter_ptr create_recurrent_neural_network_parameter(int seed,
                                                                                 int epoch,
                                                                                 Optimizer_ptr optimizer,
                                                                                 Initialization initialization,
                                                                                 Function* loss,
                                                                                 const Array_list* hidden_layers,
                                                                                 const Array_list* functions,
                                                                                 int class_label_size);

void free_recurrent_neural_network_parameter(Recurrent_neural_network_parameter_ptr parameter);

int recurrent_neural_network_parameter_size(const Recurrent_neural_network_parameter* parameter);

int recurrent_neural_network_parameter_get_class_label_size(const Recurrent_neural_network_parameter* parameter);

Function* recurrent_neural_network_parameter_get_activation_function(const Recurrent_neural_network_parameter* parameter,
                                                                    int index);

int recurrent_neural_network_parameter_get_hidden_layer(const Recurrent_neural_network_parameter* parameter, int index);

#endif
