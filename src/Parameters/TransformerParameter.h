#ifndef SEQUENCE_PROCESSING_TRANSFORMER_PARAMETER_H
#define SEQUENCE_PROCESSING_TRANSFORMER_PARAMETER_H

#include "NeuralNetworkParameter.h"
#include "Function/Function.h"

typedef struct array_list Array_list;
typedef Array_list* Array_list_ptr;

typedef struct transformer_parameter Transformer_parameter;
typedef Transformer_parameter* Transformer_parameter_ptr;

struct transformer_parameter {
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

    int l;
    int n;
    int v;
    double epsilon;

    /*
     * Owned copies of primitive parameter arrays.
     */
    Array_list_ptr input_hidden_layers;
    Array_list_ptr output_hidden_layers;
    Array_list_ptr gamma_input_values;
    Array_list_ptr gamma_output_values;
    Array_list_ptr beta_input_values;
    Array_list_ptr beta_output_values;

    /*
     * Owned shallow copies of incoming activation-function pointer lists.
     * The list storage is owned, the Function objects are borrowed.
     */
    Array_list_ptr input_functions;
    Array_list_ptr output_functions;
};

/*
 * Mirrors the Java constructor
 * TransformerParameter(seed, epoch, optimizer, initialization, loss,
 *                      wordEmbeddingLength, multiHeadAttentionLength,
 *                      vocabularyLength, epsilon,
 *                      inputHiddenLayers, outputHiddenLayers,
 *                      inputActivationFunctions, outputActivationFunctions,
 *                      gammaInputValues, gammaOutputValues,
 *                      betaInputValues, betaOutputValues).
 *
 * Ownership:
 * - borrowed: optimizer, loss, Function objects referenced by input/output
 *   activation-function lists
 * - owned: copied integer/double list contents, copied function-pointer list
 *   containers, this parameter struct
 */
Transformer_parameter_ptr create_transformer_parameter(int seed,
                                                       int epoch,
                                                       Optimizer_ptr optimizer,
                                                       Initialization initialization,
                                                       Function* loss,
                                                       int word_embedding_length,
                                                       int multi_head_attention_length,
                                                       int vocabulary_length,
                                                       double epsilon,
                                                       const Array_list* input_hidden_layers,
                                                       const Array_list* output_hidden_layers,
                                                       const Array_list* input_activation_functions,
                                                       const Array_list* output_activation_functions,
                                                       const Array_list* gamma_input_values,
                                                       const Array_list* gamma_output_values,
                                                       const Array_list* beta_input_values,
                                                       const Array_list* beta_output_values);

void free_transformer_parameter(Transformer_parameter_ptr parameter);

double transformer_parameter_get_gamma_input_value(const Transformer_parameter* parameter, int index);

double transformer_parameter_get_gamma_output_value(const Transformer_parameter* parameter, int index);

double transformer_parameter_get_beta_input_value(const Transformer_parameter* parameter, int index);

double transformer_parameter_get_beta_output_value(const Transformer_parameter* parameter, int index);

double transformer_parameter_get_epsilon(const Transformer_parameter* parameter);

int transformer_parameter_get_dk(const Transformer_parameter* parameter);

int transformer_parameter_get_l(const Transformer_parameter* parameter);

int transformer_parameter_get_n(const Transformer_parameter* parameter);

int transformer_parameter_get_v(const Transformer_parameter* parameter);

int transformer_parameter_get_input_hidden_layer(const Transformer_parameter* parameter, int index);

int transformer_parameter_get_output_hidden_layer(const Transformer_parameter* parameter, int index);

Function* transformer_parameter_get_input_activation_function(const Transformer_parameter* parameter, int index);

Function* transformer_parameter_get_output_activation_function(const Transformer_parameter* parameter, int index);

int transformer_parameter_get_input_size(const Transformer_parameter* parameter);

int transformer_parameter_get_output_size(const Transformer_parameter* parameter);

#endif
