#ifndef SEQUENCE_PROCESSING_TRANSFORMER_H
#define SEQUENCE_PROCESSING_TRANSFORMER_H

#include "Parameters/TransformerParameter.h"
#include "Tensor.h"

typedef struct array_list Array_list;
typedef Array_list* Array_list_ptr;

typedef struct computational_node Computational_node;
typedef Computational_node* Computational_node_ptr;

typedef struct vectorized_dictionary Vectorized_dictionary;
typedef Vectorized_dictionary* Vectorized_dictionary_ptr;

typedef struct vector Vector;
typedef Vector* Vector_ptr;

typedef struct transformer_model Transformer_model;
typedef Transformer_model* Transformer_model_ptr;

typedef struct transformer_packed_inputs Transformer_packed_inputs;
typedef Transformer_packed_inputs* Transformer_packed_inputs_ptr;

/*
 * Constructor shell matching the Java Transformer constructor at the minimal
 * currently grounded level.
 *
 * Ownership:
 * - borrowed: `parameters`, `dictionary`
 * - owned: Transformer shell struct and its local graph bridge
 */
Transformer_model_ptr create_transformer_model(Transformer_parameter_ptr parameters,
                                               Vectorized_dictionary_ptr dictionary);

void free_transformer_model(Transformer_model_ptr model);

Vectorized_dictionary_ptr transformer_model_get_dictionary(const Transformer_model* model);

int transformer_model_get_start_index(const Transformer_model* model);

int transformer_model_get_end_index(const Transformer_model* model);

/*
 * Borrowed alias of the locally owned graph-bridge input-node list.
 *
 * The returned list must not be freed by the caller.
 */
Array_list_ptr transformer_model_get_input_nodes(const Transformer_model* model);

/*
 * Lifecycle flag for the staged Transformer graph path.
 *
 * It becomes true once graph construction has begun or has successfully
 * claimed the instance for staged graph-backed training. It does not claim
 * full Java loss-node forward parity or `test(...)` parity.
 */
bool transformer_model_is_graph_initialized(const Transformer_model* model);

/*
 * Java helper parity for positionalEncoding(Tensor tensor, int
 * wordEmbeddingLength).
 *
 * Returns a newly allocated tensor owned by the caller.
 */
Tensor_ptr transformer_model_positional_encoding(const Transformer_model* model,
                                                 const Tensor* tensor,
                                                 int word_embedding_length);

/*
 * Local helper matching the pure data-transform portion of Java
 * createInputTensors(...), before any graph-node mutation.
 *
 * The returned pack owns:
 * - encoder input tensor
 * - decoder input tensor
 * - class-label list and its integer entries
 *
 * Getter accessors below return borrowed views into this owned pack.
 */
Transformer_packed_inputs_ptr transformer_model_create_packed_inputs(const Transformer_model* model,
                                                                    const Tensor* instance,
                                                                    int word_embedding_length);

void free_transformer_packed_inputs(Transformer_packed_inputs_ptr packed_inputs);

const Tensor* transformer_packed_inputs_get_encoder_input(const Transformer_packed_inputs* packed_inputs);

const Tensor* transformer_packed_inputs_get_decoder_input(const Transformer_packed_inputs* packed_inputs);

Array_list_ptr transformer_packed_inputs_get_class_labels(const Transformer_packed_inputs* packed_inputs);

/*
 * Java helper parity for setInputNode(int bound, Vector vector,
 * ComputationalNode node).
 *
 * This appends one positionally encoded token vector to the node value and then
 * reshapes the node value to `[bound, vector->size]`.
 *
 * Ownership:
 * - borrowed: `vector`, `node`
 * - owned by node after success: updated tensor value
 */
bool transformer_model_set_input_node(const Transformer_model* model,
                                      int bound,
                                      const Vector* vector,
                                      Computational_node_ptr node);

/*
 * Initializes the minimal Transformer graph input shell:
 * - encoder input node
 * - decoder input node
 *
 * Returns false if the model is invalid or if the existing input-node layout is
 * incompatible with this staged slice.
 */
bool transformer_model_initialize_graph_inputs(Transformer_model_ptr model);

/*
 * Applies pre-packed encoder/decoder tensors to the first two Transformer input
 * nodes. Input tensors are cloned into graph-owned node values.
 */
bool transformer_model_apply_packed_inputs(Transformer_model_ptr model,
                                           const Transformer_packed_inputs* packed_inputs);

/*
 * Adds the trailing class-label input node used later by training. If it
 * already exists, returns the existing borrowed node.
 */
Computational_node_ptr transformer_model_add_class_label_input(Transformer_model_ptr model);

/*
 * Java getOutputValue(...) parity helper for later graph-forward integration.
 *
 * Returns an owned ArrayList of owned double entries.
 */
Array_list_ptr transformer_model_get_output_value(const Computational_node* output_node);

/*
 * Builds the staged Transformer graph shell/body corresponding to the current
 * Java train(...) graph construction, but without claiming multi-input loss
 * node forward-value parity or full `test(...)` parity.
 */
bool transformer_model_build_graph(Transformer_model_ptr model);

/*
 * Staged Transformer train port using the already-built graph body, packed
 * encoder/decoder inputs, one-hot class-label node values, and class-index
 * backprop.
 *
 * This staged path deliberately does not claim loss-node forward-value parity,
 * and it reuses the already-built graph on later calls instead of rebuilding it.
 */
bool transformer_model_train(Transformer_model_ptr model, Array_list_ptr train_set);

#endif
