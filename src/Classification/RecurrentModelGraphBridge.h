#ifndef SEQUENCE_PROCESSING_RECURRENT_MODEL_GRAPH_BRIDGE_H
#define SEQUENCE_PROCESSING_RECURRENT_MODEL_GRAPH_BRIDGE_H

#include "Function/Function.h"

#include <stdbool.h>

typedef struct array_list Array_list;
typedef Array_list* Array_list_ptr;

typedef struct computational_graph Computational_graph;
typedef Computational_graph* Computational_graph_ptr;

typedef struct computational_node Computational_node;
typedef Computational_node* Computational_node_ptr;

typedef struct multiplication_node Multiplication_node;
typedef Multiplication_node* Multiplication_node_ptr;

typedef struct concatenated_node Concatenated_node;
typedef Concatenated_node* Concatenated_node_ptr;

typedef struct recurrent_model_graph_bridge Recurrent_model_graph_bridge;
typedef Recurrent_model_graph_bridge* Recurrent_model_graph_bridge_ptr;

typedef Array_list_ptr (*Recurrent_output_extractor)(const Computational_node* output_node);

/*
 * Minimal local recurrent/model graph bridge.
 *
 * Ownership:
 * - owned: bridge struct, underlying computational graph, registered input-node
 *   list container
 * - borrowed: output extractor callback
 *
 * Input-node ownership:
 * - input nodes created through this bridge are owned by the bridge until they
 *   become part of the graph topology
 * - connected input nodes are later released by the underlying graph teardown
 * - unconnected input nodes are released directly by the bridge teardown
 */
Recurrent_model_graph_bridge_ptr create_recurrent_model_graph_bridge(Recurrent_output_extractor output_extractor);

void free_recurrent_model_graph_bridge(Recurrent_model_graph_bridge_ptr bridge);

Array_list_ptr recurrent_model_graph_bridge_get_input_nodes(const Recurrent_model_graph_bridge* bridge);

Computational_graph_ptr recurrent_model_graph_bridge_get_graph(const Recurrent_model_graph_bridge* bridge);

Computational_node_ptr recurrent_model_graph_bridge_add_input_node(Recurrent_model_graph_bridge_ptr bridge,
                                                                   bool learnable,
                                                                   bool is_biased);

Computational_node_ptr recurrent_model_graph_bridge_add_function_edge(Recurrent_model_graph_bridge_ptr bridge,
                                                                      Computational_node_ptr first,
                                                                      Function* function,
                                                                      bool is_biased);

Computational_node_ptr recurrent_model_graph_bridge_add_multiplication_edge(Recurrent_model_graph_bridge_ptr bridge,
                                                                            Computational_node_ptr first,
                                                                            Multiplication_node_ptr second,
                                                                            bool is_biased);

Computational_node_ptr recurrent_model_graph_bridge_add_hadamard_edge(Recurrent_model_graph_bridge_ptr bridge,
                                                                      Computational_node_ptr first,
                                                                      Computational_node_ptr second,
                                                                      bool is_biased);

Computational_node_ptr recurrent_model_graph_bridge_add_addition_edge(Recurrent_model_graph_bridge_ptr bridge,
                                                                      Computational_node_ptr first,
                                                                      Computational_node_ptr second,
                                                                      bool is_biased);

Concatenated_node_ptr recurrent_model_graph_bridge_concat_edges(Recurrent_model_graph_bridge_ptr bridge,
                                                                Array_list_ptr nodes,
                                                                int dimension);

void recurrent_model_graph_bridge_set_output_node(Recurrent_model_graph_bridge_ptr bridge,
                                                  Computational_node_ptr output_node);

Array_list_ptr recurrent_model_graph_bridge_forward(Recurrent_model_graph_bridge_ptr bridge);

Array_list_ptr recurrent_model_graph_bridge_predict(Recurrent_model_graph_bridge_ptr bridge);

#endif
