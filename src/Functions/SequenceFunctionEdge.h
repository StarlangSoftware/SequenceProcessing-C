#ifndef SEQUENCE_PROCESSING_SEQUENCE_FUNCTION_EDGE_H
#define SEQUENCE_PROCESSING_SEQUENCE_FUNCTION_EDGE_H

#include <stdbool.h>
#include "Function/Function.h"

typedef struct computational_graph Computational_graph;
typedef Computational_graph* Computational_graph_ptr;

typedef struct computational_node Computational_node;
typedef Computational_node* Computational_node_ptr;

/*
 * Minimal local compatibility helper for Java-style function-node insertion.
 *
 * This is intentionally narrower than Java's FunctionNode abstraction. It
 * only bridges the "attach this function as a child of the first input node"
 * behavior used by the current SequenceProcessing functions.
 *
 * Ownership:
 * - on success, ownership of `function` transfers to the returned graph node
 *   and is released through the graph/node lifecycle
 * - on failure, ownership remains with the caller
 */
Computational_node_ptr add_sequence_function_edge(Computational_graph_ptr graph,
                                                  Computational_node_ptr input,
                                                  Function* function,
                                                  bool is_biased);

#endif
