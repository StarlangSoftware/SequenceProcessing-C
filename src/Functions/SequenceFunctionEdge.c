#include "SequenceFunctionEdge.h"

#include <stddef.h>

/*
 * Narrow local declaration of the graph hook already exported by
 * ComputationalGraph-C. We declare only the symbol needed for this bridge here
 * instead of including ComputationalGraph.h, because that header currently
 * pulls unrelated higher-level dependencies into this local slice.
 */
extern Computational_node_ptr add_edge(Computational_graph_ptr graph,
                                       Computational_node_ptr first,
                                       void* second,
                                       bool is_biased);

Computational_node_ptr add_sequence_function_edge(Computational_graph_ptr graph,
                                                  Computational_node_ptr input,
                                                  Function* function,
                                                  bool is_biased) {
    if (graph == NULL || input == NULL || function == NULL) {
        return NULL;
    }
    return add_edge(graph, input, function, is_biased);
}
