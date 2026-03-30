#include "RecurrentModelGraphBridge.h"

#include "ArrayList.h"
#include "HashMap/HashMap.h"
#include "Memory/Memory.h"
#include "NeuralNetworkParameter.h"
#include "Node/ComputationalNode.h"

#include <stddef.h>

typedef void* Classification_performance_ptr;

/*
 * Local mirror of the ComputationalGraph-C struct layout. This stays in the
 * .c file because the sibling public header currently pulls a missing
 * ClassificationPerformance dependency into the local build.
 */
struct computational_graph {
    Hash_map_ptr node_map;
    Hash_map_ptr reverse_node_map;
    Array_list_ptr input_nodes;
    Computational_node_ptr output_node;
    void (*train)(struct computational_graph* graph, Array_list_ptr train_set, Neural_network_parameter_ptr parameters);
    Classification_performance_ptr (*test)(struct computational_graph* graph, Array_list_ptr test_set);
    Array_list_ptr (*get_class_labels)(Computational_node_ptr output_node);
};

struct recurrent_model_graph_bridge {
    Computational_graph_ptr graph;
    Array_list_ptr registered_input_nodes;
    Recurrent_output_extractor output_extractor;
};

extern Computational_graph_ptr create_computational_graph(void);
extern void free_computational_graph(Computational_graph_ptr graph);
extern Computational_node_ptr add_edge(Computational_graph_ptr graph,
                                       Computational_node_ptr first,
                                       void* second,
                                       bool is_biased);
extern Computational_node_ptr add_multiplication_edge(Computational_graph_ptr graph,
                                                      Computational_node_ptr first,
                                                      Multiplication_node_ptr second,
                                                      bool is_biased);
extern Computational_node_ptr add_edge_with_hadamard(Computational_graph_ptr graph,
                                                     Computational_node_ptr first,
                                                     Computational_node_ptr second,
                                                     bool is_biased,
                                                     bool is_hadamard);
extern Computational_node_ptr add_addition_edge(Computational_graph_ptr graph,
                                                Computational_node_ptr first,
                                                Computational_node_ptr second,
                                                bool is_biased);
extern Concatenated_node_ptr concat_edges(Computational_graph_ptr graph, Array_list_ptr nodes, int dimension);
extern Array_list_ptr forward_calculation(Computational_graph_ptr graph);
extern Array_list_ptr predict_by_computational_graph(Computational_graph_ptr graph);
extern void back_propagation(Computational_graph_ptr graph, Optimizer_ptr optimizer, const int* class_label_index);

Recurrent_model_graph_bridge_ptr create_recurrent_model_graph_bridge(Recurrent_output_extractor output_extractor) {
    Recurrent_model_graph_bridge_ptr result = malloc_(sizeof(Recurrent_model_graph_bridge));
    if (result == NULL) {
        return NULL;
    }
    result->graph = create_computational_graph();
    result->registered_input_nodes = create_array_list();
    result->output_extractor = output_extractor;
    if (result->graph == NULL || result->registered_input_nodes == NULL) {
        if (result->graph != NULL) {
            free_computational_graph(result->graph);
        }
        if (result->registered_input_nodes != NULL) {
            free_array_list(result->registered_input_nodes, NULL);
        }
        free_(result);
        return NULL;
    }
    result->graph->get_class_labels = (Array_list_ptr (*)(Computational_node_ptr)) output_extractor;
    return result;
}

void free_recurrent_model_graph_bridge(Recurrent_model_graph_bridge_ptr bridge) {
    int i;
    if (bridge == NULL) {
        return;
    }
    if (bridge->graph != NULL && bridge->registered_input_nodes != NULL) {
        for (i = 0; i < bridge->registered_input_nodes->size; i++) {
            Computational_node_ptr node = array_list_get(bridge->registered_input_nodes, i);
            if (node != NULL &&
                !hash_map_contains(bridge->graph->node_map, node) &&
                bridge->graph->output_node != node) {
                free_computational_node(node);
            }
        }
    }
    if (bridge->registered_input_nodes != NULL) {
        free_array_list(bridge->registered_input_nodes, NULL);
    }
    if (bridge->graph != NULL) {
        free_computational_graph(bridge->graph);
    }
    free_(bridge);
}

Array_list_ptr recurrent_model_graph_bridge_get_input_nodes(const Recurrent_model_graph_bridge* bridge) {
    if (bridge == NULL) {
        return NULL;
    }
    return bridge->registered_input_nodes;
}

Computational_graph_ptr recurrent_model_graph_bridge_get_graph(const Recurrent_model_graph_bridge* bridge) {
    if (bridge == NULL) {
        return NULL;
    }
    return bridge->graph;
}

Computational_node_ptr recurrent_model_graph_bridge_add_input_node(Recurrent_model_graph_bridge_ptr bridge,
                                                                   bool learnable,
                                                                   bool is_biased) {
    Computational_node_ptr node;
    if (bridge == NULL || bridge->graph == NULL) {
        return NULL;
    }
    node = create_computational_node3(learnable, is_biased);
    if (node == NULL) {
        return NULL;
    }
    array_list_add(bridge->registered_input_nodes, node);
    array_list_add(bridge->graph->input_nodes, node);
    return node;
}

Computational_node_ptr recurrent_model_graph_bridge_add_function_edge(Recurrent_model_graph_bridge_ptr bridge,
                                                                      Computational_node_ptr first,
                                                                      Function* function,
                                                                      bool is_biased) {
    if (bridge == NULL || bridge->graph == NULL || first == NULL || function == NULL) {
        return NULL;
    }
    return add_edge(bridge->graph, first, function, is_biased);
}

Computational_node_ptr recurrent_model_graph_bridge_add_multiplication_edge(Recurrent_model_graph_bridge_ptr bridge,
                                                                            Computational_node_ptr first,
                                                                            Multiplication_node_ptr second,
                                                                            bool is_biased) {
    if (bridge == NULL || bridge->graph == NULL || first == NULL || second == NULL) {
        return NULL;
    }
    return add_multiplication_edge(bridge->graph, first, second, is_biased);
}

Computational_node_ptr recurrent_model_graph_bridge_add_hadamard_edge(Recurrent_model_graph_bridge_ptr bridge,
                                                                      Computational_node_ptr first,
                                                                      Computational_node_ptr second,
                                                                      bool is_biased) {
    if (bridge == NULL || bridge->graph == NULL || first == NULL || second == NULL) {
        return NULL;
    }
    return add_edge_with_hadamard(bridge->graph, first, second, is_biased, true);
}

Computational_node_ptr recurrent_model_graph_bridge_add_addition_edge(Recurrent_model_graph_bridge_ptr bridge,
                                                                      Computational_node_ptr first,
                                                                      Computational_node_ptr second,
                                                                      bool is_biased) {
    if (bridge == NULL || bridge->graph == NULL || first == NULL || second == NULL) {
        return NULL;
    }
    return add_addition_edge(bridge->graph, first, second, is_biased);
}

Concatenated_node_ptr recurrent_model_graph_bridge_concat_edges(Recurrent_model_graph_bridge_ptr bridge,
                                                                Array_list_ptr nodes,
                                                                int dimension) {
    if (bridge == NULL || bridge->graph == NULL || nodes == NULL) {
        return NULL;
    }
    return concat_edges(bridge->graph, nodes, dimension);
}

void recurrent_model_graph_bridge_set_output_node(Recurrent_model_graph_bridge_ptr bridge,
                                                  Computational_node_ptr output_node) {
    if (bridge == NULL || bridge->graph == NULL) {
        return;
    }
    bridge->graph->output_node = output_node;
}

Array_list_ptr recurrent_model_graph_bridge_forward(Recurrent_model_graph_bridge_ptr bridge) {
    if (bridge == NULL || bridge->graph == NULL || bridge->output_extractor == NULL) {
        return NULL;
    }
    return forward_calculation(bridge->graph);
}

Array_list_ptr recurrent_model_graph_bridge_predict(Recurrent_model_graph_bridge_ptr bridge) {
    if (bridge == NULL || bridge->graph == NULL || bridge->output_extractor == NULL) {
        return NULL;
    }
    return predict_by_computational_graph(bridge->graph);
}

void recurrent_model_graph_bridge_back_propagation(Recurrent_model_graph_bridge_ptr bridge,
                                                   Optimizer_ptr optimizer,
                                                   const int* class_label_index) {
    if (bridge == NULL || bridge->graph == NULL || optimizer == NULL || class_label_index == NULL) {
        return;
    }
    back_propagation(bridge->graph, optimizer, class_label_index);
}
