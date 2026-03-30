#ifndef SEQUENCE_PROCESSING_BORROWED_FUNCTION_PROXY_H
#define SEQUENCE_PROCESSING_BORROWED_FUNCTION_PROXY_H

#include "Function/Function.h"

/*
 * Small local compatibility wrapper for borrowed Function instances.
 *
 * ComputationalGraph-C nodes free their attached `Function*` on teardown.
 * Recurrent models need to reuse borrowed parameter activations and owned
 * Switch objects across the graph, so graph nodes receive a proxy that is
 * graph-owned while the target Function remains borrowed.
 */
Function* create_borrowed_function_proxy(const Function* target);

void free_borrowed_function_proxy(Function* proxy);

#endif
