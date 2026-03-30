#include "BorrowedFunctionProxy.h"

#include "Memory/Memory.h"

typedef struct borrowed_function_proxy {
    Function function;
    const Function* target;
} Borrowed_function_proxy;

static Tensor_ptr calculate_borrowed_function_proxy(const void* function, const Tensor* tensor) {
    const Borrowed_function_proxy* proxy = function;
    if (proxy == NULL || proxy->target == NULL || proxy->target->calculate == NULL) {
        return NULL;
    }
    return proxy->target->calculate(proxy->target, tensor);
}

static Tensor_ptr derivative_borrowed_function_proxy(const void* function, const Tensor* tensor, const Tensor* backward) {
    const Borrowed_function_proxy* proxy = function;
    if (proxy == NULL || proxy->target == NULL || proxy->target->derivative == NULL) {
        return NULL;
    }
    return proxy->target->derivative(proxy->target, tensor, backward);
}

Function* create_borrowed_function_proxy(const Function* target) {
    Borrowed_function_proxy* proxy;
    if (target == NULL) {
        return NULL;
    }
    proxy = malloc_(sizeof(Borrowed_function_proxy));
    if (proxy == NULL) {
        return NULL;
    }
    proxy->target = target;
    proxy->function.function_type = target->function_type;
    proxy->function.calculate = calculate_borrowed_function_proxy;
    proxy->function.derivative = derivative_borrowed_function_proxy;
    return &proxy->function;
}

void free_borrowed_function_proxy(Function* proxy) {
    if (proxy == NULL) {
        return;
    }
    free_(proxy);
}
