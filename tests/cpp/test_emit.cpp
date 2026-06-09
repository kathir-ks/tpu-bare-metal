// test_emit.cpp — build graphs and print emitted StableHLO (no TPU needed).
#include "graph.hpp"
#include <cstdio>

using namespace tpu;

static void banner(const char* s) { printf("\n==== %s ====\n", s); }

int main() {
    // 1. plain matmul
    {
        banner("matmul 4x4");
        Graph g;
        auto a = g.input("a", {4, 4});
        auto b = g.input("b", {4, 4});
        auto c = g.dot(a, b);
        printf("%s", g.emit({c}).c_str());
    }

    // 2. softmax-ish: row softmax over last dim of [2,3]
    {
        banner("softmax [2,3]");
        Graph g;
        auto x = g.input("x", {2, 3});
        auto m = g.stop_gradient(g.reduce_max(x, {1}, true));
        auto e = g.exp(g.sub(x, m));
        auto s = g.reduce_sum(e, {1}, true);
        auto y = g.div(e, s);
        printf("%s", g.emit({y}).c_str());
    }

    // 3. autodiff: loss = sum((a@b - t)^2), grads wrt a,b
    {
        banner("mse grad");
        Graph g;
        auto a = g.input("a", {2, 2});
        auto b = g.input("b", {2, 2});
        auto t = g.input("t", {2, 2});
        auto pred = g.dot(a, b);
        auto diff = g.sub(pred, t);
        auto loss = g.reduce_sum(g.mul(diff, diff), {0, 1});  // scalar
        auto grads = g.grad(loss, {a, b});
        printf("loss rank=%lld  ga rank=%lld gb rank=%lld\n",
               (long long)loss.rank(), (long long)grads[0].rank(),
               (long long)grads[1].rank());
        printf("%s", g.emit({loss, grads[0], grads[1]}).c_str());
    }

    printf("\nEMIT TEST DONE\n");
    return 0;
}
