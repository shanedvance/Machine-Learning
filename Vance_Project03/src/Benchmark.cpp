//
// Created by Shane on 5/13/2018.
//

#include "Benchmark.h"

Benchmark::Benchmark() = default;

Benchmark::~Benchmark() = default;

int Benchmark::fitnessFn(int x, int y)
{
    return x - 2 * y;
}