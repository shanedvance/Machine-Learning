//
// Created by Shane on 5/13/2018.
//

#ifndef GENETICALGORITHM_BENCHMARK_H
#define GENETICALGORITHM_BENCHMARK_H


class Benchmark
{
public:

    typedef int (Benchmark::*Fitness)(int, int);

    Benchmark();
    ~Benchmark();
    int fitnessFn(int, int);
};


#endif //GENETICALGORITHM_BENCHMARK_H
