#include <iostream>
#include "GeneticAlgorithms.h"
#include "Benchmark.h"

using namespace std;

int main()
{

    Benchmark bm;
    GeneticAlgorithms ga(bm);

    Population p = ga.simpleGA(&Benchmark::fitnessFn, 100, 2, Range(-512, 512), 100, 0.5, 0.01);

    cout << endl << "Min. value: " << p.cost << endl;
    cout << "[x: " << p.individual.at(0) << ", y: " << p.individual.at(1) << "]" << endl;

    return 0;
}