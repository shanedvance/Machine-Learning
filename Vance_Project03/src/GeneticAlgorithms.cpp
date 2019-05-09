//
// Created by Shane on 5/10/2018.
//

#include <iostream>
#include <vector>
#include <limits>
#include "GeneticAlgorithms.h"

using namespace std;

/**
 *
 * This sets up the genetic algorithm class.
 *
 * @param newBM passes this for evaluating the functions
 */

GeneticAlgorithms::GeneticAlgorithms(const Benchmark &newBM) : bm(newBM) {}

/**
 *
 * This is a version of the Genetic Algorithms (GA) that is known as Simple GA. This is the simplest algorithms of the
 * GA's. This will find the best (most optimal solution) of the given fitness function with respect to its dimension
 * and population. This is an effort of finding the minimum given 2 constraints: the range of the x and y and
 * f(x, y) >= 0
 *
 * @param f the fitness (cost) evaluator
 * @param ns the size of the population
 * @param dim the amount of chromosomes in the population (the dimension)
 * @param rng the range of values
 * @param t_max the max amount of generations
 * @param cr the crossover rate
 * @param mr the mutation probability
 * @return the best solution
 */

Population GeneticAlgorithms::simpleGA(Benchmark::Fitness fn, int ns, unsigned int dim, Range rng, int t_max,
                                       double cr, MutationRate mr)
{

    vector<double> bestSolutions;

    /* Initialize population                                                  */
    vector<Population> p = this->generate_population(fn, ns, dim, rng);

    Population gBest = this->getBestSolution(fn, p); // set the all time best to the current minimum of the population

    /* Search for the best fitness in each generation                        */
    for (int t = 1; t <= t_max; ++t)
    {

        vector<NewPopulation> np; // create a new population

        for (unsigned int s = 0; s < static_cast<unsigned int>(ns); s += 2)
        {

            /* This selects parents using the Roulette Wheel selector        */
            Parent parent = this->select(p);

            /*****************************************************************
             *******Crossover and mutation are a binary-value encoding********
             *****************************************************************/
            /* This is the mating process of the parents to create offspring */
            Child child = this->crossOver(parent, cr);

            /* This performs a mutation using the mutation probability       */
            this->mutate(&child.one, mr, rng);
            this->mutate(&child.two, mr, rng);

            /* Add the children to the new population                        */
            np.emplace_back(NewPopulation(child.one));
            np.emplace_back(NewPopulation(child.two));

        }

        /* Evaluate the fitness of the new population                        */
        this->evaluate(fn, &np);

        /* Make sure f(x, y) >= 0                                            */
        this->reduce(&p, &np);

        Population best = this->getBestSolution(fn, p);

        /* check to see if we improved on the all time best                  */
        if (best.cost < gBest.cost)
        {
            gBest = best;
        }

        // output the best solution for each generation and average fitness
        cout << "Best Fitness: " << best.cost << endl;
        cout << "Avg. Fitness: " << (this->totalFitness / ns) << endl << endl;

    }

    return gBest; // returns the overall best solution

}