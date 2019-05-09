//
// Created by Shane on 5/10/2018.
//

#ifndef OPTIMIZATION_GENETICALGORITHMS_H
#define OPTIMIZATION_GENETICALGORITHMS_H

#include <random>
#include <algorithm>
#include "Benchmark.h"

using namespace std;

/**
 *
 * This defines the new population
 *
 * @typedef Population and NewPopulation
 */

typedef struct Population
{

    /**
     * This setup up the population structure
     *
     * @param genes
     */

    explicit Population(vector<int> genes) : individual(move(genes)) {}
    vector<int> individual;
    double cost{};

    // this overides the default less than operator for custom use
    bool operator<(const Population &population) const
    {
        return cost < population.cost;
    }

} NewPopulation;

/**
 * @typedef Range this
 */

typedef struct Range
{
    Range() = default;
    Range(int lb, int ub) : LB(lb), UB(ub) {}
    int LB, UB;
} Range;

/**
 *
 * This creates an individual whether it be a child or parent.
 *
 * @typedef Parent and Child
 */

typedef struct Parent
{

    vector<int> one;
    vector<int> two;

} Child;

typedef double MutationRate;

class GeneticAlgorithms
{

    public:

        explicit GeneticAlgorithms(const Benchmark &);
        Population simpleGA(Benchmark::Fitness, int, unsigned int, Range, int, double, MutationRate);

    private:

        Benchmark bm;
        int ns;
        unsigned int dim;
        double totalFitness;

        int random(Range rng)
        {
            // Set up pseudo-random generator using built-in C++ mersenne_twister_engine
            random_device seed{};
            mt19937 engine{seed()};
            uniform_int_distribution<int> generate_random_number{rng.LB, rng.UB};

            return generate_random_number(engine);
        }

        double random(double lb, double ub)
        {
            // Set up pseudo-random generator using built-in C++ mersenne_twister_engine
            random_device seed{};
            mt19937 engine{seed()};
            uniform_real_distribution<double> generate_random_number{lb, ub};

            return generate_random_number(engine);
        }

        vector<int> random_vector(int dim, Range rng)
        {

            vector<int> temp;

            // Create a vector of dimensionSize containing random real numbers
            for (int i = 0; i < dim; ++i)
            {
                temp.emplace_back(random(rng));
            }

            return temp;

        }

        vector<Population> generate_population(Benchmark::Fitness fn, int ns, unsigned int dim, Range rng)
        {

            this->ns = ns;
            this->dim = dim;

            vector<Population> populations;
            for (int i = 0; i < ns; ++i)
            {
                Population p(this->random_vector(dim, rng));
                while ((p.cost = (this->bm.*fn)(p.individual.at(0), p.individual.at(1))) < 0)
                {
                    p.individual = this->random_vector(dim, rng);
                }

                populations.emplace_back(p); // add the new pop to the vector
            }

            this->totalFitness = this->sum(populations);

            return populations; // return the population

        }

        vector<int> join(const vector<int> &p1, const vector<int> &p2, unsigned int d)
        {
            vector<int> temp;

            // go through {0...d} and join
            for (unsigned int i = 0; i < d; ++i)
            {
                temp.emplace_back(p1.at(i));
            }

            // go through {d...dim} and join
            for (auto j = d; j < this->dim; ++j)
            {
                temp.emplace_back(p2.at(j));
            }

            return temp;
        }

        // uses roulette wheel selection
        vector<int> selectParent(const vector<Population> &p)
        {

            // setting the lower values to have high probability as being chosen since we are minimizing we need to
            // have them have a high chance of being selected
            double totalNormFitness = 0.0;
            for (Population pop : p)
            {
                totalNormFitness += 1.0 / (1.0 + pop.cost);
            }

            double r = this->random(0.0, totalNormFitness); // get random number
            unsigned int s = 0;
            for (/* initialized */; s < static_cast<unsigned int>(this->ns) - 1 && r > 0; ++s)
            {
                r = r - (1.0 / (1.0 + p.at(s).cost)); // reduce because the fitness is high
            }
            return p.at(s).individual; // return the fittest parent
        }

        Parent select(const vector<Population> &p)
        {
            Parent parent;
            parent.one = this->selectParent(p); // select individual for parent one
            parent.two = this->selectParent(p); // select individual for parent two
            return parent;
        }

        Child crossOver(const Parent &parent, double cr)
        {
            Child child;
            if (this->random(0.0, 1.0) < cr)
            {
                // grab random number between 1 and 2 (the dimension amount)
                unsigned int d = static_cast<unsigned int>(this->random(Range(1, this->dim)));
                child.one = this->join(parent.one, parent.two, d); // create child one
                child.two = this->join(parent.two, parent.one, d); // create child two
            }
            else
            {
                child.one = parent.one; // child one is exact copy of parent 1
                child.two = parent.two; // child two is exact copy of parent 2
            }
            return child; // return the children
        }

        // this performs a Integer-valued mutation
        void mutate(vector<int> *child, MutationRate mr, Range rng)
        {
            // grab random number between 0 and 1 and check if it is < the mutation rate

            // perform mutation for each dimension
            for (unsigned int i = 0; i < this->dim; ++i)
            {
                if (this->random(0.0, 1.0) < mr)
                {
                    // this uses the Gaussian distribution approach making sure the mutated number is in bounds
                    child->at(i) += random(Range(-2, 2)); // using a small random step size of the range -2 and 2

                    // check if we are in bounds
                    if (child->at(i) > rng.UB) { child->at(i) = rng.UB; }
                    if (child->at(i) < rng.LB) { child->at(i) = rng.LB; }
                }
            }
        }

        void reduce(vector<Population> *p, vector<NewPopulation> *np)
        {
            // go through each member of the population and make sure their fitness being f(x, y) >= 0 is true
            for (unsigned int s = 0 ; s < this->ns; ++s)
            {
                // check if f(x, y) >= 0 else set it to the parent
                if (np->at(s).cost < 0)
                {
                    np->at(s) = p->at(s); // let this child die out by replacing with parent
                }

                // check if the child improved on the parent else set the child to the parent
                if (this->random(Range(0, 1)) && np->at(s).cost > p->at(s).cost)
                {
                    np->at(s) = p->at(s);
                }
            }

            /* Swap the Population and NewPopulation */
            p->swap(*np);

            // update the total fitness
            this->totalFitness = this->sum(*p);
        }

        // grab the sum of all the cost of each individual of the population
        double sum(vector<Population> p)
        {
            double result = 0.0;
            for (const Population &pop : p)
            {
                result = result + pop.cost; // compute the total sum
            }
            return result; // return the result
        }

        Population getBestSolution(Benchmark::Fitness fn, const vector<Population> &p)
        {
            auto x_best = min_element(p.begin(), p.end());
            return *x_best;
        }

        void evaluate(Benchmark::Fitness fn, vector<Population> *p)
        {
            // evaluate the function for each individual in the population
            for (Population &pop : *p)
            {
                pop.cost = (this->bm.*fn)(pop.individual.at(0), pop.individual.at(1));
            }
            this->totalFitness = this->sum(*p);
        }

};

#endif //OPTIMIZATION_GENETICALGORITHMS_H