import numpy as np
import pandas as pd


class GAknapsack:

    def __init__(self,
                 weights,
                 values,
                 max_weight,
                 population_size=10,
                 max_generation=10,
                 best_parent=1,
                 nextgen_parent=5,
                 tournament_size=5,
                 mutation_prob=0.5,
                 max_mutation=5):

        # optimization variable
        self.weights = weights
        self.values = values
        self.max_weight = max_weight

        # GA setting
        self.pop_size = population_size
        self.max_generation = max_generation
        self.best_parent = best_parent
        self.nextgen_parent = nextgen_parent
        self.tournament_size = tournament_size
        self.mutation_prob = mutation_prob
        self.max_mutaion = max_mutation

        self.gene_length = self.weights.shape[0]

        # init GA
        self.pop = self._init_pop()
        self.compute_fitness()

    def _init_pop(self):
        population = []
        for i in range(self.pop_size):
            z = np.zeros(self.gene_length)
            pick = np.random.randint(0, self.gene_length, self.gene_length // 2)
            z[pick] = 1
            population.append(z)

        return np.array(population)

    def run_ga(self, log=False, output_path="./ga_log/"):
        if log:
            for generation in range(1, self.max_generation + 1):
                if generation % 25 == 0:
                    print(generation)
                self.compute_fitness()
                self.selection()
                self.breeding()
                df = pd.DataFrame(self.pop)
                df.to_csv(f'{output_path}/gen_{generation}.csv', index=False)
            self.compute_fitness()
            df = pd.DataFrame(self.pop)
            df.to_csv(f'{output_path}/_gen_last.csv', index=False)
        else:
            for generation in range(1, self.max_generation + 1):
                if generation % 25 == 0:
                    print(generation)
                self.compute_fitness()
                self.selection()
                self.breeding()
            self.compute_fitness()

    def compute_fitness(self):
        sum_weight = np.sum(self.pop * self.weights, axis=1)
        sum_value = np.sum(self.pop * self.values, axis=1)
        overload = sum_weight > self.max_weight
        fitness = sum_value
        fitness[overload] = -1

        self.fitness = np.round(fitness, 3)
        # sort fitness ind descending
        self.fitness_sort_ind = np.argsort(self.fitness)[::-1]

    def selection(self):
        # sort fitness ind descending
        self.next_gen = np.zeros_like(self.pop)

        # select best parent to next gen
        for i in range(self.best_parent):
            self.next_gen[i] = self.pop[self.fitness_sort_ind[i]]

        # perform tournament tournament selection
        for i in range(self.best_parent, self.nextgen_parent):
            tour_winner = self._tournament_selection()
            self.next_gen[i] = tour_winner

    def breeding(self):
        parent = self.next_gen[:self.nextgen_parent]
        parent_ind = np.arange(len(parent))
        gene_ind = np.arange(0, self.gene_length)

        for i in range(self.nextgen_parent, self.pop_size - 1, 2):
            parent_a, parent_b = parent[np.random.choice(parent_ind, 2, replace=False)]
            loc = np.sort(np.random.choice(gene_ind, 2, replace=False))
            offspring_a, offspring_b = self._crossover(parent_a, parent_b, loc)
            self.next_gen[i] = offspring_a
            if i + 1 < self.pop_size:
                self.next_gen[i + 1] = offspring_b

        for i in range(self.nextgen_parent, self.pop_size):
            self._mutation(self.next_gen[i])

        self.pop = self.next_gen

    def _tournament_selection(self):
        for i in range(self.best_parent, self.nextgen_parent):
            tour_fitness = np.random.choice(self.fitness, self.tournament_size)
            tour_fittest = np.max(tour_fitness)

            fittest_invidual = self.pop[np.where(self.fitness == tour_fittest)[0][0]]
            return fittest_invidual

    def _crossover(self, a, b, loc):
        start = loc[0]
        end = loc[1]

        a[start:end], b[start:end] = b[start:end], a[start:end]

        return a, b

    def _mutation(self, a):
        if np.random.rand() > self.mutation_prob:
            loc = np.random.randint(0, self.gene_length)
            a[loc] = int(not a[loc])

            for i in range(self.max_mutaion):
                if np.random.rand() > self.mutation_prob:
                    loc = np.random.randint(0, self.gene_length)
                    a[loc] = int(not a[loc])
                else:
                    return a
        return a
