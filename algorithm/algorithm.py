from algorithm.chromosome import Chromosome
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter


class GeneticAlgorithm:
    def __init__(self, population_size=None, domain_definition=None, parameters=None, fitness_function=None,
                 precision=None, crossover_rate=None, mutation_rate=None, epochs=None, random_add=None):
        self.population = None
        self.population_size = population_size
        self.domain_definition = domain_definition
        self.parameters = parameters
        self.fitness_function = fitness_function
        self.precision = precision
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.epochs = epochs
        self.random_add = random_add
        self.cnt_add = None
        self.best_chromosome = None
        self.evolution_of_best_chromosome = []
        self.evolution_of_average_fitness = []
        self.show_parameters = True
        self.output_file = open("./data/output.txt", "w")

        if self.domain_definition is not None and self.precision is not None:
            Chromosome.set_domain_definition(self.domain_definition, self.precision)
        # self.best_chromosome = None
        # self.best_fitness = None
        # self.best_epoch = None

    def __str__(self):
        return f"Population size: {self.population_size}\n" \
               f"Domain definition: {self.domain_definition}\n" \
               f"Parameters: {self.parameters}\n" \
               f"Precision: {self.precision}\n" \
               f"Crossover rate: {self.crossover_rate}\n" \
               f"Mutation rate: {self.mutation_rate}\n" \
               f"Epochs: {self.epochs}"


    def initialize_from_file(self, input_file):
        self.population_size = int(input_file.readline().split(':')[1].strip())
        self.domain_definition = tuple(float(x) for x in input_file.readline().split(':')[1].strip().split(' '))
        self.parameters = [float(x) for x in input_file.readline().split(':')[1].strip().split(' ')]
        # self.fitness_function = lambda x: self.parameters[0] * x * x + self.parameters[1] * x + self.parameters[2]
        self.fitness_function = lambda x: self.parameters[0] * x * x * x + self.parameters[1] * x * x + self.parameters[2] * x + self.parameters[3]
        self.precision = float(input_file.readline().split(':')[1].strip())
        self.crossover_rate = float(input_file.readline().split(':')[1].strip())
        self.mutation_rate = float(input_file.readline().split(':')[1].strip())
        self.epochs = int(input_file.readline().split(':')[1].strip())
        self.random_add = float(input_file.readline().split(':')[1].strip())
        print("Parameters read from file: ", self.random_add)
        self.cnt_add = int(self.random_add * self.population_size)
        Chromosome.set_domain_definition(self.domain_definition, self.precision, self.fitness_function)

    def run(self):
        self.initialize_population()
        for epoch in range(self.epochs):
            self.evaluate_population()
            self.select_parents()
            self.crossover()
            self.mutate()
            self.population.append(self.best_chromosome)
            for j in range(self.cnt_add):
                self.population.append(Chromosome())
            self.print_epoch(epoch)
            self.evolution_of_best_chromosome.append(self.best_chromosome)
            self.evolution_of_average_fitness.append(np.mean([x.fitness for x in self.population]))
            self.show_parameters = False
        print("\n\n\n", file=self.output_file)
        print(f"Best chromosome: {self.best_chromosome.binary_to_real()}", file=self.output_file)


        for i, chromosome in enumerate(self.evolution_of_best_chromosome):
            print(f"Epoch {i}: {chromosome.binary_to_real()}  {chromosome.fitness}", file=self.output_file)

        # plt.plot(self.evolution_of_best_chromosome, label="Best chromosome")
        # plt.plot(self.evolution_of_average_fitness, label="Average fitness")
        # plt.xlabel("Epoch")
        # plt.ylabel("Fitness")
        # plt.legend()
        # plt.suptitle("Evolution of best chromosome")
        # plt.savefig("evolution_of_best_chromosome.png")
        # plt.show()

        # plot evolution of best chromosome in time

        # x = np.arange(0, self.epochs, 1)
        # fig = plt.figure(figsize=(12, 8))
        # plt.xlabel("Epochs")
        # plt.ylabel("Fitness")
        # plt.suptitle("Evolution of best chromosome and average fitness")
        # x = [0]
        # y = [self.evolution_of_best_chromosome[0]]
        # ln, = plt.plot(x, y, '-')
        #
        # def update(frame):
        #     x.append(x[-1] + 1)
        #     y.append(self.evolution_of_best_chromosome[x[-1]])
        #     ln.set_data(x, y)
        #     fig.gca().relim()
        #     fig.gca().autoscale_view()
        #     return ln,
        # animation = FuncAnimation(fig, update, interval=100, frames=self.epochs - 2, cache_frame_data=False, repeat=False)
        # plt.show()

    def initialize_population(self):
        if self.show_parameters:
            print("Initializing population...", file=self.output_file)
        self.population = [Chromosome() for _ in range(self.population_size)]
        # self.population[0].real_to_binary(random.uniform(self.domain_definition[0], self.domain_definition[1]))
        if self.show_parameters:
            for chromosome in self.population:
                print("".join([str(x) for x in chromosome.genes]), file=self.output_file)

    def evaluate_population(self):
        if self.show_parameters:
            print("\nPopulation after evaluation:", file=self.output_file)
            for chromosome in self.population:
                chromosome_string = "".join([str(x) for x in chromosome.genes])
                print(f"{chromosome_string} value={chromosome.binary_to_real()} fitness={chromosome.fitness}", file=self.output_file)

    def select_parents(self):
        if self.show_parameters:
            print("\nProbabilities:", file=self.output_file)
        fitness_sum = sum([chromosome.fitness for chromosome in self.population])
        probabilities = [chromosome.fitness / fitness_sum for chromosome in self.population]

        # select the best chromosome
        self.best_chromosome = self.population[probabilities.index(max(probabilities))]
        aux = []

        if self.show_parameters:
            # plt.figure()
            # plt.pie(np.array(probabilities), labels=np.array([round(i.fitness, 3) for i in self.population]), autopct='%1.1f%%')
            # plt.show()
            print(" ".join([str(x) for x in probabilities]), file=self.output_file)
        for i in range(1, self.population_size):
            probabilities[i] += probabilities[i - 1]
        if self.show_parameters:
            print(" ".join([str(x) for x in ([0] + probabilities)]), file=self.output_file)
        selected_parents = []
        # for i in range(self.population_size - 1):
        for i in range(self.population_size - 1 - int(self.cnt_add)):
            r = random.random()

            # binary search
            left = 0
            right = self.population_size - 1
            pos = -1
            while left <= right:
                mid = (left + right) // 2
                if probabilities[mid] >= r:
                    pos = mid
                    right = mid - 1
                else:
                    left = mid + 1
            aux.append(Chromosome(self.population[pos].genes))
            selected_parents.append((r, pos))
        self.population = [x for x in aux]
        if self.show_parameters:
            print("\nSelected parents:", file=self.output_file)
            for s in selected_parents:
                print(f"u={s[0]} chromosome={s[1]}", file=self.output_file)

            print("\nPopulation after selection:", file=self.output_file)
            for chromosome in self.population:
                chromosome_string = "".join([str(x) for x in chromosome.genes])
                print(f"{chromosome_string} value={chromosome.binary_to_real()} fitness={chromosome.fitness}", file=self.output_file)

    def crossover(self):
        if self.show_parameters:
            print(f"\nCrossover probability: {self.crossover_rate}", file=self.output_file)
        crossover_parents = []
        used = [False for _ in range(self.population_size - 1)]
        # for i in range(self.population_size - 1):
        for i in range(len(self.population)):
            if random.random() < self.crossover_rate:
                crossover_parents.append(i)
                used[i] = True

        if len(crossover_parents) % 2 == 1:
            used[crossover_parents[-1]] = False
            crossover_parents.pop()

        crossover_parents = [self.population[i] for i in crossover_parents]

        random.shuffle(crossover_parents)
        for i in range(0, len(crossover_parents), 2):
            # print(crossover_parents[i].genes, crossover_parents[i + 1].genes)
            pos = random.randint(1, Chromosome.L - 2)
            if self.show_parameters:
                print(f"Point of crossover: {pos}", file=self.output_file)
                c1 = "".join([str(x) for x in crossover_parents[i].genes])
                c2 = "".join([str(x) for x in crossover_parents[i + 1].genes])
                c1 = c1[:pos] + "|" + c1[pos:]
                c2 = c2[:pos] + "|" + c2[pos:]
                print(f"Crossover parents:  {c1} {c2}", file=self.output_file)
                # print("Crossover parents: ", "".join([str(x) for x in crossover_parents[i].genes]), "".join([str(x) for x in crossover_parents[i + 1].genes]), file=self.output_file)
            for j in range(pos, Chromosome.L):
                crossover_parents[i].genes[j], crossover_parents[i + 1].genes[j] = crossover_parents[i + 1].genes[j], crossover_parents[i].genes[j]

            crossover_parents[i].fitness = self.fitness_function(crossover_parents[i].binary_to_real())
            crossover_parents[i + 1].fitness = self.fitness_function(crossover_parents[i + 1].binary_to_real())
            # print(crossover_parents[i].genes, crossover_parents[i + 1].genes)
            if self.show_parameters:
                c1 = "".join([str(x) for x in crossover_parents[i].genes])
                c2 = "".join([str(x) for x in crossover_parents[i + 1].genes])
                c1 = c1[:pos] + "|" + c1[pos:]
                c2 = c2[:pos] + "|" + c2[pos:]
                print(f"Crossover children: {c1} {c2}", file=self.output_file)
                # print("Crossover children:", "".join([str(x) for x in crossover_parents[i].genes]), "".join([str(x) for x in crossover_parents[i + 1].genes]) + "\n", file=self.output_file)

        self.population = [self.population[i] for i in range(len(self.population)) if not used[i]]
        for chromosome in crossover_parents:
            self.population.append(chromosome)

        if self.show_parameters:
            print("\nPopulation after crossover:", file=self.output_file)
            for chromosome in self.population:
                chromosome_string = "".join([str(x) for x in chromosome.genes])
                print(f"{chromosome_string} value={chromosome.binary_to_real()} fitness={chromosome.fitness}", file=self.output_file)

    def mutate(self):
        if self.show_parameters:
            print("\nMutation probability:", self.mutation_rate, file=self.output_file)
        # for i in range(self.population_size - 1):
        for i in range(len(self.population)):
            if random.random() < self.mutation_rate:
                pos = random.randint(0, Chromosome.L - 1)
                # print(i, pos)
                if self.show_parameters:
                    print(f"Mutate chromosome {i} at position {pos}", file=self.output_file)
                self.population[i].genes[pos] = 1 - self.population[i].genes[pos]
                self.population[i].fitness = self.fitness_function(self.population[i].binary_to_real())
        if self.show_parameters:
            print("\nPopulation after mutation:", file=self.output_file)
            for chromosome in self.population:
                chromosome_string = "".join([str(x) for x in chromosome.genes])
                print(f"{chromosome_string} value={chromosome.binary_to_real()} fitness={chromosome.fitness}", file=self.output_file)

    def print_epoch(self, epoch):
        best_chromosome_string = "".join([str(x) for x in self.best_chromosome.genes])
        if self.show_parameters:
            print(f"Epoch {epoch}:", file=self.output_file)
            print(f"Best chromosome: {best_chromosome_string} value={self.best_chromosome.binary_to_real()} fitness={self.best_chromosome.fitness}", file=self.output_file)
            print(f"Average fitness: {sum([chromosome.fitness for chromosome in self.population]) / self.population_size}", file=self.output_file)






