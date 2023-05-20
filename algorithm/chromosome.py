import math
import random


class Chromosome:
    domain_definition = None
    precision = None
    L = None
    step = None
    fitness_function = None
    @staticmethod
    def set_domain_definition(domain_definition, precision, fitness_function):
        Chromosome.fitness_function = fitness_function
        Chromosome.domain_definition = domain_definition
        Chromosome.precision = precision
        Chromosome.L = int(math.ceil(math.log2((domain_definition[1] - domain_definition[0]) * (10 ** precision))))
        Chromosome.step = (domain_definition[1] - domain_definition[0]) / (2 ** Chromosome.L)

    def __init__(self, genes=None):
        if genes is None:
            self.genes = random.choices([0, 1], k=Chromosome.L)
        else:
            self.genes = [x for x in genes]
        self.fitness = Chromosome.fitness_function(self.binary_to_real())

    def __str__(self):
        return f"{self.genes}"

    def binary_to_real(self):
        val = 0
        aux = self.genes[::-1]
        for i in range(Chromosome.L):
            if aux[i] == 1:
                val += 2 ** i
        return Chromosome.domain_definition[0] + val * Chromosome.step

    def real_to_binary(self, real_value):
        val = real_value
        binary = []
        val = (real_value - Chromosome.domain_definition[0]) // Chromosome.step
        for i in range(Chromosome.L):
            binary.append(int(val % 2))
            val = val // 2
        self.genes = binary[::-1]


