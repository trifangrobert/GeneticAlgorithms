from algorithm.algorithm import GeneticAlgorithm

# read from input.txt file
alg = GeneticAlgorithm()
with open("./data/input.txt", "r") as input_file:
    alg.initialize_from_file(input_file)

# print(alg)
alg.run()






