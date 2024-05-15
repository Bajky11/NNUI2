import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, distance_matrix, population_size, n_generations, mutation_rate):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate

    def initial_population(self):
        population = []
        for _ in range(self.population_size):
            path = np.random.permutation(len(self.distance_matrix))
            population.append(path)
        return population

    def calculate_fitness(self, population):
        fitness = []
        for path in population:
            path_distance = sum(self.distance_matrix[path[i], path[(i + 1) % len(path)]] for i in range(len(path)))
            fitness.append(1 / path_distance)
        return fitness

    def select_parents(self, population, fitness,k=3):
        return self.select_parents_tournament(population, fitness)

    def crossover(self, parent1, parent2):
        start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
        child = [None]*len(parent1)
        child[start:end] = parent1[start:end]
        # Include genes from parent2 that are not in the child
        current_pos = end
        for gene in parent2:
            if gene not in child:
                if current_pos >= len(child):
                    current_pos = 0
                child[current_pos] = gene
                current_pos += 1
        return np.array(child)

    def mutate(self, path):
        if np.random.rand() < self.mutation_rate:
            swap_idx1, swap_idx2 = np.random.choice(len(path), 2, replace=False)
            path[swap_idx1], path[swap_idx2] = path[swap_idx2], path[swap_idx1]
        return path

    def run(self):
        population = self.initial_population()
        for _ in range(self.n_generations):
            fitness = self.calculate_fitness(population)
            new_population = self.select_parents(population, fitness)
            children = []
            for i in range(0, len(new_population), 2):
                parent1, parent2 = new_population[i], new_population[(i + 1) % len(new_population)]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                children.append(self.mutate(child1))
                children.append(self.mutate(child2))
            population = children
        return min(population, key=lambda x: sum(self.distance_matrix[x[i], x[(i + 1) % len(x)]] for i in range(len(x))))
    
    # Různé způsoby pro výber rodičů pro další generace( Seřazené podle efektivity )
    def select_parents_tournament(self, population, fitness, k=3):
        selected_parents = []
        for _ in range(len(population)):
            tournament = np.random.choice(len(population), k)
            best = tournament[np.argmax([fitness[i] for i in tournament])]
            selected_parents.append(population[best])
        return selected_parents
    
    def select_parents_rank(self, population, fitness):
        rank = np.argsort(np.argsort(fitness))
        probabilities = [rank[i] / sum(rank) for i in range(len(rank))]
        indices = np.random.choice(len(population), size=len(population), replace=True, p=probabilities)
        parents = [population[idx] for idx in indices]
        return parents
    
    def select_parents_roulette(self, population, fitness):
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        indices = np.random.choice(len(population), size=len(population), replace=True, p=probabilities)
        parents = [population[idx] for idx in indices]
        return parents
    
    # ----------------------------------------------------------------
    # Různé způsoby pro křížení
    def crossover_ox(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        child = [None] * size
        child[start:end] = parent1[start:end]
    
        position = end
        for i in range(end, end + size):
            candidate = parent2[i % size]
            if candidate not in child:
                if position >= size:
                    position = 0
                while child[position] is not None:
                    position += 1
                child[position] = candidate
    
        return np.array(child)
    
    def crossover_cx(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size
        cycle = 0
        index = 0
    
        while None in child:
            if child[index] is None:
                cycle += 1
                value = parent1[index]
                while True:
                    child[index] = parent1[index] if cycle % 2 else parent2[index]
                    index = np.where(parent1 == parent2[index])[0][0]
                    if value == parent1[index]:
                        break
            index = (index + 1) % size
    
        return np.array(child)
    
    def crossover_pmx(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        child = [None] * size
        child[start:end] = parent1[start:end]
    
        for i in range(start, end):
            if parent2[i] not in child[start:end]:
                value = parent2[i]
                while True:
                    index = np.where(parent1 == value)[0][0]
                    if child[index] is None:
                        child[index] = parent2[i]
                        break
                    value = parent2[index]
    
        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]
    
        return np.array(child)
    # ----------------------------------------------------------------


# Načtení souboru
file_path = './results.csv'
data = pd.read_csv(file_path)

# Zobrazení prvních pár řádků pro kontrolu
data.head()

# Převod tabulky na matici vzdáleností
distance_matrix = data.iloc[:, 3:].values

# Parametry genetického algoritmu
population_size = 100
n_generations = 500
mutation_rate = 0.02

# Spuštění genetického algoritmu
ga = GeneticAlgorithm(distance_matrix, population_size, n_generations, mutation_rate)
best_path = ga.run()

# Výpočet délky nejlepší cesty
best_path_distance = sum(distance_matrix[best_path[i], best_path[(i + 1) % len(best_path)]] for i in range(len(best_path)))

print("Nejlepší nalezená cesta:", best_path)
print("Délka nejlepší cesty:", best_path_distance)

# Načtení informací o místech (název, x, y)
locations = data[['name', 'x', 'y']]

# Extrakce koordinátů nejlepší nalezené cesty
path_coords = locations.iloc[best_path]

# Vykreslení grafu
plt.figure(figsize=(10, 8))

# Vykreslení míst jako body
plt.scatter(locations['x'], locations['y'], color='blue', label='Místa')
for i, txt in enumerate(locations['name']):
    plt.annotate(txt, (locations['x'][i], locations['y'][i]), fontsize=8, ha='right')

# Vykreslení nejlepší cesty
plt.plot(path_coords['x'], path_coords['y'], color='red', linestyle='-', linewidth=2, marker='o', label='Nejlepší cesta')

plt.title('Nejlepší nalezená cesta pomocí genetického algoritmu')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
