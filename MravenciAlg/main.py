import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Definice třídy AntColony
class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
            self.pheromone *= self.decay
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # going back to where we started
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

# Načtení souboru
file_path = './results.csv'
data = pd.read_csv(file_path)

# Zobrazení prvních pár řádků pro kontrolu
data.head()

# Převod tabulky na matici vzdáleností
distance_matrix = data.iloc[:, 3:].values
epsilon = 1e-10
distance_matrix += epsilon

#Počet mravenců (n_ants): Zvýšení počtu mravenců může vést k lepšímu průzkumu prostoru řešení, ale zároveň zvýší výpočetní náročnost.
#Počet nejlepších cest (n_best): Více nejlepších cest může pomoci konvergenci, ale příliš vysoká hodnota může způsobit předčasnou konvergenci.
#Počet iterací (n_iterations): Více iterací obvykle vede k lepším výsledkům, ale také zvýší výpočetní čas.
#Rozpad feromonů (decay): Hodnota mezi 0 a 1. Nižší hodnoty způsobí rychlejší rozpad feromonů, což může pomoci prozkoumávat nové cesty.
#Vliv feromonů (alpha): Vyšší hodnota zvýší vliv feromonů při volbě cesty.
#Vliv vzdálenosti (beta): Vyšší hodnota zvýší vliv vzdálenosti při volbě cesty.

# Parametry algoritmu
n_ants = 80
n_best = 3
n_iterations = 10
decay = 0.7
alpha = 1
beta = 5
# Inicializace a spuštění algoritmu
ant_colony = AntColony(distance_matrix, n_ants, n_best, n_iterations, decay, alpha, beta)
shortest_path = ant_colony.run()

print("Nejkratší nalezená cesta:", shortest_path)

# Načtení informací o místech (název, x, y)
locations = data[['name', 'x', 'y']]

# Extrakce koordinátů nejkratší nalezené cesty
path_indices = [edge[0] for edge in shortest_path[0]] + [shortest_path[0][-1][1]]
path_coords = locations.iloc[path_indices]

# Vykreslení grafu
plt.figure(figsize=(10, 8))

# Vykreslení míst jako body
plt.scatter(locations['x'], locations['y'], color='blue', label='Místa')
for i, txt in enumerate(locations['name']):
    plt.annotate(txt, (locations['x'][i], locations['y'][i]), fontsize=8, ha='right')

# Vykreslení nejkratší cesty
plt.plot(path_coords['x'], path_coords['y'], color='red', linestyle='-', linewidth=2, marker='o', label='Nejkratší cesta')

plt.title('Nejkratší nalezená cesta pomocí mravenčího algoritmu')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

