import numpy as np
from math import radians, cos, sin, sqrt, atan2
import matplotlib.pyplot as plt
import pandas as pd

# Funkce pro výpočet vzdálenosti mezi dvěma GPS souřadnicemi pomocí Haversinovy formule
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Poloměr Země v kilometrech

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Načtení CSV souboru
csv_file_path = 'Pubs.csv'
data_csv = pd.read_csv(csv_file_path, delimiter=';')

# Extrahování zeměpisných šířek a délek
latitudes = data_csv['latitude']
longitudes = data_csv['longitude']

# Inicializace matice vzdáleností
num_places = len(latitudes)
distance_matrix = np.zeros((num_places, num_places))

# Výpočet vzdáleností mezi každým párem bodů
for i in range(num_places):
    for j in range(num_places):
        if i != j:
            distance_matrix[i, j] = haversine(latitudes[i], longitudes[j], latitudes[j], longitudes[j])

# Převod GPS souřadnic na kartézské souřadnice
def gps_to_cartesian(lat, lon, ref_lat, ref_lon):
    R = 6371.0  # Poloměr Země v kilometrech
    x = R * cos(radians(lat)) * cos(radians(lon) - radians(ref_lon))
    y = R * cos(radians(lat)) * sin(radians(lon) - radians(ref_lon))
    return x, y

# Referenční bod (první místo)
ref_lat, ref_lon = latitudes[0], longitudes[0]

# Převod všech GPS souřadnic na kartézské souřadnice
cartesian_coords = [gps_to_cartesian(lat, lon, ref_lat, ref_lon) for lat, lon in zip(latitudes, longitudes)]
cartesian_coords = np.array(cartesian_coords)

# Uložení výsledků do CSV souboru
output_file_path = 'C:/Users/Lukáš Bajer/Desktop/NNUI2/results.csv'

# Vytvoření seznamu záhlaví
headers = ['name', 'x', 'y'] + [f'distance_to_{i}' for i in range(num_places)]

# Vytvoření seznamu dat
results = []
for i, (place, (x, y)) in enumerate(zip(data_csv['Place name'], cartesian_coords)):
    distances = distance_matrix[i].tolist()
    results.append([place, x, y] + distances)

# Uložení dat do CSV souboru
results_df = pd.DataFrame(results, columns=headers)
results_df.to_csv(output_file_path, index=False)

# Vykreslení grafu
plt.figure(figsize=(10, 8))
plt.scatter(cartesian_coords[:, 0], cartesian_coords[:, 1])

for i, place in enumerate(data_csv['Place name']):
    plt.annotate(place, (cartesian_coords[i, 0], cartesian_coords[i, 1]))

plt.title("Mapa míst v reálných vzdálenostech")
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.grid(True)
plt.show()
