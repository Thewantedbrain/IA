import heapq

# Grafo: Aeropuerto de Origen -> Diccionario de datos
aeropuertos = {
    "Aeropuerto Caracas": {"Aeropuerto Madrid": 10, "Aeropuerto Nueva York": 8},
    "Aeropuerto Madrid": {"Aeropuerto Caracas": 10, "Aeropuerto Pekín": 12},
    "Aeropuerto Nueva York": {"Aeropuerto Caracas": 8, "Aeropuerto Tokio": 10},
    "Aeropuerto Tokio": {"Aeropuerto Nueva York": 10, "Aeropuerto Pekín": 9},
    "Aeropuerto Pekín": {"Aeropuerto Madrid": 12, "Aeropuerto Tokio": 9}
}

# Función heurística basada en una distancia aproximada
def heuristic(point_a, point_b):
    return abs(len(point_a) - len(point_b))

# Implementación del Algoritmo A-Epsilon
def a_epsilon(inicio, meta, epsilon=1.2):
    open_set = []
    heapq.heappush(open_set, (0, inicio))
    came_from = {}
    g_score = {point: float('inf') for point in aeropuertos}
    g_score[inicio] = 0
    f_score = {point: float('inf') for point in aeropuertos}
    f_score[inicio] = heuristic(inicio, meta)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == meta:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(inicio)
            return path[::-1]

        for neighbor, distance in aeropuertos[current].items():
            tentative_g_score = g_score[current] + distance
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + epsilon * heuristic(neighbor, meta)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Ejecución del algoritmo de Aeropuerto Caracas a Aeropuerto Pekín
inicio = "Aeropuerto Caracas"
meta = "Aeropuerto Pekín"
ruta = a_epsilon(inicio, meta, epsilon=5)
print(f"Ruta de vuelo optimizada: {' -> '.join(ruta)}")