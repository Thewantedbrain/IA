import heapq
# Grafo: Objeto -> Diccionario de datos
barquisimeto_sitios = {
    "Plaza Bolívar": {"Hospital Central": 2, "Avenida Lara": 3},
    "Hospital Central": {"Avenida Lara": 1.5, "Plaza Bolívar": 2},
    "Avenida Lara": {"Parque del Este": 4, "Terminal de Pasajeros": 5, "Hospital Central": 1.5, "Plaza Bolívar": 3},
    "Parque del Este": {"Las Trinitarias": 3, "Avenida Lara": 4},
    "Terminal de Pasajeros": {"La Carucieña": 6, "El Obelisco": 2.5, "Avenida Lara": 5},
    "Las Trinitarias": {"Parque del Este": 3},
    "La Carucieña": {"El Obelisco": 3.5, "Terminal de Pasajeros": 6},
    "El Obelisco": {"Florencio Jiménez": 2, "Terminal de Pasajeros": 3, "La Carucieña": 3.5},
    "Florencio Jiménez": {"Zona Industrial I": 5, "El Obelisco": 2},
    "Zona Industrial I": {"Florencio Jiménez": 5}
}
# Función heurística basada en una distancia aproximada
def heuristic(point_a, point_b):
    # Heurística simple basada en la diferencia en la longitud de nombres de puntos
    return abs(len(point_a) - len(point_b))

# Implementación del Algoritmo A-Epsilon
def a_epsilon(inicio, meta, epsilon=1.2):
    open_set = []
    heapq.heappush(open_set, (0, inicio))  # Prioridad y nodo inicial
    came_from = {}
    g_score = {point: float('inf') for point in barquisimeto_sitios}
    g_score[inicio] = 0
    f_score = {point: float('inf') for point in barquisimeto_sitios}
    f_score[inicio] = heuristic(inicio, meta)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == meta:
            # Reconstrucción de la ruta
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(inicio)
            return path[::-1]  # Retorna la ruta en el orden correcto

        for neighbor, distance in barquisimeto_sitios[current].items():
            tentative_g_score = g_score[current] + distance
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + epsilon * heuristic(neighbor, meta)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # Retorna None si no se encuentra ruta

# Ejecutar el algoritmo de Plaza Bolívar a Zona Industrial I
# Prof. María, acá puede cambiar el punto de inicio y fin, y ajustar el parámetro epsilon para
# Con valores más altos de epsilon, el algoritmo prioriza la heurística y explora más nodos rápidamente
inicio = "Plaza Bolívar"
meta = "Zona Industrial I"
route = a_epsilon(inicio, meta, epsilon=1.2)
print(f"Ruta de recolección optimizada: {' -> '.join(route)}")