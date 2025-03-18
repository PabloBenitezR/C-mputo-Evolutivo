import numpy as np

# Configuración
N = 14                      # Número de reinas
pob_size = 100              # Tamaño de la población
proba_muta = 0.2            # Tasa de mutación
codificacion = 'real'     # 'entero', 'real' o 'binario'
E = 10                       # Número de experimentos
GEN_MAX = 5000              # Límite de generaciones por experimento

# Fitness (máximo 91)
def fitness(casillas):
    conflictos = 0
    for i in range(N):
        for j in range(i + 1, N):
            if casillas[i] == casillas[j] or abs(casillas[i] - casillas[j]) == abs(i - j):
                conflictos += 1
    return (N * (N - 1) // 2) - conflictos

# Decodificación
def decode(cromosoma):
    if codificacion == 'entero':
        return cromosoma
    elif codificacion == 'real':
       return np.clip(np.floor(cromosoma * N).astype(int), 0, N - 1)
    elif codificacion == 'binario':
        genes = [int(''.join(map(str, cromosoma[i * 4:(i + 1) * 4])), 2) for i in range(N)]
        return np.array([min(g, N - 1) for g in genes])

# Inicialización
def poblacion():
    if codificacion == 'entero':
        return [np.random.randint(0, N, N) for _ in range(pob_size)]
    elif codificacion == 'real':
        return [np.random.rand(N) for _ in range(pob_size)]
    elif codificacion == 'binario':
        return [np.random.randint(0, 2, N * 4) for _ in range(pob_size)]

# Cruce
def cruzar(p1, p2):
    cp = np.random.randint(1, len(p1) - 1)
    return np.concatenate([p1[:cp], p2[cp:]])

# Mutación
def mutar(crom):
    if codificacion == 'entero':
        if np.random.rand() < proba_muta:
            idx = np.random.randint(N)
            crom[idx] = np.random.randint(0, N)
    elif codificacion == 'real':
        if np.random.rand() < proba_muta:
            idx = np.random.randint(N)
            crom[idx] = np.clip(crom[idx] + np.random.normal(0, 0.1), 0, 1)
    elif codificacion == 'binario':
        for i in range(len(crom)):
            if np.random.rand() < proba_muta:
                crom[i] = 1 - crom[i]
    return crom

# EJECUTAR EXPERIMENTOS
for exp in range(1, E + 1):
    pob = poblacion()
    generacion = 0
    final = False

    while generacion < GEN_MAX:
        # Evaluación
        scored = [(chrom, fitness(decode(chrom))) for chrom in pob]
        scored.sort(key=lambda x: -x[1])
        pob = [chrom for chrom, fit in scored]

        # Verificar solución
        if scored[0][1] == (N * (N - 1) // 2):
            print(f"✅ Experimento {exp} finalizado: Solución encontrada en generación {generacion}")
            print("Cromosoma decodificado:", decode(scored[0][0]))
            final = True
            break

        # Elitismo y reproducción
        next_gen = pob[:2]
        while len(next_gen) < pob_size:
            idx1, idx2 = np.random.choice(50, 2, replace=False)
            p1, p2 = pob[idx1], pob[idx2]
            child = cruzar(p1, p2)
            child = mutar(child)
            next_gen.append(child)

        pob = next_gen
        generacion += 1

    if not final:
        print(f"❌ Experimento {exp} finalizado: No se encontró solución en {GEN_MAX} generaciones")
