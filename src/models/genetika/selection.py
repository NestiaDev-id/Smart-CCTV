import random

def RouletteWheel(populasi):
    fitness_balik = []
    fitness_relatif = []
    fitness_kumulatif = []

    # Langkah 1: Hitung kebalikan fitness (semakin kecil error, semakin bagus)
    for individu in populasi:
        fitness = individu['Fitness']
        if fitness == 0:
            pembalik = 1e6  # jika fitness 0, beri nilai sangat besar
        else:
            pembalik = 1 / fitness
        fitness_balik.append(pembalik)

    # Langkah 2: Hitung total dan fitness relatif (proporsi)
    total_fitness_balik = sum(fitness_balik)
    for nilai in fitness_balik:
        fitness_relatif.append(nilai / total_fitness_balik)

    # Langkah 3: Hitung fitness kumulatif
    kumulatif = 0
    for nilai in fitness_relatif:
        kumulatif += nilai
        fitness_kumulatif.append(kumulatif)

    # Langkah 4: Lakukan seleksi berdasarkan nilai acak
    r = random.random()
    for i, prob_kumulatif in enumerate(fitness_kumulatif):
        if r <= prob_kumulatif:
            return populasi[i]
