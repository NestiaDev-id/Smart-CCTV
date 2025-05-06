import random
# Fungsi crossover untuk dua parent biner, hasilkan dua anak
def crossover(parent1, parent2):
    anak1 = {}
    anak2 = {}

    for kunci in parent1:
        if kunci == 'Fitness':
            anak1[kunci] = None
            anak2[kunci] = None
            continue

        # Ambil dua string biner
        biner1 = parent1[kunci]
        biner2 = parent2[kunci]

        # Tentukan titik potong acak
        titik = random.randint(1, len(biner1) - 1)

        # Lakukan crossover satu titik
        anak1[kunci] = biner1[:titik] + biner2[titik:]
        anak2[kunci] = biner2[:titik] + biner1[titik:]

    return anak1, anak2
