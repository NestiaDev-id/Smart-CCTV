from pembentukan_populasi import pembentukan_populasi_awal

def algoritma_genetika(train, test, jumlahKromosom, generations, probability):
    no_improvement_count = 0
    best_fitness_overall = float('inf')
    best_individu_overall = None
    fitness_history = []

    # Pembentukan populasi awal
    populasi = pembentukan_populasi_awal(jumlahKromosom)
    print(f'Pembentukan populasi awal: {populasi}')

    for generasi in range(1, generations + 1):
        print(f"\n=== Iterasi ke-{generasi} ===")
        new_populasi = []

        # Evaluasi fitness populasi
        # populasi = evaluasi_fitness(populasi, train, test)
        # fitness_history.append(min(populasi, key=lambda x: x['Fitness'])['Fitness'])

        # Mutasi atau crossover berdasarkan probabilitas
        while len(new_populasi) < jumlahKromosom:
            if generasi % probability == 0:
                # parent = RouletteWhell(populasi)  # Mutasi
                # offspring = mutasi2(train, test, parent)
                # offspring = evaluasi_fitness([offspring], train, test)[0]
                # new_populasi.append(offspring)
                print(">> Melakukan mutasi...")
            else:
                # parent1 = RouletteWhell(populasi)
                # parent2 = RouletteWhell(populasi)
                # offspring1, offspring2 = crossover(train, test, parent1, parent2)
                # offspring1 = evaluasi_fitness([offspring1], train, test)[0]
                # offspring2 = evaluasi_fitness([offspring2], train, test)[0]
                # new_populasi.extend([offspring1, offspring2])
                print(">> Melakukan crossover...")

        # Seleksi individu terbaik di generasi
        best_individu_generasi = min(new_populasi, key=lambda x: x['Fitness'])

        # Update individu terbaik global
        if best_individu_generasi['Fitness'] < best_fitness_overall:
            best_fitness_overall = best_individu_generasi['Fitness']
            best_individu_overall = best_individu_generasi
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Tambah individu acak jika tidak ada perbaikan terlalu lama
        if no_improvement_count >= 12:
            print(">> Tidak ada perbaikan signifikan, melakukan regenerasi sebagian populasi...")
            sorted_populasi = sorted(populasi, key=lambda x: x['Fitness'])
            size_elite = int(jumlahKromosom * 0.3)
            elite_individuals = sorted_populasi[:size_elite]
            individu_baru = pembentukan_populasi_awal(jumlahKromosom - size_elite)
            populasi = elite_individuals + individu_baru
            no_improvement_count = 0
        else:
            populasi = new_populasi[:jumlahKromosom]

    # Evaluasi terakhir untuk mendapatkan individu terbaik
    populasi = evaluasi_fitness(populasi, train, test)
    individu_fitness_min = min(populasi, key=lambda x: x['Fitness'])

    print("\n=== Hasil Akhir ===")
    print("Individu dengan fitness terbaik:", individu_fitness_min)

    return individu_fitness_min, generations, fitness_history
