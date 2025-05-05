import random
from .holt_winters import holt
from .evaluate import MAPE

import sys
sys.set_int_max_str_digits(10240)


def to_binary(value, num_bits=10):
    # Mengalikan nilai dengan 1000 dan mengubahnya menjadi integer
    scaled_value = int(value * (2** num_bits))  
    # Mengubah integer menjadi representasi biner
    return bin(scaled_value)[2:].zfill(num_bits) 

def to_decimal(binary_value):
    return int(binary_value, 2) / (2 ** len(binary_value))  # Mengubah dari biner ke desimal

# Algoritma Genetika
def pembentukan_populasi_awal(jumlahKromosom):
    biner = []
    for i in range(jumlahKromosom):
        # Generate random number untuk alpha beta dan gamma
        alpha = random.uniform(0.01, 0.99)
        beta = random.uniform(0.01, 0.99)
        gamma = random.uniform(0.01, 0.99)
        # Mengubah alpha, beta, gamma ke format biner
        alpha_biner = to_binary(alpha)
        beta_biner = to_binary(beta)
        gamma_biner = to_binary(gamma)
        biner.append({
            'Alpha': alpha_biner,
            'Beta': beta_biner,
            'Gamma': gamma_biner,
            'Fitness': None
            })
    return biner

# Fungsi evaluasi_fitness tanpa print yang tidak diperlukan
def evaluasi_fitness(populasi, train, test):
    evaluasi_fitness = []
    
    for i, individu in enumerate(populasi):
        # Mengambil nilai alpha, beta, gamma dalam format biner
        alpha_biner = individu['Alpha']
        beta_biner = individu['Beta']
        gamma_biner = individu['Gamma']
        
        # Konversi nilai biner menjadi desimal
        alpha = to_decimal(alpha_biner)
        beta = to_decimal(beta_biner)
        gamma = to_decimal(gamma_biner)
        
        # Menggunakan nilai alpha, beta, gamma untuk melakukan peramalan
        _, hasil_ramalan, _ = holt(train, alpha, beta, gamma)
        
        # Hitung fitness berdasarkan MAPE
        individu['Fitness'] = MAPE(test, hasil_ramalan)
        
        # Ubah ke biner lagi
        alpha = to_binary(alpha)
        beta = to_binary(beta)
        gamma = to_binary(gamma)

        # Menyusun hasil evaluasi fitness untuk individu
        evaluasi_fitness.append({
            'Alpha': alpha,
            'Beta': beta,
            'Gamma': gamma,
            'Fitness': individu['Fitness']
        })

    return evaluasi_fitness    

def RouletteWhell(populasi):
    fitness_balik = []
    fitness_relatif = []
    fitness_kumulatif = []
    
    for individu in populasi:
        # proses pembalikan
        hitung_pembalik = 1 / individu['Fitness']
        fitness_balik.append(hitung_pembalik)


    total_fitness_balik = sum(fitness_balik)
    
    # hitung fitnes relatif
    for i in range(len(populasi)):
        hitung_relatif = fitness_balik[i] / total_fitness_balik
        fitness_relatif.append(hitung_relatif)

    # hitung fitness kumulatif
    for i in range(len(populasi)):
        if i == 0:
            hitung_kumulatif = 0 + fitness_relatif[i]
        else:
            hitung_kumulatif = fitness_kumulatif[i - 1] + fitness_relatif[i]
        fitness_kumulatif.append(hitung_kumulatif)

    # print("Fitness relatif:", fitness_relatif)
    # print("Fitness kumulatif:", fitness_kumulatif)

    r1 = random.random()
    
    for i, cumulative_probability in enumerate(fitness_kumulatif):
        if r1 <= cumulative_probability:
            return populasi[i]

def mutasi2(train, test, populasi):
    # Ubah bilangan menjadi biner
    # parents1 = {
    #     'Alpha': to_binary(populasi['Alpha']),
    #     'Beta': to_binary(populasi['Beta']),
    #     'Gamma': to_binary(populasi['Gamma'])
    # }

    # Gabungkan biner
    parents1 = populasi['Alpha'] + populasi['Beta'] + populasi['Gamma']
    # print('mutasi sebelum di proses: ',parents1)
    
    point1, point2 = sorted(random.sample(range(len(parents1)), 2))
    # print('Titik potong', point1, point2)

    
    # Balikkan segmen yang diinginkan
    reversed_segment = parents1[point1:point2][::-1]
    # print('reverse: ',reversed_segment, point1, point2)
    # print('parents1[point1:point2]: ',parents1[point1:point2])
    # Gabungkan kembali segmen-segmen menjadi child1
    child1 = parents1[:point1] + reversed_segment + parents1[point2:]
    # print('gabung anak mutasi: ',child1)
    # kromosom_reversed = parents1[::-1]  # Membalik string
    # print(kromosom_reversed)

    # Pisahkan kembali ke alpha, beta, gamma
    child1 = {
        'Alpha': child1[:10],
        'Beta': child1[10:20],
        'Gamma': child1[20:30],
        'Fitness': None
    }

    # Mengubah biner ke desimal
    child1['Alpha'] = to_decimal(child1['Alpha'])
    child1['Beta'] = to_decimal(child1['Beta'])
    child1['Gamma'] = to_decimal(child1['Gamma'])

    # Hitung nilai fitness menggunakan Holt dan MAPE
    _, hasil_ramalan_child1, _ = holt(train, child1['Alpha'], child1['Beta'], child1['Gamma'])
    
    child1['Fitness'] = MAPE(test, hasil_ramalan_child1)

    # ubah ke biner lagi
    child1['Alpha'] = to_binary(child1['Alpha'])
    child1['Beta'] = to_binary(child1['Beta'])
    child1['Gamma'] = to_binary(child1['Gamma'])
    
    return {
        'Alpha': child1['Alpha'],
        'Beta': child1['Beta'],
        'Gamma': child1['Gamma'],
        'Fitness': child1['Fitness']
    }

def crossover(train, test, parents1, parents2):
    # Ubah bilangan menjadi biner
    # print(parents1)
    # print(parents2)
    parents1 = {
        'Alpha': to_binary(parents1['Alpha']),
        'Beta': to_binary(parents1['Beta']),
        'Gamma': to_binary(parents1['Gamma'])
    }
    parents2 = {
        'Alpha': to_binary(parents2['Alpha']),
        'Beta': to_binary(parents2['Beta']),
        'Gamma': to_binary(parents2['Gamma'])
    }
    

    # Gabungkan biner
    parents1 = parents1['Alpha'] + parents1['Beta'] + parents1['Gamma']
    parents2 = parents2['Alpha'] + parents2['Beta'] + parents2['Gamma']

    # Menentukan titik potong
    point1, point2 = sorted(random.sample(range(len(parents1)), 2))
    # print(f"Titik potong: {point1}, {point2}")

    # Membagi berdasarkan titik potong
    bagian1_child1 = parents1[:point1] + parents2[point1:point2] + parents1[point2:]
    bagian1_child2 = parents2[:point1] + parents1[point1:point2] + parents2[point2:]

    # Pisahkan kembali ke alpha, beta, gamma
    child1 = {
        'Alpha': bagian1_child1[:10],
        'Beta': bagian1_child1[10:20],
        'Gamma': bagian1_child1[20:30],
        'Fitness': None
    }
    child2 = {
        'Alpha': bagian1_child2[:10],
        'Beta': bagian1_child2[10:20],
        'Gamma': bagian1_child2[20:30],
        'Fitness': None
    }
    

    # Mengubah biner ke desimal
    child1['Alpha'] = to_decimal(child1['Alpha'])
    child1['Beta'] = to_decimal(child1['Beta'])
    child1['Gamma'] = to_decimal(child1['Gamma'])

    child2['Alpha'] = to_decimal(child2['Alpha'])
    child2['Beta'] = to_decimal(child2['Beta'])
    child2['Gamma'] = to_decimal(child2['Gamma'])

    _, hasil_ramalan_child1, _ = holt(train, child1['Alpha'], child1['Beta'], child1['Gamma'])
    _, hasil_ramalan_child2, _ = holt(train, child2['Alpha'], child2['Beta'], child2['Gamma'])
    
    child1['Fitness'] = MAPE(test, hasil_ramalan_child1)
    child2['Fitness'] = MAPE(test, hasil_ramalan_child2)

    # Mengubah ke biner lagi
    child1['Alpha'] = to_binary(child1['Alpha'])
    child1['Beta'] = to_binary(child1['Beta'])
    child1['Gamma'] = to_binary(child1['Gamma'])

    child2['Alpha'] = to_binary(child2['Alpha'])
    child2['Beta'] = to_binary(child2['Beta'])
    child2['Gamma'] = to_binary(child2['Gamma'])

    # print('Anak 1', child1['Alpha'], child1['Beta'], child1['Gamma'], child1['Fitness'])
    # print('Anak 2', child2['Alpha'], child2['Beta'], child2['Gamma'], child2['Fitness'])

    # Mengembalikan 2 anak
    return {
        'Alpha': child1['Alpha'],
        'Beta': child1['Beta'],
        'Gamma': child1['Gamma'],
        'Fitness': child1['Fitness']
    }, {
        'Alpha': child2['Alpha'],
        'Beta': child2['Beta'],
        'Gamma': child2['Gamma'],
        'Fitness': child2['Fitness']
    }


    # Mengembalikan anak dengan fitness lebih kecil
    # if child1['Fitness'] < child2['Fitness']:
    #     return {
    #     'Alpha': child1['Alpha'],
    #     'Beta': child1['Beta'],
    #     'Gamma': child1['Gamma'],
    #     'Fitness': child1['Fitness']
    # }
    # else:
    #     return {
    #     'Alpha': child2['Alpha'],
    #     'Beta': child2['Beta'],
    #     'Gamma': child2['Gamma'],
    #     'Fitness': child2['Fitness']
    # }

def algoritma_genetika6(train, test,  jumlahKromosom, generations, probability):
    no_improvement_count = 0
    best_fitness_overall = float('inf')
    best_individu_overall = float('inf')
    fitness_history = []
    

    # Membentuk populasi awal
    populasi = pembentukan_populasi_awal(jumlahKromosom)
    print(f'Pembentukan populasi awal: ', populasi)
        
    for generasi in range(1, generations + 1):
        print(f"\nIterasi ke-{generasi}")
        new_populasi = []
        gabung_populasi = []

        # Evaluasi fitness populasi
        print(f"Populasi: {populasi}")
        hitung_nilai_fitness_awal = evaluasi_fitness(populasi, train, test)
        # print(f'Fitness values: ', hitung_nilai_fitness_awal)
        
        # Mutasi atau crossover tergantung probabilitas
        if len(new_populasi) < len(populasi):
            if generasi % probability == 0:  # Mutasi
                parent1 = RouletteWhell(hitung_nilai_fitness_awal)
                offspring1 = mutasi2(train, test, parent1)
                fitness_values_offspring = evaluasi_fitness([offspring1], train, test)
                # print('Daftar dari offspring mutasi : ', fitness_values_offspring)
                # print('Masuk mutasi')
                new_populasi.append(offspring1)
            else:  # Crossover
                parent1 = RouletteWhell(hitung_nilai_fitness_awal)  # Pilih induk pertama
                parent2 = RouletteWhell(hitung_nilai_fitness_awal)  # Pilih induk kedua
                # Crossover: Menggabungkan dua induk untuk menghasilkan dua keturunan
                offspring1, offspring2 = crossover(train, test, parent1, parent2)
                fitness_values_offspring = evaluasi_fitness([offspring1, offspring2], train, test)
                # print("Daftar dari offspring crossofer: ", evaluasi_fitness([offspring1, offspring2], train, test))
                new_populasi.extend(fitness_values_offspring)

        # Cek individu terbaik di generasi ini
        best_individu_generasi = min(new_populasi, key=lambda x: x['Fitness'])

        # Update best fitness global
        if best_individu_generasi['Fitness'] < best_fitness_overall:
            best_fitness_overall = best_individu_generasi['Fitness']
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Hapus dan tambahkan individu baru jika tidak ada perbaikan
        if no_improvement_count >= 12:
            print("Tidak ada perbaikan, menghapus individu dan menambahkan individu secara acak.")

            # Elitisme - simpan top individu
            sorted_populasi = sorted(populasi, key=lambda x: x['Fitness'])
            size_elite = int(jumlahKromosom * 0.3)  # Simpan 30% terbaik
            elite_individuals = sorted_populasi[:size_elite]

            # Tambah individu acak baru
            individu_baru = pembentukan_populasi_awal(jumlahKromosom - size_elite)

            # Update populasi
            populasi = elite_individuals + individu_baru

            # Reset penghitung
            no_improvement_count = 0

        # Gabung populasi lama dengan individu baru dari mutasi/crossover
        gabung_populasi = populasi + new_populasi
        populasi = gabung_populasi[:jumlahKromosom]

  
    print(f"berikut daftar populasi",populasi)
    
    populasi = evaluasi_fitness(populasi, train, test)
    
    individu_fitness_min = min(populasi, key=lambda x: x['Fitness'])
    print("Individu dengan fitness minimum:", individu_fitness_min)


    print(f"\nIndividu Terbaik di Seluruh Generasi: {individu_fitness_min}")
    return individu_fitness_min, generations, fitness_history