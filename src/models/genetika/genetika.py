import random
from .buildModel import build_and_train_model_for_ga

import sys
sys.set_int_max_str_digits(10240)
def to_binary(value, num_bits=10):
    # Mengalikan nilai dengan 1000 dan mengubahnya menjadi integer
    scaled_value = int(value * (2** num_bits))  
    # Mengubah integer menjadi representasi biner
    return bin(scaled_value)[2:].zfill(num_bits) 

def to_decimal(binary_value):
    return int(binary_value, 2) / (2 ** len(binary_value))  # Mengubah dari biner ke desimal

HYPERPARAMETER_RANGES = {
    'learning_rate': {'type': 'float', 'min': 0.0001, 'max': 0.01, 'num_bits': 7}, # 2^7 = 128 steps
    'cnn_filters_l1': {'type': 'int_choice', 'choices': [16, 32, 64], 'num_bits': 2}, # 2^2 = 4 choices (perlu 3 choices)
    'cnn_kernel_l1': {'type': 'int_choice', 'choices': [3, 5], 'num_bits': 1},
    'lstm_units': {'type': 'int_choice', 'choices': [32, 64, 128], 'num_bits': 2},
    'seq_length': {'type': 'int_choice', 'choices': [10, 20, 30], 'num_bits': 2},
    'dropout_cnn': {'type': 'float', 'min': 0.1, 'max': 0.5, 'num_bits': 4}, # 2^4 = 16 steps
    'activation_cnn': {'type': 'categorical', 'choices': ['relu', 'tanh'], 'num_bits': 1},
    'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd'], 'num_bits': 1}
}

def encode_value(value, param_details):
    """Encode satu nilai hyperparameter ke biner."""
    if param_details['type'] == 'float':
        # Skalakan nilai ke rentang [0, 2^num_bits - 1]
        min_val, max_val = param_details['min'], param_details['max']
        num_bits = param_details['num_bits']
        if value < min_val: value = min_val
        if value > max_val: value = max_val
        scaled_value = int(((value - min_val) / (max_val - min_val)) * (2**num_bits - 1))
        return bin(scaled_value)[2:].zfill(num_bits)
    elif param_details['type'] == 'int_choice':
        # Cari index dari choice, lalu ubah index ke biner
        try:
            idx = param_details['choices'].index(value)
        except ValueError: # Jika value tidak ada di choices, ambil yang pertama
            idx = 0
        return bin(idx)[2:].zfill(param_details['num_bits'])
    elif param_details['type'] == 'categorical':
        try:
            idx = param_details['choices'].index(value)
        except ValueError:
            idx = 0
        return bin(idx)[2:].zfill(param_details['num_bits'])
    return ""

def decode_binary_segment(binary_segment, param_details):
    """Decode segmen biner ke nilai hyperparameter asli."""
    int_val = int(binary_segment, 2)
    if param_details['type'] == 'float':
        min_val, max_val = param_details['min'], param_details['max']
        num_bits = param_details['num_bits']
        # Deskalakan dari [0, 2^num_bits - 1] ke [min_val, max_val]
        return min_val + (int_val / (2**num_bits - 1)) * (max_val - min_val)
    elif param_details['type'] == 'int_choice':
        # Ambil choice berdasarkan index
        return param_details['choices'][int_val]
    elif param_details['type'] == 'categorical':
        return param_details['choices'][int_val]
    return None

def generate_random_hyperparameters():
    """Menghasilkan satu set hyperparameter acak."""
    params = {}
    for name, details in HYPERPARAMETER_RANGES.items():
        if details['type'] == 'float':
            params[name] = random.uniform(details['min'], details['max'])
        elif details['type'] == 'int_choice' or details['type'] == 'categorical':
            params[name] = random.choice(details['choices'])
    return params

def encode_hyperparameters_to_chromosome(hyperparams):
    """Menggabungkan semua hyperparameter yang sudah di-encode menjadi satu chromosome."""
    chromosome_str = ""
    for name, details in HYPERPARAMETER_RANGES.items():
        chromosome_str += encode_value(hyperparams.get(name), details)
    return chromosome_str

def decode_chromosome_to_hyperparameters(chromosome_str):
    """Memecah chromosome menjadi hyperparameter individual."""
    hyperparams = {}
    current_pos = 0
    for name, details in HYPERPARAMETER_RANGES.items():
        num_bits = details['num_bits']
        segment = chromosome_str[current_pos : current_pos + num_bits]
        hyperparams[name] = decode_binary_segment(segment, details)
        current_pos += num_bits
    return hyperparams

def pembentukan_populasi_awal(jumlahKromosom):
    populasi = []
    for _ in range(jumlahKromosom):
        random_params = generate_random_hyperparameters()
        chromosome = encode_hyperparameters_to_chromosome(random_params)
        populasi.append({
            'chromosome': chromosome,
            'hyperparameters': random_params, # Simpan juga hasil decode untuk kemudahan
            'fitness': float('-inf') # Inisialisasi fitness (asumsi akurasi, ingin dimaksimalkan)
        })
    return populasi


def evaluasi_fitness(populasi, train_X, train_y, val_X, val_y, epochs_ga=10):
    print("\nMemulai Evaluasi Fitness untuk Populasi...")
    for i, individu in enumerate(populasi):
        print(f" Mengevaluasi individu {i+1}/{len(populasi)}...")
        # Decode chromosome jika belum ada dictionary 'hyperparameters' atau jika ingin memastikan konsistensi
        # Jika 'hyperparameters' sudah diisi saat pembuatan individu dan tidak diubah, tidak perlu decode ulang.
        # Namun, lebih aman untuk selalu decode dari 'chromosome' sebagai satu-satunya sumber kebenaran gen.
        decoded_params = decode_chromosome_to_hyperparameters(individu['chromosome'])
        individu['hyperparameters'] = decoded_params # Update agar konsisten

        fitness_value = build_and_train_model_for_ga(
            individu['hyperparameters'],
            train_X, train_y,
            val_X, val_y,
            epochs_ga=epochs_ga
        )
        individu['fitness'] = fitness_value
        print(f" Individu {i+1} Fitness: {fitness_value:.4f}")
    print("Evaluasi Fitness Selesai.\n")
    return populasi # Mengembalikan populasi dengan fitness yang terupdate

def roulette_wheel_selection(populasi):

    """Seleksi individu berdasarkan fitness (asumsi fitness lebih tinggi lebih baik)."""
    total_fitness = sum(ind['fitness'] for ind in populasi if ind['fitness'] > float('-inf')) # Hindari fitness buruk

    # Jika semua fitness sangat buruk atau nol
    if total_fitness == 0:
        return random.choice(populasi) # Pilih secara acak jika tidak ada fitness positif

    # Hitung probabilitas relatif
    probabilities = []
    for ind in populasi:
        if ind['fitness'] > float('-inf'):
            probabilities.append(ind['fitness'] / total_fitness)
        else:
            probabilities.append(0) # Individu dengan fitness sangat buruk tidak dipilih

    # Hitung probabilitas kumulatif
    cumulative_probabilities = []
    current_sum = 0
    for p in probabilities:
        current_sum += p
        cumulative_probabilities.append(current_sum)

    # Pilih individu
    r = random.random()
    for i, cumulative_prob in enumerate(cumulative_probabilities):
        if r <= cumulative_prob:
            return populasi[i]
    
    return populasi[-1] # Fallback, seharusnya tidak sering terjadi jika fitness > 0
def mutation_operator(chromosome, mutation_rate=0.1):
    """Melakukan bit-flip mutation pada chromosome."""
    mutated_chromosome_list = list(chromosome)
    for i in range(len(mutated_chromosome_list)):
        if random.random() < mutation_rate:
            mutated_chromosome_list[i] = '1' if mutated_chromosome_list[i] == '0' else '0'
    return "".join(mutated_chromosome_list)


def crossover_operator(parent1_chromosome, parent2_chromosome, crossover_rate=0.8):
    """Melakukan two-point crossover pada dua chromosome parent."""
    if random.random() > crossover_rate:
        # Tidak ada crossover, offspring adalah salinan parent
        return parent1_chromosome, parent2_chromosome

    len_chromo = len(parent1_chromosome)
    if len_chromo < 2: # Tidak bisa crossover jika panjang < 2
        return parent1_chromosome, parent2_chromosome

    # Two-point crossover
    pt1, pt2 = sorted(random.sample(range(len_chromo), 2))
    
    offspring1_chromo = parent1_chromosome[:pt1] + parent2_chromosome[pt1:pt2] + parent1_chromosome[pt2:]
    offspring2_chromo = parent2_chromosome[:pt1] + parent1_chromosome[pt1:pt2] + parent2_chromosome[pt2:]
    
    return offspring1_chromo, offspring2_chromo

def algoritma_genetika_yolo_cnn_lstm(train_X, train_y, val_X, val_y,
                                     jumlah_kromosom, generations,
                                     crossover_rate, mutation_rate,
                                     epochs_per_eval, # Epoch untuk melatih model di setiap evaluasi fitness
                                     elite_size=2): # Jumlah individu elit
    
    # 1. Inisialisasi Populasi
    populasi = pembentukan_populasi_awal(jumlah_kromosom)
    
    # 2. Evaluasi Fitness Awal
    print("Memulai Evaluasi Fitness untuk Populasi Awal...")
    populasi = evaluasi_fitness(populasi, train_X, train_y, val_X, val_y, epochs_ga=epochs_per_eval)
    
    best_individu_overall = None
    best_fitness_overall = float('-inf') # Asumsi akurasi (maksimalkan)
    fitness_history = [] # Untuk melacak fitness terbaik per generasi

    # Inisialisasi best_individu_overall dengan yang terbaik dari populasi awal
    for ind in populasi:
        if ind['fitness'] > best_fitness_overall:
            best_fitness_overall = ind['fitness']
            best_individu_overall = ind.copy() # Salin individu
    
    if best_individu_overall:
        print(f"Best Fitness Awal: {best_fitness_overall:.4f}, Hyperparams: {best_individu_overall['hyperparameters']}")
    else:
        print("Tidak ada individu valid di populasi awal.")
        return None, []

    fitness_history.append(best_fitness_overall)
    no_improvement_count = 0

    # 3. Loop Generasi
    for gen in range(generations):
        print(f"\n--- Generasi {gen + 1}/{generations} ---")
        
        new_population = []
        
        # a.i Elitisme
        populasi.sort(key=lambda x: x['fitness'], reverse=True) # Urutkan dari fitness tertinggi
        for i in range(min(elite_size, len(populasi))): # Pastikan elite_size tidak melebihi ukuran populasi
            if populasi[i]['fitness'] > float('-inf'): # Hanya bawa elit yang valid
                 new_population.append(populasi[i].copy())
        
        # a.ii Loop untuk Mengisi Sisa Populasi Baru
        while len(new_population) < jumlah_kromosom:
            parent1 = roulette_wheel_selection(populasi)
            parent2 = roulette_wheel_selection(populasi)
            
            offspring1_chromo, offspring2_chromo = crossover_operator(
                parent1['chromosome'], parent2['chromosome'], crossover_rate
            )
            
            offspring1_chromo_mutated = mutation_operator(offspring1_chromo, mutation_rate)
            offspring2_chromo_mutated = mutation_operator(offspring2_chromo, mutation_rate)
            
            # Buat individu offspring baru
            new_population.append({
                'chromosome': offspring1_chromo_mutated,
                'hyperparameters': decode_chromosome_to_hyperparameters(offspring1_chromo_mutated),
                'fitness': float('-inf') # Akan dievaluasi
            })
            if len(new_population) < jumlah_kromosom:
                new_population.append({
                    'chromosome': offspring2_chromo_mutated,
                    'hyperparameters': decode_chromosome_to_hyperparameters(offspring2_chromo_mutated),
                    'fitness': float('-inf')
                })
        
        # b. Ganti Populasi Lama dengan Populasi Baru
        populasi = new_population
        
        # c. Evaluasi Fitness Populasi Baru
        print(f"Memulai Evaluasi Fitness untuk Generasi {gen + 1}...")
        populasi = evaluasi_fitness(populasi, train_X, train_y, val_X, val_y, epochs_ga=epochs_per_eval)
        
        # d. Lacak Individu Terbaik
        current_gen_best_fitness = float('-inf')
        current_gen_best_individu = None
        for ind in populasi:
            if ind['fitness'] > current_gen_best_fitness:
                current_gen_best_fitness = ind['fitness']
            if ind['fitness'] > best_fitness_overall:
                best_fitness_overall = ind['fitness']
                best_individu_overall = ind.copy() # Salin individu
                no_improvement_count = 0 # Reset jika ada perbaikan
                print(f"  ** Individu Terbaik Baru Ditemukan di Generasi {gen+1}! Fitness: {best_fitness_overall:.4f} **")
                print(f"     Hyperparams: {best_individu_overall['hyperparameters']}")

        if current_gen_best_fitness > float('-inf'):
             fitness_history.append(current_gen_best_fitness)
             print(f"Fitness Terbaik Generasi {gen+1}: {current_gen_best_fitness:.4f}")
        else: # Jika tidak ada fitness valid di generasi ini (jarang terjadi)
            fitness_history.append(fitness_history[-1] if fitness_history else float('-inf'))


        if best_individu_overall and best_individu_overall['fitness'] == current_gen_best_fitness :
             pass # Tidak ada perbaikan dari yang sudah ada
        else:
             no_improvement_count +=1


        # e. Mekanisme Stagnasi (Contoh sederhana)
        if no_improvement_count >= 10: # Jika tidak ada perbaikan dalam 10 generasi
            print("Tidak ada perbaikan signifikan, mencoba meningkatkan laju mutasi sementara.")
            # Anda bisa meningkatkan mutation_rate untuk beberapa generasi berikutnya
            # atau melakukan diversifikasi populasi seperti di kode Holt-Winters Anda.
            # Untuk sekarang, kita reset saja agar tidak terjebak selamanya.
            # Jika Anda ingin mekanisme diversifikasi, bisa diadaptasi dari kode sebelumnya.
            no_improvement_count = 0 # Reset untuk menghindari loop
            # Contoh diversifikasi sederhana: ganti sebagian populasi dengan individu acak baru
            num_to_replace = int(0.3 * jumlah_kromosom) # Ganti 30%
            populasi.sort(key=lambda x: x['fitness']) # Urutkan dari fitness terendah
            new_random_individuals = pembentukan_populasi_awal(num_to_replace)
            # Evaluasi individu baru ini sebelum dimasukkan
            new_random_individuals = evaluasi_fitness(new_random_individuals,  train_X, train_y, val_X, val_y, epochs_ga=epochs_per_eval)

            populasi = populasi[num_to_replace:] + new_random_individuals # Ganti yang terburuk
            print("Diversifikasi populasi dilakukan.")


    print("\n--- Optimasi Algoritma Genetika Selesai ---")
    if best_individu_overall:
        print(f"Individu Terbaik Keseluruhan Ditemukan:")
        print(f"  Fitness (Val Accuracy): {best_individu_overall['fitness']:.4f}")
        print(f"  Hyperparameters: {best_individu_overall['hyperparameters']}")
        print(f"  Chromosome: {best_individu_overall['chromosome']}")
    else:
        print("Tidak ada individu terbaik yang ditemukan.")
        
    return best_individu_overall, fitness_history