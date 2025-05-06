import random

def mutasi(individu, prob_mutasi=0.01):
    individu_baru = {}

    for kunci, gen in individu.items():
        if kunci == 'Fitness':
            individu_baru[kunci] = None  # nilai fitness akan dihitung ulang
            continue

        gen_baru = ""
        for bit in gen:
            if random.random() < prob_mutasi:
                gen_baru += '1' if bit == '0' else '0'  # flip bit
            else:
                gen_baru += bit

        individu_baru[kunci] = gen_baru

    return individu_baru
