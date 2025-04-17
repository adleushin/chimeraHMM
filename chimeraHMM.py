import sys
import numpy as np
import os
from Bio import SeqIO
import argparse
import matplotlib.pyplot as plt
import json

def read_fasta_clean(filename):
    """Читает fasta и возвращает строку без не-ATGC."""
    record = next(SeqIO.parse(filename, "fasta"))
    seq = str(record.seq).upper()
    seq = ''.join([c for c in seq if c in "ATGC"])
    return seq

def gc_content(seq):
    """Вычисляет долю GC."""
    if not seq: return 0.0
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq)

def make_chimera(seq1, seq2, total_length=1000, mean_frag_len=300, seed=None):
    """Создает химерную последовательность из seq1 и seq2."""
    if seed is not None:
        np.random.seed(seed)
    s = ''
    true_states = []
    sources = [seq1, seq2]
    state = np.random.choice([0,1])
    while len(s) < total_length:
        curseq = sources[state]
        frag_len = min(int(np.random.exponential(mean_frag_len)) + 1, total_length - len(s))
        max_start = len(curseq) - frag_len
        if max_start <= 0:
            continue
        start = np.random.randint(0, max_start+1)
        frag = curseq[start:start+frag_len]
        if set(frag) - set("ATGC"):
            continue
        s += frag
        true_states.extend([state+1]*frag_len)
        state = 1 - state # чередовать
    return s[:total_length], true_states[:total_length]

def get_emiss_probs(gc_content):
    at = (1 - gc_content) / 2
    gc = gc_content / 2
    return {'A': at, 'T': at, 'G': gc, 'C': gc}
    
def viterbi(seq, emi1, emi2, p_stay, p_switch):
    n = len(seq)
    delta = np.zeros((2, n))
    psi = np.zeros((2, n), dtype=int)
    b = [np.log(emi1[seq[0]]), np.log(emi2[seq[0]])]
    delta[0, 0] = np.log(0.5) + b[0]
    delta[1, 0] = np.log(0.5) + b[1]
    for t in range(1, n):
        for j in (0, 1):
            emi = emi1 if j == 0 else emi2
            p_emit = np.log(emi[seq[t]])
            trans0 = delta[0, t - 1] + (np.log(p_stay) if j == 0 else np.log(p_switch))
            trans1 = delta[1, t - 1] + (np.log(p_switch) if j == 0 else np.log(p_stay))
            if trans0 > trans1:
                delta[j, t] = trans0 + p_emit
                psi[j, t] = 0
            else:
                delta[j, t] = trans1 + p_emit
                psi[j, t] = 1
    path = np.zeros(n, dtype=int)
    path[-1] = 0 if delta[0, -1] > delta[1, -1] else 1
    for t in range(n - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]
    return np.array([x+1 for x in path])

def save_output(filename, states, probs):
    d = {
        'states': ''.join(str(x) for x in states),
        'emiss_probs': probs
    }
    with open(filename, 'w') as fout:
        json.dump(d, fout, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Chimera HMM Viterbi Demo")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    parser_run = subparsers.add_parser('run', help='Run standard mode')
    parser_run.add_argument('--dbdir', required=True, help='Папка с хромосомами (FASTAs) двух организмов')
    parser_run.add_argument('--length', type=int, default=10000)
    parser_run.add_argument('--fragmean', type=int, default=300)
    parser_run.add_argument('--seed', type=int, default=42)

    parser_test = subparsers.add_parser('test', help='Apply Viterbi to input with models from dir')
    parser_test.add_argument('input', help='FASTA файл для теста')
    parser_test.add_argument('--dbdir', required=True, help='Папка с хромосомами (FASTAs) двух организмов')
    parser_test.add_argument('--fragmean', type=int, default=300)
    parser_test.add_argument('--out', type=str, default='viterbi_test_output.json', help='JSON файл вывода')

    args = parser.parse_args()

    if args.mode == 'run':
        db_files = [os.path.join(args.dbdir, fn) for fn in os.listdir(args.dbdir)
                    if fn.lower().endswith('.fasta') or fn.lower().endswith('.fa')]
        if len(db_files) != 2:
            sys.exit("Нужно два fasta-файла в dbdir")
        # Take first two
        fasta1, fasta2 = db_files[:2]
        seq1 = read_fasta_clean(fasta1)
        seq2 = read_fasta_clean(fasta2)
        print(f"GC seq1 ({fasta1}): {gc_content(seq1):.4f}")
        print(f"GC seq2 ({fasta2}): {gc_content(seq2):.4f}")

        chimera, state_path = make_chimera(seq1, seq2, total_length=args.length,
                                           mean_frag_len=args.fragmean, seed=args.seed)
        emi1 = get_emiss_probs(gc_content(seq1))
        emi2 = get_emiss_probs(gc_content(seq2))
        avglen = args.fragmean
        p_stay = (avglen - 1)/avglen
        p_switch = 1.0/avglen

        result_path = viterbi(chimera, emi1, emi2, p_stay, p_switch)
        state_path_arr = np.array(state_path)

        print("\nПервые 100 состояний (истинные): ")
        print(''.join(str(x) for x in state_path_arr[:100]))
        print("\nПервые 100 состояний (Viterbi):   ")
        print(''.join(str(x) for x in result_path[:100]))
        acc = (result_path == state_path_arr).mean()
        print(f"\nОбщая точность: {acc:.3%}")

        plt.figure(figsize=(15,4))
        plt.plot(state_path_arr[:], label="Истинные состояния", lw=2)
        plt.plot(result_path[:], label="Viterbi предсказание", lw=1.5, ls='--')
        plt.title("Скрытые состояния true vs Viterbi")
        plt.xlabel("Позиция (t)")
        plt.ylabel("Состояние (1 или 2)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif args.mode == 'test':
        # 1. Read all fasta files from dbdir - must be at least 2
        db_files = [os.path.join(args.dbdir, fn) for fn in os.listdir(args.dbdir)
                    if fn.lower().endswith('.fasta') or fn.lower().endswith('.fa')]
        if len(db_files) != 2:
            sys.exit("Нужно два fasta-файла в dbdir")
        # Take first two
        fasta1, fasta2 = db_files[:2]
        seq1 = read_fasta_clean(fasta1)
        seq2 = read_fasta_clean(fasta2)
        #Эмиссионные вероятности для этих двух моделей
        emi1 = get_emiss_probs(gc_content(seq1))
        emi2 = get_emiss_probs(gc_content(seq2))
        avglen = args.fragmean
        p_stay = (avglen - 1)/avglen
        p_switch = 1.0/avglen
        # 2. Ваша последовательность
        testseq = read_fasta_clean(args.input)
        result_path = viterbi(testseq, emi1, emi2, p_stay, p_switch)

        out_probs = {
            'file1': os.path.basename(fasta1), 'emi1': emi1,
            'file2': os.path.basename(fasta2), 'emi2': emi2,
            'p_stay': p_stay, 'p_switch': p_switch,
        }
        # 3. Output
        outpath = os.path.join(os.path.dirname(args.input), args.out)
        save_output(outpath, result_path, out_probs)
        print(f"Скрытые состояния и вероятности записаны в {outpath}")

if __name__ == '__main__':
    main()