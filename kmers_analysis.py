from Bio import SeqIO
import numpy as np
from collections import Counter
from itertools import product


def get_kmer_counts(seq, k=6):
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    return counts

def build_vocab(k=6):
    return [''.join(p) for p in product("ACGT", repeat=k)]

k = 6
vocab = build_vocab(k)
kmer_index = {kmer: i for i, kmer in enumerate(vocab)}
dim = len(vocab)  # 4^k = 4096

def window_to_vector(seq, k=6):
    vec = np.zeros(dim, dtype=np.float32)
    
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        idx = kmer_index.get(kmer)
        if idx is not None:
            vec[idx] += 1

    # normalize (important for comparison)
    vec /= np.sum(vec) + 1e-9
    return vec

def genome_to_kmers(
    fasta_file,
    window_size=2000,
    stride=750,
    k=6
):

    records = list(
        SeqIO.parse(fasta_file, "fasta")
    )

    genome_seq = str(
        records[0].seq
    ).upper()

    windows = []

    for start in range(
        0,
        len(genome_seq) - window_size + 1,
        stride
    ):
        windows.append(
            genome_seq[start:start+window_size]
        )

    kmer_matrix = np.zeros(
        (len(windows), dim),
        dtype=np.float32
    )

    for i, w in enumerate(windows):
        kmer_matrix[i] = window_to_vector(
            w,
            k=k
        )

    return kmer_matrix