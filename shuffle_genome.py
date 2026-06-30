from collections import defaultdict, Counter
import random
from Bio import SeqIO
import numpy as np

def build_debruijn_graph(sequence, k):
    """
    Build a de Bruijn graph from a sequence.
    Nodes: (k-1)-mers
    Edges: k-mers
    """
    graph = defaultdict(list)

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        prefix = kmer[:-1]
        suffix = kmer[1:]
        graph[prefix].append(suffix)

    return graph


def eulerian_trail(graph):
    """
    Hierholzer's algorithm for Eulerian path in a directed graph.
    """
    graph = {node: edges[:] for node, edges in graph.items()}

    # start node: pick one with outgoing edges
    start = next(iter(graph))

    stack = [start]
    path = []

    while stack:
        node = stack[-1]

        if node in graph and graph[node]:
            stack.append(graph[node].pop())
        else:
            path.append(stack.pop())

    return path[::-1]


def reconstruct_sequence(path):
    """
    Reconstruct sequence from (k-1)-mer path.
    """
    seq = path[0]
    for node in path[1:]:
        seq += node[-1]
    return seq


def kmer_preserving_shuffle(sequence, k=6, seed=None):
    """
    Shuffle sequence while preserving exact k-mer counts.
    """
    if seed is not None:
        random.seed(seed)

    if len(sequence) < k:
        return sequence

    graph = build_debruijn_graph(sequence, k)

    path = eulerian_trail(graph)

    shuffled = reconstruct_sequence(path)

    return shuffled

def load_fasta_all_contigs(fasta_path):
    sequences = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        
    # concatenate contigs into one pseudo-chromosome
    return "".join(sequences)

def extract_kmers(seq, k):
    return Counter(seq[i:i+k] for i in range(len(seq) - k + 1))


fasta_file = "/Users/jamesgillespie/Downloads/Bioinformatics_Workshop_Gillespie/new genomes/ecoli/Escherichia coli 316.fasta"
records = list(SeqIO.parse(fasta_file, "fasta"))

# If the FASTA contains a single genome
#needs to be uppercase or tokenizer doesn't work
genome_seq = load_fasta_all_contigs(fasta_file)
genome_seq = str(genome_seq).upper()

original = genome_seq
shuffled = kmer_preserving_shuffle(original, k=6)

print("Same 6-mer distribution:",
      extract_kmers(original, 6) == extract_kmers(shuffled, 6))

