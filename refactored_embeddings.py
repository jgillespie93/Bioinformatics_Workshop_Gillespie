
#==================================
#J. Gillespie
#==================================
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Bio import SeqIO
#import umap currently broken
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True)

max_length = tokenizer.model_max_length

# genomes = {
#     "E_coli_E11":
#         "/Users/jamesgillespie/Downloads/Bioinformatics_Workshop_Gillespie/genomes/Escherichia_coli_strain_E11.fasta",

#     "E_coli_E21":
#         "/Users/jamesgillespie/Downloads/Bioinformatics_Workshop_Gillespie/genomes/Escherichia_coli_strain_E21.fasta",

#     "S_aureus":
#         "/Users/jamesgillespie/Downloads/Bioinformatics_Workshop_Gillespie/genomes/Staphylococcus_aureus_LACPHL-SPEC-2025-00335.fasta",

#     "S_aureus2":
#         "/Users/jamesgillespie/Downloads/Bioinformatics_Workshop_Gillespie/genomes/Staphylococcus_aureus_FT526_174.fasta",

#      "S_aureus3":
#          "/Users/jamesgillespie/Downloads/Bioinformatics_Workshop_Gillespie/genomes/Staphylococcus_aureus_LACPHL-SPEC-2025-00347.fasta"

#     # add more here later
#     # "Influenza": "...",
#     # "Yeast": "...",
# }

import os

base_dir = "/Users/jamesgillespie/Downloads/Bioinformatics_Workshop_Gillespie/new genomes"
#groups = ["Pseudomonadota", "Bacillota", "outgroups"]
groups = ["tb"]

genomes = {}

for group in groups:
    folder = os.path.join(base_dir, group)

    for filename in os.listdir(folder):
        if filename.endswith((".fa", ".fasta", ".fna")):
            name = os.path.splitext(filename)[0]
            genomes[name] = os.path.join(folder, filename)

def load_genome(path):
    seq = []

    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip().upper())

    return "".join(seq)

def embed_genome(
    fasta_file,
    tokenizer,
    model,
    window_size=2000,
    stride=1000,
    batch_size=32
):

    # records = list(SeqIO.parse(fasta_file, "fasta"))

    # genome_seq = str(records[0].seq).upper()
    genome_seq = load_genome(fasta_file)

    windows = []

    for start in range(
        0,
        len(genome_seq) - window_size + 1,
        stride
    ):
        windows.append(
            genome_seq[start:start + window_size]
        )

    print(f"{fasta_file}")
    print(f"Windows: {len(windows)}")

    embeddings = []

    model.eval()

    for i in range(0, len(windows), batch_size):

        batch = windows[i:i+batch_size]

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(
                tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True
            )

        hidden = outputs.hidden_states[-1]

        mask = tokens["attention_mask"].unsqueeze(-1)

        pooled = (
            (hidden * mask).sum(dim=1)
            /
            mask.sum(dim=1).clamp(min=1e-9)
        )

        embeddings.append(
            pooled.cpu().numpy()
        )

    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    return embeddings


genome_embeddings = {}

for name, path in genomes.items():

    print(f"Embedding {name}")

    genome_embeddings[name] = embed_genome(
        path,
        tokenizer,
        model
    )

    print(
        name,
        genome_embeddings[name].shape
    )


all_embeddings = np.vstack(
    list(genome_embeddings.values())
)

labels = []

for name, emb in genome_embeddings.items():

    labels.extend(
        [name] * len(emb)
    )

labels = np.array(labels)


#==================================
#Plotting data and understanding distribution
#==================================


pca = PCA(n_components=2)

X_all = pca.fit_transform(
    all_embeddings
)

plt.figure(figsize=(8,6))

for genome in np.unique(labels):

    mask = labels == genome

    plt.scatter(
        X_all[mask,0],
        X_all[mask,1],
        s=5,
        alpha=0.6,
        label=genome
    )

plt.legend()
plt.title("Genome Embedding PCA")
plt.show()


#==================================
#Check distances between embeddings
#==================================

D = cosine_distances(
    all_embeddings
)

species = np.unique(labels)


for s1 in species:

    for s2 in species:

        mask1 = labels == s1
        mask2 = labels == s2

        block = D[
            np.ix_(mask1, mask2)
        ]

        print(
            f"{s1} vs {s2}: "
            f"{block.mean():.4f}"
        )


all_embeddings = []
genome_labels = []
species_labels = []

for genome_name, emb in genome_embeddings.items():

    all_embeddings.append(emb)

    genome_labels.extend(
        [genome_name] * len(emb)
    )

    if genome_name.startswith("E"):
        species = "E_coli"
    else:
        species = "S_aureus"

    species_labels.extend(
        [species] * len(emb)
    )

all_embeddings = np.vstack(all_embeddings)

genome_labels = np.array(genome_labels)
species_labels = np.array(species_labels)


within_genome = []
within_species = []
between_species = []

n = len(genome_labels)

for i in range(n):
    for j in range(i+1, n):

        same_genome = (
            genome_labels[i] ==
            genome_labels[j]
        )

        same_species = (
            species_labels[i] ==
            species_labels[j]
        )

        if same_genome:

            within_genome.append(D[i,j])

        elif same_species:

            within_species.append(D[i,j])

        else:

            between_species.append(D[i,j])



plt.figure(figsize=(8,5))

plt.hist(
    within_genome,
    bins=50,
    alpha=0.5,
    label="Within genome"
)

plt.hist(
    within_species,
    bins=50,
    alpha=0.5,
    label="Between genomes, same species"
)

plt.hist(
    between_species,
    bins=50,
    alpha=0.5,
    label="Between species"
)

plt.xlabel("Cosine distance")
plt.ylabel("Count")
plt.title("Embedding distance structure")
plt.legend()
plt.show()


#=====================
#refactored fast version
#=====================

all_embeddings = np.vstack(list(genome_embeddings.values()))

genome_labels = []
species_labels = []

for genome_name, emb in genome_embeddings.items():
    genome_labels.extend([genome_name] * len(emb))

    if genome_name.startswith("E"):
        species = "E_coli"
    else:
        species = "S_aureus"

    species_labels.extend([species] * len(emb))

genome_labels = np.array(genome_labels)
species_labels = np.array(species_labels)

n_pairs = 200000  # adjust for smoothness vs speed

within_genome = []
within_species = []
between_species = []

n = len(all_embeddings)

for _ in range(n_pairs):

    i, j = np.random.randint(0, n, 2)

    if i == j:
        continue

    d = cosine_distances(
        all_embeddings[i].reshape(1, -1),
        all_embeddings[j].reshape(1, -1)
    )[0, 0]

    same_genome = genome_labels[i] == genome_labels[j]
    same_species = species_labels[i] == species_labels[j]

    if same_genome:
        within_genome.append(d)

    elif same_species:
        within_species.append(d)

    else:
        between_species.append(d)




plt.figure(figsize=(8,5))

plt.hist(within_genome, bins=50, alpha=0.5, label="Within genome")
plt.hist(within_species, bins=50, alpha=0.5, label="Same species (different genome)")
plt.hist(between_species, bins=50, alpha=0.5, label="Between species")

plt.xlabel("Cosine distance")
plt.ylabel("Count")
plt.title("Embedding distance structure (sampled pairs)")
plt.legend()
plt.show()


