import torch
from Bio import SeqIO
import numpy as np

def embed_genome(
    fasta_file,
    tokenizer,
    model,
    window_size=2000,
    stride=1000,
    batch_size=32
):

    records = list(SeqIO.parse(fasta_file, "fasta"))

    genome_seq = str(records[0].seq).upper()

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