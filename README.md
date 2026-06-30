# 🧬 Genome Embedding & Similarity Workshop

## Overview

In this practical, we will explore how DNA sequences can be transformed into numerical representations using a pretrained transformer model (**Nucleotide Transformer**). 

We will then use these embeddings to:
1. Visualise genomic relationships using dimensionality reduction
2. Quantify similarity using k-nearest neighbours (k-NN)
3. Evaluate whether learned embeddings capture biological structure

---

## Learning objectives

By the end of this session you will be able to:

- Convert nucleotide sequences into embedding vectors using a pretrained transformer
- Reduce high-dimensional biological data using UMAP / PCA
- Interpret clustering patterns in genomic space
- Train and evaluate a simple k-NN classifier on embeddings

---

## Provided pre-computed genomes

Group 1: Pseudomonadota Around Escherichia coli

| Species                  | Relationship                                             | Expected Similarity |
| ------------------------ | -------------------------------------------------------- | ------------------- |
| *Escherichia fergusonii* | Same genus                                               | Very high           |
| *Shigella flexneri*      | Essentially *E. coli*-like (genomically extremely close) | Very high           |
| *Salmonella enterica*    | Same family                                              | High                |
| *Klebsiella pneumoniae*  | Same family                                              | High                |
| *Enterobacter cloacae*   | Same family                                              | Moderate-high       |
| *Citrobacter freundii*   | Same family                                              | Moderate-high       |
| *Yersinia pestis*        | Same family                                              | Moderate            |
| *Pseudomonas aeruginosa* | Same phylum, different order                             | Lower               |
| *Vibrio cholerae*        | Same phylum                                              | Lower               |

Group 2: Bacillota Around Staphylococcus aureus

| Species                        | Relationship                 | Expected Similarity |
| ------------------------------ | ---------------------------- | ------------------- |
| *Staphylococcus epidermidis*   | Same genus                   | Very high           |
| *Staphylococcus saprophyticus* | Same genus                   | Very high           |
| *Staphylococcus haemolyticus*  | Same genus                   | High                |
| *Macrococcus caseolyticus*     | Sister genus                 | High                |
| *Bacillus subtilis*            | Same phylum                  | Moderate            |
| *Listeria monocytogenes*       | Same phylum                  | Moderate            |
| *Enterococcus faecalis*        | Same phylum                  | Moderate-low        |
| *Streptococcus pneumoniae*     | Same phylum, different order | Moderate-low        |

---

## 0. Prerequisites

This workshop is written in Python and runs in a Docker container with all dependencies preinstalled.

Before starting this practical, ensure you have:

### Required software

- Docker installed on your machine  
  - Mac: https://docs.docker.com/desktop/install/mac-install/
  - Windows: https://docs.docker.com/desktop/install/windows-install/
  - Linux: https://docs.docker.com/engine/install/

- A modern web browser (Chrome / Firefox recommended)

### Optional (but useful)

- Git (for cloning workshop materials)
- ~4–8 GB free disk space for Docker image

---


## 1. Setup: Pull and run the workshop environment

Pull the git repo:

```bash
github.com/jgillespie93/bioinformatics-workshop

```
Navigate to the bioinformatics workshop folder in terminal
```bash
cd Downloads/Bioinformatics_Workshop_Gillespie

```
Pull the prebuilt Docker image:

```bash
docker pull bioinf-workshop
```
Start Jupyter Lab:

```bash
docker run -p 8888:8888 -v $(pwd):/workspace bioinf-workshop

```
Then in your browser open
```bash
http://localhost:8888

```

A Jupyter notebook instance should open in your browser (a web based interactive computing platform). Open the notebook from the left hand panel

```bash
Bioinformatics_workshop.ipynb
```
and follow the rest of the instructions from there.
