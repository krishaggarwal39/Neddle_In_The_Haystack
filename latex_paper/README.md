# Research Paper: Recursive Boundary Exposure Retrieval (RBER)
**Format:** IEEE Conference / Journal (Two-Column, 6+ Pages)  
**Authors:** Krish Aggarwal (Corresponding Author) & Veer Vikram Singh  
**Repository:** [https://github.com/krishaggarwal39/Neddle_In_The_Haystack.git](https://github.com/krishaggarwal39/Neddle_In_The_Haystack.git)  
**Permanent Archive (Zenodo DOI):** [10.5281/zenodo.14028391](https://doi.org/10.5281/zenodo.14028391)

---

## 📄 Overview

This directory contains the complete, publication-grade LaTeX research paper for the **Recursive Boundary Exposure Retrieval (RBER)** framework. The paper is formatted for IEEE Conference style and spans over **6 full pages** in standard two-column format, featuring extensive mathematical formulations, formal proofs, algorithmic pseudocode, 5 empirical comparison/ablation tables, and 5 native TikZ/PGF vector figures (including an interactive Web UI workspace mockup).

All diagrams are written in native **LaTeX TikZ / tcolorbox**, ensuring that the project compiles with **zero missing image errors** on Overleaf!

---

## 🚀 How to Open and Compile on Overleaf (1-Click Guide)

### Method 1: Uploading the Pre-Packaged ZIP (Recommended)
1. In this folder, locate `RBER_Research_Paper_Overleaf.zip`.
2. Go to [Overleaf.com](https://www.overleaf.com/) and log in.
3. Click **New Project** $\to$ **Upload Project**.
4. Drag and drop `RBER_Research_Paper_Overleaf.zip` (or browse to select it).
5. Overleaf will automatically unpack all files (`main.tex`, `references.bib`, etc.).
6. Click **Recompile** (Compiler: `pdfLaTeX` or `XeLaTeX`).

### Method 2: Manual File Upload
1. Create a **New Blank Project** on Overleaf named `RBER_Research_Paper`.
2. Delete the default `main.tex` and upload `main.tex` and `references.bib` from this `latex_paper/` folder.
3. Click **Recompile**.

---

## 📑 Required Sections & Structure Checklist

| Section # | Required Section | Description in Paper |
|:---:|:---|:---|
| — | **Title & Author Affiliation** | Krish Aggarwal (CSE Dept, GitHub, Email) & Veer Vikram Singh. |
| — | **Abstract** | Problem definition, burger analogy, core metric claims (100% EM, 88.8% token reduction, $O(\log_3 N)$ convergence). |
| — | **Index Terms / Keywords** | Long-context LLMs, Lost-in-the-Middle, RAG, Positional Attention Bias, Edge Computing. |
| **I** | **Introduction & Background** | The U-shaped attention distribution; primacy/recency bias; shortcomings of RoPE scaling, context shuffling, and dense chunking. Core thesis: *Do not fight positional bias; exploit it.* |
| **II** | **Theoretical Intuition: The Burger Analogy** | The 5-layer burger model (Top Bun, Lettuce, Patty, Cheese, Bottom Bun); physical peeling transformation; elevating hidden interior text to exterior boundaries. Includes **Figure 2**. |
| **III** | **Key Claims & Novel Contributions** | 5 core pillars: Paradigm inversion, chunk-level max-pooling, vector-guided peeling, seam padding, and edge-native zero-leakage execution. |
| **IV** | **Mathematical Methodology & Formal Framework** | Rigorous mathematical equations (1–16): Tri-partition operator $\mathcal{T}$, chunk-level max-pooling $S(Q, B)$, boundary view $V(B)$, deterministic evaluator $\mathcal{C}(B, Q)$, vector guidance $\gamma$, seam padding $\delta_p$, and **Theorem 1 (Logarithmic Boundary Convergence)** with formal proof. |
| **V** | **Algorithmic Framework** | **Algorithm 1:** Recursive Boundary Exposure Retrieval (\RBER{}) line-by-line pseudocode. <br>**Algorithm 2:** Concurrent Parallel Boundary Screening & Speculative Tiering. |
| **VI** | **Experimental Datasets & Benchmark Setup** | 1) Adversarial Needle-in-a-Haystack (NIAH) on 21,408 words (`long.txt`) across 21 relative depth intervals ($0\%$ to $100\%$) for both lexical and paraphrased queries. <br>2) LongBench MultiFieldQA-en (10 contexts: 2,076–7,551 words). <br>3) Real-world multi-page academic coursework PDF corpora. |
| **VII** | **Implementation & System Environment** | System architecture (**Figure 3**); FastAPI asynchronous backend; SQLite Write-Ahead Logging (WAL) with binary vector BLOBs and zlib compression; Ollama local runtime; React 18 / Vite UI. |
| **VII-A** | **Software & Hardware Specifications** | **Table I:** Apple Silicon M-Series MPS acceleration, unified memory bandwidth (68.25 GB/s), power footprint, Python 3.11, PyPDF, HTTPX connection pooling. |
| **VIII** | **Accuracy & Comparison with Existing Methods** | **Table II:** Empirical NIAH comparison across 5 paradigms (Vanilla Llama 3.2 128k, Naive Ternary Cosine, Dense Chunk RAG, BM25 Lexical, and \RBER{}). <br>**Figure 4:** Retrieval Recall vs. Needle Insertion Depth (flat 100% vs U-curve collapse). <br>**Table III:** LongBench MultiFieldQA-en results. In-depth baseline failure analysis. |
| **IX** | **Ablation Studies** | **Table IV:** Quantitative ablation on Parallel Screening, Chunk-Level Max-Pooling vs Mean Pooling, Vector Guidance, and Seam Padding ($\delta_p$). |
| **X** | **Interactive System Workspace** | **Figure 5:** Full two-column high-fidelity vector mockup of the React 18 / Vite web dashboard showing document selection, question input, recursive peeling timeline, and cited evidence card. |
| **XI** | **Discussion & Practical Applications** | Enterprise applications: Commercial contract auditing, Electronic Health Records (EHR), Financial 10-K filings. Limitations: Cross-partition seam cuts, multi-hop reasoning. Privacy advantages: 100% local edge execution, zero cloud data leakage. |
| **XII** | **Code Re-Usability & Reproducibility** | GitHub repository link, Zenodo DOI archive, Apache 2.0 / MIT licensing, and step-by-step CLI commands for exact reproduction. |
| **XIII** | **Conclusion & Future Directions** | Summary of findings; future work on dynamic $k$-ary adaptive partitioning and multi-document tree peeling. |
| — | **Acknowledgment** | Credit to open-source communities (Ollama, FastAPI, Meta Llama). |
| — | **References** | 23+ authentic peer-reviewed BibTeX citations (TACL, NeurIPS, EMNLP, ICLR, SIGIR). |

---

## 📊 Summary of Included Tables & Figures

### Tables
* **Table I:** System Specifications, Hardware Environment, and Execution Profiles.
* **Table II:** Empirical Needle-in-a-Haystack (NIAH) Performance Comparison on 21,408-Word Haystack.
* **Table III:** LongBench MultiFieldQA-en Benchmark Comparison (Token F1, Words Passed, Model Calls, Latency).
* **Table IV:** Ablation Analysis of Architectural Components (Parallel Screening, Max-Pooling, Vector Guidance, Seam Padding).

### Figures
* **Figure 1:** The Classical U-Shaped Attention Distribution vs. \RBER's Effective Boundary Elevation.
* **Figure 2:** The Burger Analogy: Physical Peeling and Boundary Exposure Mechanics.
* **Figure 3:** End-to-End System Architecture Diagram (React UI, FastAPI Gateway, SQLite WAL, Ollama LLM Engine).
* **Figure 4:** Retrieval Accuracy vs. Needle Insertion Depth ($0\%$ to $100\%$) across 21 test intervals.
* **Figure 5:** High-Fidelity Vector Representation of the Interactive \RBER{} Web Workspace UI.

---

## 📦 File Inventory

* `main.tex`: Master LaTeX document.
* `references.bib`: Complete BibTeX bibliography with 23+ authentic citations.
* `README.md`: This documentation and setup guide.
* `RBER_Research_Paper_Overleaf.zip`: Ready-to-upload archive for Overleaf.

