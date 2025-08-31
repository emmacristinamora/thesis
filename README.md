# Debates, Discourse, and Media  
*A Computational Analysis of Temporal Shifts in U.S. Presidential Debates and Media Framing Across the Political Spectrum*

## 🧠 Overview

This project explores how key political concepts (e.g. *freedom*, *security*, *truth*) are framed and rhetorically deployed in U.S. presidential debates over time, and how these framings align with or diverge from coverage in ideologically distinct media outlets.  
It is the basis for my Bachelor's thesis in Artificial Intelligence & Economics at Bocconi University.

---

## 🗂️ Project Structure

| Folder/File | Description |
|-------------|-------------|
| `01_preprocessing.ipynb` | Load, clean, and structure debate and media texts; tokenization and segmentation |
| `02_exploration_topics_terms.ipynb` | Frequency analysis, topic modeling (BERTopic), and TF-IDF to identify key terms |
| `03_framing_analysis.ipynb` | Zero-shot classification of utterances into high-level frames (e.g., security, morality) |
| `04_rhetoric_modeling.ipynb` | Rhetorical strategy classification (acclaim, attack, defense) using Benoit’s Functional Theory |
| `05_sentiment_emotion.ipynb` | Sentiment and emotion detection around key concepts |
| `06_semantic_polarization.ipynb` | Embedding-based analysis of ideological drift across time and party |
| `07_debate_media_alignment.ipynb` | Compare debate and media framing using zero-shot and SBERT similarity |
| `08_visualization_dashboard.ipynb` | Generate final plots for inclusion in the thesis |

---

## 📁 Data Sources

- **Debates**: U.S. Commission on Presidential Debates (public transcripts)
- **Media**: TBD — selection in progress with attention to political balance and topic coverage

> 🔒 *Raw data is excluded from version control (`/data/`) for size and privacy reasons.*

---

## 🛠️ Requirements

To replicate the analysis, create a virtual environment and install dependencies from `requirements.txt` (to be added). Core packages include:

- `transformers` (for zero-shot models)
- `sentence-transformers`
- `bertopic`
- `scikit-learn`
- `spaCy`
- `pandas`, `matplotlib`, `seaborn`

---

## 📌 Notes

- Notebooks are modular and ordered by number.  
- Each notebook is designed to be run independently once the preprocessed data is available.  
- The analysis focuses on interpretability, reproducibility, and robustness.

---

## 📄 License

This repository is for academic use only and part of a Bachelor's thesis submitted to Bocconi University, 2025.

---

## 🙋‍♀️ Author

**Emma Mora**  
BSc Mathematical and Computing Sciences for Artificial Intelligence, Bocconi University  
Email: *emma.mora@studbocconi.it*  