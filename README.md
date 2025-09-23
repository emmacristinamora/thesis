# Political Discourse Analysis in U.S. Presidential Debates and Media (1960–2024)

## Abstract
This study explores how U.S. political discourse has evolved from 1960 to 2024, focusing on presidential debates and, to a lesser extent, ideologically aligned media. Using a multi-layered computational pipeline, we investigate changes in political language across themes, sentiment, framing, rhetoric, and ideological content. While the public often perceives a dramatic deterioration in political discourse, our results reveal a more nuanced picture: political language has grown more negative, fragmented, and emotionally charged, but not to the extreme levels often feared. Instead, we identify a steady discursive drift—ideologically stable within parties, but increasingly asymmetrical across them. We argue that these shifts, while moderate in magnitude, have substantial implications for how language encodes political identity, frames public debate, and ultimately sustains or undermines democratic discourse.

---

## Repository Structure

```
thesis/
│
├── data/                       # All datasets (debates, media, labels, intermediate files)
│   ├── debates_full.csv
│   ├── media_balanced.csv
│   ├── distribution_sentiment.csv
│   ├── distribution_emotion.csv
│   ├── distribution_frames.csv
│   ├── debates_metadata.csv
│   └── ... (see folder for full structure)
│
├── notebooks/                 # Jupyter Notebooks for each stage of analysis
│   ├── 01_preprocessing.ipynb
│   ├── 02_topic_modeling.ipynb
│   ├── 03_media_topic_alignment.ipynb
│   ├── 04_framing_analysis.ipynb
│   ├── 05_sentiment_emotion.ipynb
│   ├── 06_rhetorical_analysis.ipynb
│   ├── 07_ideological_drift.ipynb
│   ├── 08_aggregate_results.ipynb
│   └── 09_findings.ipynb
│
├── src/                       # Python scripts used in the pipeline
│   ├── extract_debates.py
│   ├── normalize_utterances.py
│   ├── media_parse_factiva.py
│   ├── media_dataset_cleaning.py
│   ├── rhetoric_api_annotation.py
│   ├── ideology_api_annotation.py
│   └── ... (others for dataset creation)
│
├── results/                   # All generated figures and visualizations
│   ├── final_findings/
│   ├── topic_modeling/
│   └── figures/
│
├── models/                    # Saved model files or embedding arrays
│
├── requirements.txt           # Python package dependencies
├── environment.yml            # Conda environment file
└── README.md                  # This file
```

---

## Instructions

1. Clone this repository
2. Set up the environment using `environment.yml` or `requirements.txt`
3. Run notebooks sequentially from `01_preprocessing.ipynb` to `09_findings.ipynb`
4. All outputs will be saved in `/data` and `/results`

---

## Acknowledgements

The author thanks Professor [Name] for guidance in shaping the project scope and feedback on results. While the thesis was completed independently, the direction and framing benefited from earlier conceptual conversations.

---

## License

This project is intended for academic and non-commercial use.