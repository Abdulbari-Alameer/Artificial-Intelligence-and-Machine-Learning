
# Anomaly Scoring using KNN Distance and Density Deviation

## Overview

This project implements an anomaly scoring method based on two components:

- Normalized k-nearest neighbor (KNN) distance
- Normalized deviation from a density threshold

The final anomaly score is computed as:

\[
\text{Score} = w_1 \times D + w_2 \times R
\]

where:

- **D** is the normalized distance score.
- **R** is the normalized density-deviation score.
- **w₁ + w₂ = 1**, ensuring that the final score remains within the range **[0, 1]**.

## Features

- KNN-based anomaly detection
- Robust percentile-based normalization
- Density threshold using the median
- Weighted anomaly score in the range [0,1]
- Automatic anomaly threshold estimation
- Visualization of anomaly scores
- Scatter plot highlighting detected anomalies

## Project Structure

```
Anomaly-Scoring-KNN-Density/
│
├── anomaly_scorer.py
├── report.docx
├── requirements.txt
├── README.md
└── outputs/
    ├── score_distribution.png
    └── scatter_anomaly.png
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python anomaly_scorer.py
```

The generated figures will be saved automatically inside the `outputs` folder.

## Output

The program generates:

- Anomaly score distribution plot
- Spatial anomaly scatter plot
- Threshold value
- Number of detected anomalies
- Minimum, maximum, and mean anomaly scores

## Author

**Abdulbari Mohammed Alameer**  
Master of Technology (M.Tech)  
Department of Computer Science and Engineering  
Tezpur University
