# Data

The data is **not included** in this repository. It was provided to participants of the GCI World 2026 September course (Matsuo-Iwasawa Lab, The University of Tokyo) and is not redistributed here.

## Expected layout

To run the notebook locally, place the two files here:

```
data/
└── telecom/
    ├── Client.csv   # 100,000 rows × 50 columns: tenure, handset, household, credit class
    └── Record.csv   # 100,000 rows × 51 columns: usage, billing, call quality, churn label
```

The two tables share the key `Customer_ID` and have one row per customer.

## Target

`churn` = 1 if the customer left 31–60 days after the observation date. The file has 49.6% churners, and its sampling method is not documented.
