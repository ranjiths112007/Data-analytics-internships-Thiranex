<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://www.thiranex.in/favicon.ico">
  <img src="https://www.thiranex.in/favicon.ico" width="52" alt="Thiranex">
</picture>

<br/>

```
T H I R A N E X   ×   D A T A   A N A L Y T I C S
```

# Internship Portfolio

**Ranjith S** · `THX-JUN1226-263` · June 12 – July 11, 2026

[![Domain](https://img.shields.io/badge/Domain-Data%20Analytics-0d1117?style=flat-square&labelColor=58a6ff&color=0d1117)](https://www.thiranex.in)
[![Mode](https://img.shields.io/badge/Mode-Remote%20·%20Project%20Based-0d1117?style=flat-square&labelColor=3fb950&color=0d1117)](#)
[![Tasks](https://img.shields.io/badge/Tasks-4%20of%204%20Completed-0d1117?style=flat-square&labelColor=ffa657&color=0d1117)](#projects)
[![Progress](https://img.shields.io/badge/Progress-80%25-0d1117?style=flat-square&labelColor=bc8cff&color=0d1117)](#)

</div>

---

## The Month

I joined Thiranex on June 12 with a basic understanding of Python and a vague idea of what data analytics actually meant in practice. What followed was the most concentrated learning period of my life.

Four projects. Four real datasets downloaded from Kaggle. Four weeks of building things from scratch, breaking them, fixing them at midnight, and slowly developing the kind of intuition that no tutorial ever gave me.

**Week 1.** The Sales & Revenue Dashboard sounded manageable until I opened the Superstore dataset and saw what real business data actually looks like — inconsistent date formats, unlabelled columns, product names typed seventeen different ways. I spent the first two days cleaning before writing a single chart. My first dashboard attempt looked like a bar graph drawn by someone who had never seen a bar graph. By the fourth iteration, it was telling a story. That shift — from displaying data to communicating insight — was the first thing I genuinely learned.

**Week 2.** Customer Segmentation. Running K-Means clustering for the first time and watching 1,000 customers self-organise into distinct behavioural groups felt like unlocking something. The silhouette score kept coming back wrong and I couldn't figure out why for three hours — I was scaling features after the train/test split instead of before, leaking information into the model. One bug. Three hours. More understanding than a week of reading.

**Week 3.** Predictive Analytics with SARIMA. Time-series forecasting had always looked intimidating in every resource I'd encountered. I downloaded the Air Passengers dataset, built the pipeline step by step, and watched the model predict seasonal peaks it had never been told about — purely learned from the historical pattern. That moment felt like solving something real.

**Week 4.** Data Cleaning & Reporting Automation. I built a pipeline that takes a raw messy CSV and produces a clean dataset plus a full business report automatically. No manual steps. By this point what had taken me two days in week one took six hours. I could feel the difference.

Thirty days. Four projects. Hundreds of lines of code. The kind of practical foundation that actually transfers.

---

## Projects

<br/>

### 01 · Sales & Revenue Analysis Dashboard

> Build a dashboard to analyse sales and revenue data from raw business records.

| | |
|---|---|
| **Dataset** | [Superstore Sales — Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) · `Sample - Superstore.csv` |
| **Tools** | Python · Pandas · Matplotlib · OpenPyXL |
| **Output** | `superstore_dashboard.xlsx` |
| **Due** | 08 Jun 2026 · Submitted on track |

**What it does**

Imports raw CSV transaction data, computes KPIs (total sales, revenue, profit margin, order count), and generates a multi-panel Excel dashboard with charts covering monthly trends, category performance, regional breakdowns, and top products.

**What I learned**

How businesses actually track performance. The difference between a chart that shows data and a dashboard that drives decisions. Why axis labels and colour choices matter more than the model behind them.

```
Sales & Revenue Analysis Dashboard/
├── Sample - Superstore.csv.zip   ← raw dataset (zipped)
└── superstore_dashboard.xlsx     ← output dashboard
```

---

### 02 · Customer Segmentation

> Segment customers based on purchasing behaviour using RFM analysis and K-Means clustering.

| | |
|---|---|
| **Dataset** | [E-Commerce Retail — UCI / Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data) · transactional data |
| **Tools** | Python · Scikit-learn · Pandas · Matplotlib |
| **Output** | `customer_segmentation_dashboard.png` · `customer_segments.csv` |
| **Due** | 15 Jun 2026 · Submitted on track |

**What it does**

Engineers RFM (Recency, Frequency, Monetary) features from raw transactions, finds the optimal number of clusters using Elbow + Silhouette methods, assigns every customer to a segment, and generates a 8-panel visualisation dashboard covering cluster scatter, revenue heatmap, segment distribution, and actionable recommendations.

**Results**

| Segment | Customers | Avg Spend | Revenue Share |
|---|---|---|---|
| Champions | 159 | $48,343 | 67% |
| Loyal Customers | 268 | $11,102 | 26% |
| Potential | 386 | $1,946 | 6.5% |
| At Risk | 187 | $468 | 0.8% |

The top 16% of customers drove 67% of revenue. That single insight changes how a business allocates its retention budget.

**Dashboard**

![Customer Segmentation Dashboard](Customer%20Segmentation%20Project/customer_segmentation_dashboard.png)

```
Customer Segmentation Project/
├── customer_segmentation.py             ← full pipeline
├── customer_segmentation_dashboard.png  ← output visualisation
└── customer_segments.csv                ← labelled customer data
```

---

### 03 · Predictive Analytics Using Historical Data

> Build a multi-model forecasting pipeline to predict future trends from historical time-series data.

| | |
|---|---|
| **Dataset** | [Air Passengers — Kaggle](https://www.kaggle.com/datasets/rakannimer/air-passengers) · `AirPassengers.csv` · 144 monthly records |
| **Tools** | Python · Statsmodels (SARIMA) · Scikit-learn · XGBoost · Pandas |
| **Output** | `predictive_analytics_output.png` |
| **Due** | 22 Jun 2026 · Submitted on track |

**What it does**

Loads the Air Passengers dataset, engineers lag and rolling-average features, trains four models side-by-side (Linear Regression, Random Forest, XGBoost, SARIMA), evaluates each on a held-out 24-month test set, then generates a 12-month rolling forecast with confidence intervals.

**Model Comparison (test set)**

| Model | MAE | MAPE | R² |
|---|---|---|---|
| Linear Regression | 16.9 | 3.8% | 0.917 |
| Random Forest | 34.3 | 7.0% | 0.603 |
| **SARIMA** | **6.6** | **4.0%** | **0.970** |

SARIMA captured seasonality that the ML models missed — the right tool for the job isn't always the most complex one.

**Dashboard**

![Predictive Analytics Dashboard](Predictive%20Analytics%20Using%20Historical%20Data/predictive_analytics_output.png)

```
Predictive Analytics Using Historical Data/
├── predictive_analytics.py              ← full pipeline
├── AirPassengers.csv                    ← dataset (included)
└── predictive_analytics_output.png      ← output dashboard
```

---

### 04 · Data Cleaning & Reporting Automation

> Automate the full data cleaning and business reporting workflow — zero manual steps.

| | |
|---|---|
| **Dataset** | [Superstore Sales — Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) · `Sample - Superstore.csv` |
| **Tools** | Python · Pandas · NumPy · Matplotlib |
| **Output** | `cleaned_superstore.csv` · `report.png` |
| **Due** | 29 Jun 2026 · Submitted on track · Under review |

**What it does**

A single script that ingests a raw messy CSV, runs a full cleaning pipeline (duplicates, nulls, date parsing, column standardisation, bad value removal), exports the clean data, and generates a complete 6-panel business intelligence report — all automatically.

**Cleaning pipeline**

```
Raw CSV
  → detect encoding (UTF-8 / Latin-1)
  → drop duplicate rows
  → fill numeric nulls with column median
  → fill text nulls with column mode
  → parse date columns
  → standardise column names
  → remove negative sales
  → export cleaned_superstore.csv
  → generate report.png
```

**Dashboard**

![Data Cleaning Report](Data%20Cleaning%20%26%20Reporting%20Automation/report.png)

```
Data Cleaning & Reporting Automation/
├── data_cleaning_report.py    ← automation script
├── Sample - Superstore.csv    ← raw input dataset
├── cleaned_superstore.csv     ← cleaned output
└── report.png                 ← generated report
```

---

## Datasets

All datasets are publicly available on Kaggle. Download and place in the matching project folder before running.

| Project | Dataset | Link |
|---|---|---|
| Sales Dashboard | Superstore Sales | [kaggle.com/datasets/vivek468/superstore-dataset-final](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) |
| Customer Segmentation | E-Commerce Retail (UCI) | [kaggle.com/datasets/carrie1/ecommerce-data](https://www.kaggle.com/datasets/carrie1/ecommerce-data) |
| Predictive Analytics | Air Passengers | [kaggle.com/datasets/rakannimer/air-passengers](https://www.kaggle.com/datasets/rakannimer/air-passengers) |
| Data Cleaning | Superstore Sales | [kaggle.com/datasets/vivek468/superstore-dataset-final](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) |

> The Air Passengers CSV is already included in the repo. All others must be downloaded from Kaggle (free account required).

---

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/Data-analytics-internships-Thiranex.git
cd Data-analytics-internships-Thiranex

# Install dependencies (once)
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost openpyxl

# Run any project
python "Customer Segmentation Project/customer_segmentation.py"
python "Predictive Analytics Using Historical Data/predictive_analytics.py"
python "Data Cleaning & Reporting Automation/data_cleaning_report.py"
```

Each script is self-contained. Drop the dataset CSV in the same folder as the script, run it, outputs generate in the same directory.

---

## Stack

```
Python 3.x          core language
Pandas              data manipulation and cleaning
NumPy               numerical computing
Matplotlib          visualisation and dashboards
Scikit-learn        K-Means clustering, regression models, metrics
Statsmodels         SARIMA time-series modelling
XGBoost             gradient boosted forecasting
OpenPyXL            Excel output generation
Kaggle              all real-world datasets
```

---

## Repository Structure

```
Data-analytics-internships-Thiranex-main/
│
├── Sales & Revenue Analysis Dashboard/
│   ├── Sample - Superstore.csv.zip
│   └── superstore_dashboard.xlsx
│
├── Customer Segmentation Project/
│   ├── customer_segmentation.py
│   ├── customer_segmentation_dashboard.png
│   └── customer_segments.csv
│
├── Predictive Analytics Using Historical Data/
│   ├── predictive_analytics.py
│   ├── AirPassengers.csv
│   └── predictive_analytics_output.png
│
├── Data Cleaning & Reporting Automation/
│   ├── data_cleaning_report.py
│   ├── Sample - Superstore.csv
│   ├── cleaned_superstore.csv
│   └── report.png
│
├── Thiranex_OfferLetter_Ranjith_S_THX-JUN1226-263.pdf
└── README.md
```

---

## What's Next

The foundation is in place. What I'm building toward:

- **NLP & Text Analytics** — sentiment analysis, topic modelling on unstructured data
- **ML Engineering** — deploying models as REST APIs, not just notebooks
- **Business Intelligence** — Power BI and Tableau for stakeholder-facing dashboards
- **Deep Learning** — neural networks for pattern recognition at scale

Every project from here starts from what was built during this internship.

---

<div align="center">

---

**Thank you, Thiranex.**

This internship was designed around self-learning and real data — no hand-holding, no toy examples. Just actual problems, actual datasets, and the freedom to figure things out. That approach works. The gap between where I started and where I finished is the proof.

---

**Ranjith S**
Data Analytics Intern · Thiranex · Jun – Jul 2026
`THX-JUN1226-263`

[thiranex.in](https://www.thiranex.in) · Verified by Thiranex Verification Cell

</div>
