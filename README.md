<div align="center">

<img src="https://www.thiranex.in/favicon.ico" width="60" alt="Thiranex Logo" onerror="this.style.display='none'"/>

# 🎓 Data Analytics Internship
### @ [Thiranex](https://www.thiranex.in) · June – July 2026

[![Intern](https://img.shields.io/badge/Intern-Ranjith%20S-blue?style=flat-square)](https://www.thiranex.in)
[![ID](https://img.shields.io/badge/ID-THX--JUN1226--263-orange?style=flat-square)](#)
[![Duration](https://img.shields.io/badge/Duration-12%20Jun%20–%2011%20Jul%202026-green?style=flat-square)](#)
[![Mode](https://img.shields.io/badge/Mode-Remote%20%2F%20Project--Based-purple?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Completed%20✓-brightgreen?style=flat-square)](#)

</div>

---

## 📖 My Story — One Month That Changed How I Think About Data

I still remember staring at a blank Python file on my first day, not sure where to begin. Thiranex had assigned me four real-world data analytics projects, and all I had was a Kaggle account, a laptop that overheats, and the kind of nervous energy that makes you open ten browser tabs at once.

**Week 1** hit hard. The Sales & Revenue Dashboard sounded simple — until I realised the dataset had inconsistent date formats, mixed currency columns, and product names that had been typed in six different ways. I spent two full evenings just on data cleaning before writing a single line of visualisation code. Late nights at 1 AM with matplotlib documentation open on one screen and Stack Overflow on the other became my routine. My first chart looked like a kindergarten art project. By the fourth attempt, it actually told a story.

**Week 2** was Customer Segmentation — and the moment everything clicked. Running K-Means for the first time and watching 1,000 customers organise themselves into distinct behavioural groups felt genuinely magical. I remember calling a friend just to say *"I think I understand why companies know what you want to buy before you do."* The RFM analysis took me three days to get right. The silhouette score kept coming back wrong and I couldn't figure out why — turned out I was scaling after splitting instead of before. A three-hour bug that taught me more than three weeks of tutorials ever could.

**Week 3** was Predictive Analytics. Time-series forecasting with SARIMA looked intimidating in every tutorial I'd seen. I downloaded the Air Passengers dataset from Kaggle, built the pipeline, and watched the model predict seasonal spikes I hadn't told it about — just learned from the data itself. That moment felt like solving a real industry problem. Not a textbook exercise. A real one.

**Week 4** brought Data Cleaning & Reporting Automation with the Superstore dataset. By this point I had developed my own workflow: explore → clean → engineer features → model → visualise → report. What used to take me two days I completed in six hours. The growth felt tangible.

Thirty days. Four projects. Hundreds of lines of code. Countless error messages. And the kind of practical experience that no classroom ever gave me.

---

## 🗂️ Projects Completed

| # | Project | Dataset | Tools | Status |
|---|---------|---------|-------|--------|
| 1 | [Sales & Revenue Dashboard](#-project-1--sales--revenue-dashboard) | Superstore Sales | Python, Matplotlib, Pandas | ✅ Completed |
| 2 | [Customer Segmentation](#-project-2--customer-segmentation) | E-Commerce Retail | Scikit-learn, K-Means, RFM | ✅ Completed |
| 3 | [Predictive Analytics](#-project-3--predictive-analytics) | Air Passengers (Kaggle) | SARIMA, XGBoost, Regression | ✅ Completed |
| 4 | [Data Cleaning & Reporting Automation](#-project-4--data-cleaning--reporting-automation) | Superstore CSV | Pandas, Matplotlib | ✅ Completed |

---

## 📊 Project 1 — Sales & Revenue Dashboard

**Due:** 08 Jun 2026 · **Submitted:** On Track

### What I Built
An interactive sales dashboard that imports raw CSV data and generates visual KPI summaries covering total revenue, regional performance, top products, and monthly trends.

### Key Features
- Automated CSV ingestion with encoding detection
- KPI cards: Total Sales, Revenue, Profit Margin, Order Count
- Charts: Monthly trend lines, category breakdowns, regional comparisons
- Top 10 products by revenue ranked visually

### What I Learned
- How to structure a reporting pipeline from raw data to insight
- Matplotlib figure layouts using GridSpec for dashboard-style outputs
- How businesses actually use data to track performance

### Dataset
- **Source:** Kaggle — Superstore Sales Dataset
- **Size:** 9,994 rows × 21 columns
- **Key columns:** Order Date, Sales, Profit, Category, Region, Segment

```python
# Core pipeline pattern I developed
df = pd.read_csv("data.csv", encoding="latin-1")
df["Order Date"] = pd.to_datetime(df["Order Date"])
monthly = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
```

---

## 👥 Project 2 — Customer Segmentation

**Due:** 15 Jun 2026 · **Submitted:** On Track

### What I Built
A full customer segmentation system using RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering to group 1,000+ customers into distinct behavioural segments.

### Key Features
- RFM feature engineering from raw transaction data
- Optimal K selection using Elbow method + Silhouette score
- PCA dimensionality reduction for 2D cluster visualisation
- Segment labelling: Champions, Loyal, At Risk, Hibernating, New

### Results

| Segment | Customers | Avg Spend | Revenue Share |
|---------|-----------|-----------|---------------|
| 🏆 Champions | 159 | $48,343 | 67% |
| 💛 Loyal Customers | 268 | $11,102 | 26% |
| 🔵 Potential | 386 | $1,946 | 6.5% |
| ⚠️ At Risk | 187 | $468 | 0.8% |

### Key Insight
The top 16% of customers drove 67% of total revenue — a classic power-law distribution that completely changed how I think about customer value.

### What I Learned
- K-Means clustering and the importance of feature scaling *before* splitting
- How RFM analysis is used in real retail and e-commerce businesses
- PCA for visualising high-dimensional cluster data

---

## 📈 Project 3 — Predictive Analytics

**Due:** 22 Jun 2026 · **Submitted:** On Track

### What I Built
A multi-model forecasting pipeline comparing Linear Regression, Random Forest, XGBoost, and SARIMA on time-series data, with a 12-month future forecast and full evaluation dashboard.

### Models Compared

| Model | MAE | MAPE | R² |
|-------|-----|------|----|
| Linear Regression | 16.9 | 3.8% | 0.917 |
| Random Forest | 34.3 | 7.0% | 0.603 |
| XGBoost | — | — | — |
| **SARIMA** | **6.6** | **4.0%** | **0.970** |

### Key Features
- Lag features: 1-month, 3-month, 12-month lookbacks
- Rolling averages for trend smoothing
- Time-series cross-validation (no data leakage)
- 12-month rolling forecast with 80% confidence intervals

### Dataset
- **Source:** [Kaggle — Air Passengers Dataset](https://www.kaggle.com/datasets/rakannimer/air-passengers)
- **Size:** 144 monthly records (1949–1960)
- **Why this dataset:** Clean, real historical data with clear seasonality — ideal for learning time-series patterns

### What I Learned
- Why time-series splits differ from standard train/test splits
- How SARIMA handles non-stationary data through differencing
- The difference between overfitting on training data and real predictive power

---

## 🧹 Project 4 — Data Cleaning & Reporting Automation

**Due:** 29 Jun 2026 · **Submitted:** On Track · *Under Review*

### What I Built
A fully automated data cleaning and reporting pipeline that takes a raw messy CSV and produces a clean dataset plus a complete business intelligence report — with zero manual intervention.

### Cleaning Pipeline

```
Raw CSV → Detect Encoding → Remove Duplicates → Fix Nulls
       → Parse Dates → Standardise Columns → Remove Bad Values
       → Save cleaned_data.csv + Generate report.png
```

### What Gets Fixed Automatically

| Issue | Solution |
|-------|----------|
| Duplicate rows | `drop_duplicates()` with before/after count |
| Numeric nulls | Filled with column median |
| Text nulls | Filled with column mode |
| Bad date formats | `pd.to_datetime(errors='coerce')` |
| Inconsistent column names | Lowercased, spaces → underscores |
| Negative sales values | Flagged and removed |

### Dataset
- **Source:** [Kaggle — Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- **File:** `Sample - Superstore.csv`
- **Size:** 9,994 rows × 21 columns

### What I Learned
- How to build reusable, generalised cleaning functions
- The real cost of dirty data in business reporting
- Automation mindset: write it once, run it forever

---

## 🛠️ Tech Stack

```
Languages     Python 3.x
Libraries     Pandas · NumPy · Matplotlib · Scikit-learn · Statsmodels · XGBoost
Data Sources  Kaggle (real-world public datasets)
Environment   VS Code · Jupyter Notebook · Remote / Local
```

---

## 📁 Repository Structure

```
📦 thiranex-data-analytics-internship/
├── 📂 project1_sales_dashboard/
│   ├── sales_dashboard.py
│   └── report.png
├── 📂 project2_customer_segmentation/
│   ├── customer_segmentation.py
│   ├── customer_segments.csv
│   └── customer_segmentation_dashboard.png
├── 📂 project3_predictive_analytics/
│   ├── predictive_analytics.py
│   ├── forecast_12months.csv
│   └── predictive_analytics_output.png
├── 📂 project4_data_cleaning/
│   ├── data_cleaning_report.py
│   ├── cleaned_superstore.csv
│   └── report.png
└── README.md
```

---

## 🚀 How to Run Any Project

```bash
# 1. Clone the repo
git clone https://github.com/your-username/thiranex-data-analytics-internship.git
cd thiranex-data-analytics-internship

# 2. Install dependencies (once)
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost openpyxl

# 3. Download datasets from Kaggle and place in the project folder

# 4. Run any project
python project1_sales_dashboard/sales_dashboard.py
python project2_customer_segmentation/customer_segmentation.py
python project3_predictive_analytics/predictive_analytics.py
python project4_data_cleaning/data_cleaning_report.py
```

---

## 💡 Key Takeaways

After a month of working with real datasets and building end-to-end pipelines, here is what I actually learned — not from tutorials, but from doing:

**On Data:** Real data is always messier than you expect. The cleaning always takes longer than the modelling. And the insights are always hiding in the details you almost skipped.

**On Models:** A simple model that you understand beats a complex model that you don't. Linear Regression with good features often outperforms fancy ensembles on small datasets.

**On Process:** The most valuable skill isn't knowing which algorithm to use — it's knowing how to ask the right question from the data in front of you.

**On Growth:** I started this internship Googling "how to read a CSV in Python." I finished it building multi-model forecasting pipelines with automated reporting. The gap between those two things is thirty days of consistent, self-directed work.

---

## 🔮 What's Next

The skills I built here are the foundation for everything I want to do next:

- **NLP & Text Analytics** — sentiment analysis, topic modelling
- **Machine Learning Engineering** — deploying models as APIs
- **Business Intelligence** — Power BI / Tableau dashboards
- **Deep Learning** — neural networks for complex pattern recognition

Every future project starts from what I learned at Thiranex.

---

## 🙏 Acknowledgements

<div align="center">

A genuine thank you to **[Thiranex](https://www.thiranex.in)** for designing an internship that actually teaches.

No spoon-feeding. No pre-cleaned toy datasets. Real problems, real data from Kaggle, real pressure of weekly deadlines — and the freedom to figure things out through self-learning. That's what made this different.

The best learning happens when you're slightly out of your depth, working on something that actually matters.
This internship was exactly that.

---

**Ranjith S** · Intern ID: `THX-JUN1226-263`
*Data Analytics Intern · Thiranex · Jun – Jul 2026*

[![Thiranex](https://img.shields.io/badge/Powered%20by-Thiranex-blue?style=flat-square)](https://www.thiranex.in)

</div>
