"""
Data Cleaning & Reporting Automation
=====================================
Dataset : Sample - Superstore.csv  (same folder as this script)
Run     : python data_cleaning_report.py
Outputs : cleaned_superstore.csv  +  report.png
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ── CONFIG ────────────────────────────────────────────────────────────
INPUT_FILE   = "Sample - Superstore.csv"
OUTPUT_CSV   = "cleaned_superstore.csv"
OUTPUT_CHART = "report.png"
# ─────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════
try:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(INPUT_FILE, encoding="latin-1")

print(f"✅  Loaded  →  {df.shape[0]} rows  ×  {df.shape[1]} columns")


# ══════════════════════════════════════════════
# 2. CLEANING  (with before/after log)
# ══════════════════════════════════════════════
log = {}
raw_rows = len(df)

# 2a. Duplicates
dupes = df.duplicated().sum()
df.drop_duplicates(inplace=True)
log["Duplicates removed"] = dupes

# 2b. Missing values — count before fix
missing_before = df.isnull().sum()

# Fill numeric nulls with median
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Fill object nulls with mode
for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

log["Nulls fixed"] = int(missing_before.sum())

# 2c. Parse dates
for col in ["Order Date", "Ship Date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# 2d. Strip whitespace from strings
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

# 2e. Standardise column names
df.columns = (df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("-", "_"))

# 2f. Remove negative sales (if any)
if "sales" in df.columns:
    bad_sales = (df["sales"] < 0).sum()
    df = df[df["sales"] >= 0]
    log["Negative sales removed"] = int(bad_sales)

# 2g. Derived columns useful for reporting
if "order_date" in df.columns:
    df["year"]    = df["order_date"].dt.year
    df["month"]   = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

log["Clean rows"] = len(df)

print(f"\n🧹 Cleaning Summary:")
for k, v in log.items():
    print(f"   {k:<28} {v}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n💾  Cleaned file saved → {OUTPUT_CSV}")


# ══════════════════════════════════════════════
# 3. REPORT DATA
# ══════════════════════════════════════════════
# Guard: only compute stats for columns that exist
has = lambda c: c in df.columns

sales_by_cat     = df.groupby("category")["sales"].sum().sort_values() if has("category") else None
sales_by_region  = df.groupby("region")["sales"].sum().sort_values()   if has("region")  else None
profit_by_cat    = df.groupby("category")["profit"].sum()               if has("category") else None
monthly_sales    = df.groupby("year_month")["sales"].sum()              if has("year_month") else None
top_products     = (df.groupby("product_name")["sales"]
                     .sum().sort_values(ascending=False).head(10))      if has("product_name") else None
segment_share    = df["segment"].value_counts()                         if has("segment")  else None


# ══════════════════════════════════════════════
# 4. DASHBOARD
# ══════════════════════════════════════════════
BG, CARD, LINE = "#0d1117", "#161b22", "#21262d"
C = ["#58a6ff","#3fb950","#ffa657","#f85149","#bc8cff","#d29922"]
money = FuncFormatter(lambda x, _: f"${x/1000:.0f}k")

fig = plt.figure(figsize=(22, 16), facecolor=BG)
fig.suptitle("Superstore — Automated Data Cleaning & Business Report",
             fontsize=21, fontweight="bold", color="white", y=0.97)
gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.52, wspace=0.35,
                       left=0.06, right=0.97, top=0.92, bottom=0.05)

def sa(ax, title="", xl="", yl=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors="#8b949e", labelsize=9)
    for sp in ax.spines.values(): sp.set_color(LINE)
    if title: ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    if xl:    ax.set_xlabel(xl,    color="#8b949e", fontsize=9)
    if yl:    ax.set_ylabel(yl,    color="#8b949e", fontsize=9)


# ── KPI cards (row 0) ─────────────────────────────────────────────────
total_sales   = df["sales"].sum()   if has("sales")   else 0
total_profit  = df["profit"].sum()  if has("profit")  else 0
total_orders  = df["order_id"].nunique() if has("order_id") else len(df)
margin        = (total_profit / total_sales * 100) if total_sales else 0

kpis = [
    ("Total Sales",    f"${total_sales:,.0f}",   "#58a6ff"),
    ("Total Profit",   f"${total_profit:,.0f}",  "#3fb950"),
    ("Profit Margin",  f"{margin:.1f}%",          "#ffa657"),
    ("Orders",         f"{total_orders:,}",       "#bc8cff"),
    ("Rows Cleaned",   f"{raw_rows - len(df):,} fixed", "#f85149"),
]
for i, (label, val, col) in enumerate(kpis):
    ax = fig.add_axes([0.06 + i*0.185, 0.885, 0.165, 0.065])
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(1.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.70, label, ha="center", color="#8b949e", fontsize=9,  transform=ax.transAxes)
    ax.text(0.5, 0.25, val,   ha="center", color=col,       fontsize=14, fontweight="bold", transform=ax.transAxes)


# ── Panel 1: Monthly sales trend (full width) ─────────────────────────
ax1 = fig.add_subplot(gs[0, :])
sa(ax1, "Monthly Sales Trend", "Month", "Sales")
if monthly_sales is not None and len(monthly_sales) > 1:
    x = range(len(monthly_sales))
    ax1.fill_between(x, monthly_sales.values, alpha=0.2, color=C[0])
    ax1.plot(x, monthly_sales.values, color=C[0], lw=2)
    step = max(1, len(monthly_sales)//12)
    ax1.set_xticks(list(x)[::step])
    ax1.set_xticklabels(monthly_sales.index[::step], rotation=45, ha="right",
                        color="#8b949e", fontsize=8)
    ax1.yaxis.set_major_formatter(money)


# ── Panel 2: Sales by category ────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
sa(ax2, "Sales by Category", "", "Sales ($)")
if sales_by_cat is not None:
    bars = ax2.barh(sales_by_cat.index, sales_by_cat.values,
                    color=C[:len(sales_by_cat)], edgecolor=LINE, height=0.55)
    for b, v in zip(bars, sales_by_cat.values):
        ax2.text(b.get_width()+500, b.get_y()+b.get_height()/2,
                 f"${v:,.0f}", va="center", color="white", fontsize=9)
    ax2.xaxis.set_major_formatter(money)


# ── Panel 3: Sales by region ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
sa(ax3, "Sales by Region", "Region", "Sales ($)")
if sales_by_region is not None:
    bars = ax3.bar(sales_by_region.index, sales_by_region.values,
                   color=C[:len(sales_by_region)], edgecolor=LINE, width=0.55)
    for b, v in zip(bars, sales_by_region.values):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+200,
                 f"${v:,.0f}", ha="center", color="white", fontsize=8)
    ax3.yaxis.set_major_formatter(money)
    ax3.set_xticklabels(sales_by_region.index, rotation=20, ha="right", color="#8b949e")


# ── Panel 4: Customer segment pie ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
sa(ax4, "Customer Segments")
if segment_share is not None:
    wedges, _, autotexts = ax4.pie(
        segment_share.values, labels=segment_share.index,
        colors=C[:len(segment_share)], autopct="%1.1f%%",
        pctdistance=0.78, wedgeprops=dict(edgecolor=BG, linewidth=1.5))
    for t in autotexts: t.set_color("white"); t.set_fontsize(9)
    for t in ax4.texts:  t.set_color("#8b949e"); t.set_fontsize(9)


# ── Panel 5: Top 10 products ──────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
sa(ax5, "Top 10 Products by Sales", "", "Sales ($)")
if top_products is not None:
    tp = top_products.sort_values()
    bars = ax5.barh(tp.index, tp.values, color=C[1], edgecolor=LINE, height=0.65)
    for b, v in zip(bars, tp.values):
        ax5.text(b.get_width()+200, b.get_y()+b.get_height()/2,
                 f"${v:,.0f}", va="center", color="white", fontsize=8)
    ax5.xaxis.set_major_formatter(money)
    ax5.tick_params(axis="y", labelsize=8)


# ── Panel 6: Profit by category ───────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
sa(ax6, "Profit by Category", "Category", "Profit ($)")
if profit_by_cat is not None:
    colors_p = [C[1] if v >= 0 else C[3] for v in profit_by_cat.values]
    bars = ax6.bar(profit_by_cat.index, profit_by_cat.values,
                   color=colors_p, edgecolor=LINE, width=0.55)
    for b, v in zip(bars, profit_by_cat.values):
        ax6.text(b.get_x()+b.get_width()/2,
                 b.get_height() + (200 if v>=0 else -1500),
                 f"${v:,.0f}", ha="center", color="white", fontsize=9)
    ax6.yaxis.set_major_formatter(money)
    ax6.set_xticklabels(profit_by_cat.index, rotation=15, ha="right", color="#8b949e")
    ax6.axhline(0, color="#8b949e", lw=0.8, ls="--")


plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

print(f"📊  Report saved    → {OUTPUT_CHART}")
print(f"\n📋  Final Summary:")
print(f"    Total Sales   : ${total_sales:>12,.2f}")
print(f"    Total Profit  : ${total_profit:>12,.2f}")
print(f"    Profit Margin : {margin:>11.1f}%")
print(f"    Clean Rows    : {len(df):>12,}")
print(f"\n✅  All done!")
