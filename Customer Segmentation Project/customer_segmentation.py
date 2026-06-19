
"""
Customer Segmentation Project
Using RFM Analysis + K-Means Clustering on E-Commerce Data
Dataset structure mirrors UCI Online Retail Dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE REALISTIC E-COMMERCE DATASET
# ─────────────────────────────────────────────
np.random.seed(42)

def generate_ecommerce_data(n_customers=1000, n_transactions=15000):
    print("📦 Generating e-commerce dataset...")
    
    countries = ['United Kingdom', 'Germany', 'France', 'Netherlands', 'Spain', 'Belgium']
    country_weights = [0.60, 0.10, 0.10, 0.08, 0.07, 0.05]
    
    customer_ids = np.arange(10000, 10000 + n_customers)
    
    # Assign customer profiles (latent segments)
    profiles = np.random.choice(['champions', 'loyal', 'at_risk', 'hibernating', 'new'],
                                 size=n_customers,
                                 p=[0.15, 0.25, 0.20, 0.20, 0.20])
    
    profile_params = {
        'champions':   dict(freq=(20, 50),  spend=(80, 300), recency=(1, 30)),
        'loyal':       dict(freq=(10, 20),  spend=(50, 150), recency=(10, 60)),
        'at_risk':     dict(freq=(5, 10),   spend=(30, 100), recency=(60, 150)),
        'hibernating': dict(freq=(1, 4),    spend=(10, 50),  recency=(150, 365)),
        'new':         dict(freq=(1, 3),    spend=(20, 80),  recency=(1, 45)),
    }
    
    rows = []
    snapshot = datetime(2011, 12, 10)
    
    for cid, profile in zip(customer_ids, profiles):
        params = profile_params[profile]
        n_orders = np.random.randint(*params['freq'])
        country = np.random.choice(countries, p=country_weights)
        
        for _ in range(n_orders):
            days_ago = np.random.randint(*params['recency'])
            invoice_date = snapshot - timedelta(days=days_ago)
            n_items = np.random.randint(1, 8)
            
            for _ in range(n_items):
                unit_price = round(np.random.uniform(*params['spend']) / n_items, 2)
                quantity   = np.random.randint(1, 15)
                rows.append({
                    'InvoiceNo':   f"INV{np.random.randint(500000, 599999)}",
                    'StockCode':   f"SKU{np.random.randint(1000, 9999)}",
                    'Description': f"Product {np.random.randint(1, 200)}",
                    'Quantity':    quantity,
                    'InvoiceDate': invoice_date,
                    'UnitPrice':   unit_price,
                    'CustomerID':  cid,
                    'Country':     country,
                })
    
    df = pd.DataFrame(rows)
    print(f"   ✅ {len(df):,} transactions | {df['CustomerID'].nunique():,} customers")
    return df

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
def clean_data(df):
    print("\n🧹 Cleaning data...")
    before = len(df)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df.dropna(subset=['CustomerID'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    print(f"   Removed {before - len(df):,} invalid rows → {len(df):,} clean records")
    return df

# ─────────────────────────────────────────────
# 3. RFM FEATURE ENGINEERING
# ─────────────────────────────────────────────
def compute_rfm(df, snapshot_date=None):
    print("\n📐 Computing RFM features...")
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('InvoiceNo',   'nunique'),
        Monetary  = ('TotalPrice',  'sum'),
    ).reset_index()
    
    # Add extra behavioral features
    rfm['AvgOrderValue'] = df.groupby('CustomerID')['TotalPrice'].mean().values
    rfm['UniqueProducts'] = df.groupby('CustomerID')['StockCode'].nunique().values
    
    print(f"   RFM shape: {rfm.shape}")
    print(rfm[['Recency','Frequency','Monetary']].describe().round(1).to_string())
    return rfm

# ─────────────────────────────────────────────
# 4. SCALING + OPTIMAL K
# ─────────────────────────────────────────────
def find_optimal_k(X_scaled, k_range=range(2, 10)):
    print("\n🔍 Finding optimal K...")
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    
    best_k = min(5, list(k_range)[np.argmax(silhouettes)] + 2)
    print(f"   Best K by silhouette: {best_k}  (score={max(silhouettes):.3f})")
    return list(k_range), inertias, silhouettes, best_k

# ─────────────────────────────────────────────
# 5. CLUSTERING
# ─────────────────────────────────────────────
def cluster_customers(rfm, features, best_k):
    print(f"\n🎯 Clustering into {best_k} segments...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])
    
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm['Cluster'] = km.fit_predict(X_scaled)
    
    # PCA for 2D viz
    pca = PCA(n_components=2)
    rfm[['PC1','PC2']] = pca.fit_transform(X_scaled)
    
    print(f"   Cluster distribution:\n{rfm['Cluster'].value_counts().sort_index().to_string()}")
    return rfm, X_scaled, km

# ─────────────────────────────────────────────
# 6. SEGMENT LABELLING
# ─────────────────────────────────────────────
def label_segments(rfm):
    summary = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
    # Rank: low recency = good, high freq & monetary = good
    summary['Score'] = -summary['Recency'] + summary['Frequency'] + summary['Monetary'] / 100
    ranking = summary['Score'].rank(ascending=False).astype(int)
    
    labels_pool = {
        1: ('🏆 Champions',     '#2ecc71'),
        2: ('💛 Loyal Customers','#f39c12'),
        3: ('🔵 Potential',     '#3498db'),
        4: ('⚠️  At Risk',       '#e74c3c'),
        5: ('💤 Hibernating',   '#95a5a6'),
        6: ('🆕 New Customers', '#9b59b6'),
    }
    
    label_map   = {}
    color_map   = {}
    for cluster, rank in ranking.items():
        lbl, col = labels_pool.get(rank, (f'Segment {rank}', '#bdc3c7'))
        label_map[cluster] = lbl
        color_map[cluster] = col
    
    rfm['Segment']      = rfm['Cluster'].map(label_map)
    rfm['SegmentColor'] = rfm['Cluster'].map(color_map)
    return rfm, label_map, color_map

# ─────────────────────────────────────────────
# 7. FULL VISUALISATION DASHBOARD
# ─────────────────────────────────────────────
def create_dashboard(rfm, k_range, inertias, silhouettes, label_map, color_map, features):
    print("\n🎨 Building visualisation dashboard...")
    
    palette = [color_map[c] for c in sorted(color_map.keys())]
    
    fig = plt.figure(figsize=(22, 18), facecolor='#0f1117')
    fig.suptitle('Customer Segmentation Dashboard', fontsize=26, fontweight='bold',
                 color='white', y=0.98)
    
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    ax_style = dict()
    txt_kw   = dict(color='#c8ccd8', fontsize=10)
    
    def style_ax(ax, title=''):
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#8888aa', labelsize=9)
        [s.set_color('#2d3042') for s in ax.spines.values()]
        if title:
            ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_color('#8888aa')
    
    # ── Panel 1: Elbow ──
    ax1 = fig.add_subplot(gs[0, 0], **ax_style)
    ax1.plot(k_range, inertias, 'o-', color='#3498db', lw=2, markersize=7)
    ax1.fill_between(k_range, inertias, alpha=0.15, color='#3498db')
    style_ax(ax1, 'Elbow Method — Inertia vs K')
    ax1.set_xlabel('Number of Clusters', **txt_kw)
    ax1.set_ylabel('Inertia', **txt_kw)
    
    # ── Panel 2: Silhouette ──
    ax2 = fig.add_subplot(gs[0, 1], **ax_style)
    colors_sil = ['#e74c3c' if s == max(silhouettes) else '#9b59b6' for s in silhouettes]
    ax2.bar(k_range, silhouettes, color=colors_sil, edgecolor='#2d3042', linewidth=0.5)
    style_ax(ax2, 'Silhouette Score vs K')
    ax2.set_xlabel('Number of Clusters', **txt_kw)
    ax2.set_ylabel('Silhouette Score', **txt_kw)
    
    # ── Panel 3: Segment size ──
    ax3 = fig.add_subplot(gs[0, 2], **ax_style)
    seg_counts = rfm['Segment'].value_counts()
    bars = ax3.barh(seg_counts.index, seg_counts.values,
                    color=[color_map[rfm[rfm['Segment']==s]['Cluster'].iloc[0]]
                           for s in seg_counts.index],
                    edgecolor='#2d3042', linewidth=0.5, height=0.65)
    for bar, val in zip(bars, seg_counts.values):
        ax3.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', color='#c8ccd8', fontsize=9)
    style_ax(ax3, 'Customer Count per Segment')
    ax3.set_xlabel('Count', **txt_kw)
    
    # ── Panel 4: PCA scatter ──
    ax4 = fig.add_subplot(gs[1, :2], **ax_style)
    for cluster, label in label_map.items():
        mask = rfm['Cluster'] == cluster
        ax4.scatter(rfm.loc[mask,'PC1'], rfm.loc[mask,'PC2'],
                    c=color_map[cluster], s=18, alpha=0.65, label=label, linewidths=0)
    style_ax(ax4, 'Customer Clusters — PCA 2D Projection')
    ax4.set_xlabel('Principal Component 1', **txt_kw)
    ax4.set_ylabel('Principal Component 2', **txt_kw)
    ax4.legend(framealpha=0.2, facecolor='#1a1d27', edgecolor='#2d3042',
               labelcolor='white', fontsize=9, markerscale=1.8, loc='upper right')
    
    # ── Panel 5: RFM Radar / Spider ──
    ax5 = fig.add_subplot(gs[1, 2], **ax_style, polar=False)
    summary = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean()
    # normalise for heatmap
    norm_summary = (summary - summary.min()) / (summary.max() - summary.min())
    norm_summary['Recency'] = 1 - norm_summary['Recency']   # flip: lower recency = better
    sns.heatmap(norm_summary, ax=ax5, cmap='RdYlGn', annot=True, fmt='.2f',
                linewidths=0.5, linecolor='#0f1117', cbar=False,
                annot_kws={"size": 9, "color": "white"})
    ax5.set_title('RFM Heat-map by Segment\n(normalised, green=good)', color='white',
                  fontsize=11, fontweight='bold', pad=8)
    ax5.tick_params(colors='#c8ccd8', labelsize=9)
    ax5.set_xticklabels(ax5.get_xticklabels(), color='#c8ccd8')
    ax5.set_yticklabels(ax5.get_yticklabels(), color='#c8ccd8', rotation=0)
    
    # ── Panel 6: Revenue per segment ──
    ax6 = fig.add_subplot(gs[2, 0], **ax_style)
    rev = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
    rev_colors = [color_map[rfm[rfm['Segment']==s]['Cluster'].iloc[0]] for s in rev.index]
    wedges, texts, autotexts = ax6.pie(
        rev.values, labels=None, autopct='%1.1f%%',
        colors=rev_colors, pctdistance=0.75,
        wedgeprops=dict(edgecolor='#0f1117', linewidth=1.5))
    for a in autotexts:
        a.set_color('white'); a.set_fontsize(8)
    ax6.set_title('Revenue Share by Segment', color='white', fontsize=11,
                  fontweight='bold', pad=8)
    ax6.legend(rev.index, loc='lower center', bbox_to_anchor=(0.5, -0.18),
               ncol=2, framealpha=0.2, facecolor='#1a1d27', edgecolor='#2d3042',
               labelcolor='white', fontsize=8)
    ax6.set_facecolor('#1a1d27')
    
    # ── Panel 7: Recency vs Monetary ──
    ax7 = fig.add_subplot(gs[2, 1], **ax_style)
    for cluster, label in label_map.items():
        mask = rfm['Cluster'] == cluster
        ax7.scatter(rfm.loc[mask,'Recency'], rfm.loc[mask,'Monetary'],
                    c=color_map[cluster], s=rfm.loc[mask,'Frequency']*4,
                    alpha=0.6, label=label, linewidths=0)
    style_ax(ax7, 'Recency vs Monetary\n(bubble size = Frequency)')
    ax7.set_xlabel('Recency (days)', **txt_kw)
    ax7.set_ylabel('Total Revenue ($)', **txt_kw)
    
    # ── Panel 8: Avg Order Value boxplot ──
    ax8 = fig.add_subplot(gs[2, 2], **ax_style)
    segments_ordered = rfm['Segment'].value_counts().index.tolist()
    data_by_seg = [rfm[rfm['Segment']==s]['AvgOrderValue'].values for s in segments_ordered]
    bp = ax8.boxplot(data_by_seg, patch_artist=True, notch=False,
                     medianprops=dict(color='white', linewidth=2),
                     whiskerprops=dict(color='#5566aa'),
                     capprops=dict(color='#5566aa'),
                     flierprops=dict(marker='.', markerfacecolor='#5566aa', markersize=3))
    for patch, seg in zip(bp['boxes'], segments_ordered):
        c = color_map[rfm[rfm['Segment']==seg]['Cluster'].iloc[0]]
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax8.set_xticklabels([s.split(' ',1)[-1] for s in segments_ordered],
                         rotation=30, ha='right', color='#8888aa', fontsize=8)
    style_ax(ax8, 'Avg Order Value Distribution')
    ax8.set_ylabel('Avg Order Value ($)', **txt_kw)
    
    plt.savefig('/mnt/user-data/outputs/customer_segmentation_dashboard.png',
                dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("   ✅ Dashboard saved.")

# ─────────────────────────────────────────────
# 8. SEGMENT REPORT
# ─────────────────────────────────────────────
def print_report(rfm, label_map, color_map):
    print("\n" + "="*65)
    print("  CUSTOMER SEGMENTATION REPORT")
    print("="*65)
    
    recs = {
        '🏆 Champions':      "Re-engage with loyalty rewards & early-access offers.",
        '💛 Loyal Customers':"Upsell premium products; enrol in referral programme.",
        '🔵 Potential':      "Nurture with personalised emails; targeted cross-sells.",
        '⚠️  At Risk':        "Win-back campaign: discount + personalised message.",
        '💤 Hibernating':    "Low-cost reactivation email; avoid heavy spend.",
        '🆕 New Customers':  "Onboarding sequence; introduce best-sellers.",
    }
    
    summary = rfm.groupby('Segment').agg(
        Customers  = ('CustomerID', 'count'),
        Avg_Recency= ('Recency',    'mean'),
        Avg_Freq   = ('Frequency',  'mean'),
        Avg_Spend  = ('Monetary',   'mean'),
        Total_Rev  = ('Monetary',   'sum'),
    ).round(1)
    summary['Rev_%'] = (summary['Total_Rev'] / summary['Total_Rev'].sum() * 100).round(1)
    
    for seg, row in summary.iterrows():
        print(f"\n  {seg}")
        print(f"    Customers  : {int(row['Customers']):,}")
        print(f"    Avg Recency: {row['Avg_Recency']} days")
        print(f"    Avg Freq   : {row['Avg_Freq']} orders")
        print(f"    Avg Spend  : ${row['Avg_Spend']:,.0f}")
        print(f"    Revenue %  : {row['Rev_%']}%")
        print(f"    ➤ Action   : {recs.get(seg, 'Monitor and analyse.')}")
    
    print("\n" + "="*65)
    return summary

# ─────────────────────────────────────────────
# 9. EXPORT RESULTS
# ─────────────────────────────────────────────
def export_results(rfm):
    out = rfm[['CustomerID','Recency','Frequency','Monetary',
               'AvgOrderValue','UniqueProducts','Cluster','Segment']].copy()
    out.to_csv('/mnt/user-data/outputs/customer_segments.csv', index=False)
    print("\n📁 Segmented customer list → customer_segments.csv")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts']
    
    raw_df   = generate_ecommerce_data(n_customers=1000, n_transactions=15000)
    clean_df = clean_data(raw_df)
    rfm      = compute_rfm(clean_df)
    
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])
    
    k_range, inertias, silhouettes, best_k = find_optimal_k(X_scaled)
    rfm, X_scaled, km = cluster_customers(rfm, features, best_k)
    rfm, label_map, color_map = label_segments(rfm)
    
    summary = print_report(rfm, label_map, color_map)
    create_dashboard(rfm, k_range, inertias, silhouettes, label_map, color_map, features)
    export_results(rfm)
    
    print("\n✅ Project complete!\n")
