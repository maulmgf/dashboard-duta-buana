import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys

FILE_PATH = sys.argv[1]
FEATS = ['Recency','Frequency','Monetary','Total_QTY','Avg_Disc']
MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT = 0.02, 0.30, 1.2
CLUSTER_LABEL_MAP = {1:'Produk Champion',2:'Produk Aktif',3:'Produk Potensial',4:'Produk Tidur'}

df_raw = pd.read_excel(FILE_PATH, header=4)
df_raw.columns = ['NO_INVOICE','TANGGAL','NO_CUSTOMER','NAMA_CUSTOMER',
                  'ALAMAT','SALESMAN','ITEM','QTY','UNIT',
                  'HARGA_LIST','DISKON_PCT','DISKON_ITEM','HARGA_JUAL']

df = df_raw.dropna(subset=['NO_INVOICE','ITEM','HARGA_JUAL']).copy()
df = df.drop_duplicates(subset=['NO_INVOICE','ITEM','QTY','HARGA_JUAL'], keep='first')
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
for c in ['QTY','HARGA_JUAL','DISKON_PCT']:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
df = df[(df['HARGA_JUAL'] > 0) & (df['QTY'] > 0)]
df['ITEM'] = df['ITEM'].str.strip().str.upper()
df['NAMA_CUSTOMER'] = df['NAMA_CUSTOMER'].str.strip().str.upper()
df['BULAN'] = df['TANGGAL'].dt.to_period('M')
df['KUARTAL'] = df['TANGGAL'].dt.quarter

SNAPSHOT = df['TANGGAL'].max() + pd.Timedelta(days=1)
rfm = df.groupby('ITEM').agg(
    Recency=('TANGGAL', lambda x: (SNAPSHOT - x.max()).days),
    Frequency=('NO_INVOICE', 'nunique'),
    Monetary=('HARGA_JUAL', 'sum'),
    Total_QTY=('QTY', 'sum'),
    Avg_Disc=('DISKON_PCT', 'mean')
).reset_index()
rfm['Avg_Disc'] = rfm['Avg_Disc'].fillna(0)

qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(100, len(rfm)))
X_scaled = qt.fit_transform(rfm[FEATS])

for col, asc, lbl in [('Recency', True, [5,4,3,2,1]), ('Frequency', False, [1,2,3,4,5]), ('Monetary', False, [1,2,3,4,5])]:
    rfm[f'{col[0]}_Score'] = pd.qcut(rfm[col].rank(method='first', ascending=asc), q=5, labels=lbl).astype(int)
rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
rfm['RFM_Index'] = rfm['R_Score']*0.2 + rfm['F_Score']*0.4 + rfm['M_Score']*0.4

def rfm_segment(s):
    if s >= 13: return 'Champion'
    if s >= 10: return 'High Performer'
    if s >= 7: return 'Growing'
    if s >= 4: return 'At Risk'
    return 'Dormant'
rfm['Segment_RFM'] = rfm['RFM_Score'].apply(rfm_segment)

sil_l = []
K_RANGE = range(2, 10)
for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=50, max_iter=1000)
    lbl = km.fit_predict(X_scaled)
    sil_l.append(silhouette_score(X_scaled, lbl))
K_OPT = max([(k,s) for k,s in zip(K_RANGE, sil_l) if k >= 3], key=lambda x: x[1])[0]

kmeans = KMeans(n_clusters=K_OPT, init='k-means++', random_state=42, n_init=100, max_iter=1000)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

prof = rfm.groupby('Cluster')[FEATS].mean()
prof['score'] = prof['Monetary'].rank() + prof['Frequency'].rank() + prof['Recency'].rank(ascending=False)
ranks = prof['score'].rank(ascending=False).astype(int).to_dict()
rfm['Segment_Cluster'] = rfm['Cluster'].map({c: CLUSTER_LABEL_MAP.get(r, f'Cluster {r}') for c, r in ranks.items()})

profil_klaster = rfm.groupby(['Cluster','Segment_Cluster']).agg(
    Jumlah_Produk=('ITEM','count'), Recency=('Recency','mean'), Frequency=('Frequency','mean'),
    Monetary=('Monetary','mean'), Total_QTY=('Total_QTY','mean'), Avg_Disc=('Avg_Disc','mean')
).reset_index()

basket = df.groupby(['NO_INVOICE','ITEM'])['QTY'].sum().unstack(fill_value=0).astype(bool)
frequent_itemsets = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True, max_len=3)
rules_all = association_rules(frequent_itemsets, metric='lift', min_threshold=MIN_LIFT)
rules = rules_all[rules_all['confidence'] >= MIN_CONFIDENCE].sort_values('lift', ascending=False).reset_index(drop=True)

sil = silhouette_score(X_scaled, rfm['Cluster'])

rapi = lambda s: ', '.join(sorted(s))
rules_export = rules[['antecedents','consequents','support','confidence','lift']].copy()
rules_export['antecedents'] = rules_export['antecedents'].apply(rapi)
rules_export['consequents'] = rules_export['consequents'].apply(rapi)

fi_export = frequent_itemsets.copy()
fi_export['itemsets'] = fi_export['itemsets'].apply(rapi)

df_export = df[['NO_INVOICE','TANGGAL','NO_CUSTOMER','NAMA_CUSTOMER','SALESMAN','ITEM','QTY','HARGA_JUAL','DISKON_PCT','KUARTAL']].copy()
df_export['BULAN'] = df['BULAN'].astype(str)

tren = df.groupby(df['BULAN'].astype(str))['HARGA_JUAL'].sum().reset_index().rename(columns={'BULAN':'Bulan','HARGA_JUAL':'Revenue'})

seg_sum = rfm.groupby('Segment_RFM').agg(
    Jumlah_Produk=('ITEM','count'), Recency=('Recency','mean'), Frequency=('Frequency','mean'), Monetary=('Monetary','mean')
).round(2).reset_index()

kpi = pd.DataFrame({
    'Metric': ['Jumlah Produk','Jumlah Invoice','Jumlah Customer','Revenue Total','K Optimal','Silhouette Score','Total Rules','Max Lift'],
    'Value': [rfm['ITEM'].nunique(), df['NO_INVOICE'].nunique(), df['NO_CUSTOMER'].nunique(), df['HARGA_JUAL'].sum(),
              K_OPT, round(sil,4), len(rules), round(rules['lift'].max(),4)]
})

with pd.ExcelWriter('data/Dashboard_PowerBI.xlsx', engine='openpyxl') as w:
    df_export.to_excel(w, sheet_name='Transaksi_Bersih', index=False)
    rfm.to_excel(w, sheet_name='RFM_Produk', index=False)
    seg_sum.to_excel(w, sheet_name='Segment_Summary', index=False)
    profil_klaster.to_excel(w, sheet_name='Cluster_Profile', index=False)
    rules_export.to_excel(w, sheet_name='Association_Rules', index=False)
    fi_export.to_excel(w, sheet_name='Frequent_Itemsets', index=False)
    tren.to_excel(w, sheet_name='Tren_Bulanan', index=False)
    kpi.to_excel(w, sheet_name='KPI_Dashboard', index=False)

print(f"Selesai. K optimal={K_OPT}, Silhouette={sil:.4f}, Rules={len(rules)}")
