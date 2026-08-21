"""
Maternal & Child Health Disparities Across Nigerian States
============================================================
Pipeline: raw NDHS survey files -> state-level indicators -> PCA -> K-Means clustering

Data: Nigeria Demographic and Health Survey (NDHS) 2023-2024
      Women's Recode (IR) and Children's Recode (KR) files
      Source: https://dhsprogram.com/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# 1. LOAD RAW DHS FILES
# ---------------------------------------------------------------------------
# IR = Individual (Women's) Recode: 39,050 women x 6,411 variables
# KR = Children's Recode: 27,783 children x 1,331 variables
women = pd.read_stata("NGIR8BFL.DTA")
children = pd.read_stata("NGKR8BFL.DTA")

# ---------------------------------------------------------------------------
# 2. CONSTRUCT MATERNAL HEALTH INDICATORS (from women's recode)
# ---------------------------------------------------------------------------
# ANC 4+ : at least 4 antenatal visits during most recent pregnancy
women["m14_clean"] = women["m14_1"].replace(98, np.nan)  # 98 = don't know
women["anc_4plus"] = (women["m14_clean"] >= 4).astype(int)

# Skilled birth attendance: doctor OR nurse/midwife present at delivery
women["skilled_birth"] = (
    (women["m3a_1"] == 1) | (women["m3b_1"] == 1)
).astype(int)

# Modern family planning: currently using a modern contraceptive method
women["modern_fp"] = (women["v313"] == 3).astype(int)

# Facility delivery: birth occurred in a recognized public/private/NGO health facility
facility_codes = [21, 22, 23, 24, 26, 27, 31, 32, 33, 34, 36]
women["facility_delivery"] = women["m15_1"].isin(facility_codes).astype(int)

# ---------------------------------------------------------------------------
# 3. CONSTRUCT CHILD HEALTH INDICATORS (from children's recode)
# ---------------------------------------------------------------------------
# WHO cutoff: z-score < -200 (i.e. < -2.00 SD) indicates malnutrition
children["stunted"] = (children["hw70"] < -200).astype(int)      # height-for-age
children["wasted"] = (children["hw72"] < -200).astype(int)       # weight-for-height
children["underweight"] = (children["hw71"] < -200).astype(int)  # weight-for-age
children["fever"] = (children["h22"] == 1).astype(int)           # fever in reference period

# ---------------------------------------------------------------------------
# 4. AGGREGATE TO STATE LEVEL
# ---------------------------------------------------------------------------
state_maternal = women.groupby("sstate1").agg({
    "anc_4plus": "mean",
    "skilled_birth": "mean",
    "modern_fp": "mean",
    "facility_delivery": "mean",
})

state_child = children.groupby("sstate1").agg({
    "stunted": "mean",
    "wasted": "mean",
    "underweight": "mean",
    "fever": "mean",
})

final_data = state_maternal.merge(state_child, left_index=True, right_index=True)
# final_data: 37 rows (36 states + FCT) x 8 indicators

# ---------------------------------------------------------------------------
# 5. STANDARDIZE (Z-score transform)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(final_data)

# ---------------------------------------------------------------------------
# 6. PRINCIPAL COMPONENT ANALYSIS
# ---------------------------------------------------------------------------
pca = PCA()
pca_scores = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_

loadings = pd.DataFrame(
    pca.components_.T,
    index=final_data.columns,
    columns=[f"PC{i+1}" for i in range(len(final_data.columns))],
)
print("Variance explained by each component:")
print(explained_variance)
print("\nComponent loadings:")
print(loadings)

# Scree plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Proportion of Variance Explained")
plt.title("Scree Plot")
plt.grid(True)
plt.savefig("figures/scree_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 7. K-MEANS CLUSTERING (on first 3 retained PCA components -> 77.35% variance)
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_scores[:, :3])
final_data["cluster"] = clusters

state_clusters = final_data.reset_index()
cluster_summary = state_clusters.groupby("cluster")["sstate1"].apply(list)
print("\nStates by cluster:")
print(cluster_summary)

cluster_profile = final_data.groupby("cluster").mean(numeric_only=True)
print("\nMean indicator values by cluster:")
print(cluster_profile)

# ---------------------------------------------------------------------------
# 8. CLUSTER VISUALIZATION
# ---------------------------------------------------------------------------
pca_df = pd.DataFrame(pca_scores[:, :2], columns=["PC1", "PC2"])

plt.figure(figsize=(8, 6))
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=clusters, cmap="viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("State Clusters Based on Maternal and Child Health Indicators")
plt.savefig("figures/cluster_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
