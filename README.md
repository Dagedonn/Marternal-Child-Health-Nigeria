# Maternal & Child Health Disparities Across Nigerian States

**Reducing 8 correlated health indicators to 3 principal components and clustering 37 states (36 + FCT) using PCA + K-Means on 2023–2024 NDHS data.**

## The Finding

Access to care doesn't guarantee good outcomes. One cluster of states has relatively strong antenatal care attendance — yet the **highest childhood fever prevalence** of any group. Getting women into a clinic isn't the same as protecting child health once they're home. That distinction matters for how health budgets get allocated, and it's the kind of pattern you only see by analyzing indicators together instead of one at a time.

## Problem

Nigeria's maternal and child health indicators — ANC visits, skilled birth attendance, facility delivery, contraceptive use, stunting, wasting, underweight, fever — are usually reported one at a time. That misses how they interact, and it gives policymakers no clean way to group states by shared need.

**Question:** Which Nigerian states share similar maternal and child health profiles, and what actually drives the differences between them?

## Data

- **Source:** [Nigeria Demographic and Health Survey (NDHS) 2023–2024](https://dhsprogram.com/), Women's Recode (IR) and Children's Recode (KR) files
- **Raw scale:** 39,050 women × 6,411 variables (IR), 27,783 children × 1,331 variables (KR)
- **Final analysis dataset:** 37 states × 8 indicators, aggregated from individual-level survey responses

| Indicator | Definition |
|---|---|
| ANC 4+ | % women with ≥4 antenatal visits |
| Skilled Birth Attendance | % births assisted by doctor/nurse/midwife |
| Facility Delivery | % births in a recognized health facility |
| Modern FP | % women using modern contraception |
| Stunting | % children with height-for-age < -2 SD |
| Wasting | % children with weight-for-height < -2 SD |
| Underweight | % children with weight-for-age < -2 SD |
| Fever | % children with recent fever |

## Method

1. **Indicator construction** — recoded raw DHS variables into binary flags per woman/child (e.g. `skilled_birth = 1` if doctor OR nurse/midwife assisted)
2. **State-level aggregation** — collapsed individual records to one row per state (proportion-based)
3. **Standardization** — Z-score transform (indicators were on different scales)
4. **PCA** — reduced 8 indicators to 3 components explaining 77.35% of total variance
5. **K-Means (k=3)** — clustered states on the retained PCA scores

## Results

**Variance explained:**

| Component | Variance | Cumulative | Driven by |
|---|---|---|---|
| PC1 | 43.3% | 43.3% | Skilled birth, facility delivery, modern FP (+), stunting (−) — overall maternal/child health development |
| PC2 | 21.3% | 64.6% | Fever, wasting, underweight — child health burden |
| PC3 | 12.8% | 77.4% | ANC 4+ visits — antenatal care utilization specifically |

**Three state clusters:**

| Cluster | States | Profile |
|---|---|---|
| 0 — Advantaged | 16 | Highest skilled birth attendance, facility delivery, FP uptake; lowest stunting |
| 1 — Access ≠ outcome | 8 | Highest ANC attendance, but highest fever prevalence and elevated malnutrition |
| 2 — Disadvantaged | 13 | Lowest maternal healthcare utilization; high child malnutrition |

*Full state-by-cluster membership and cluster mean tables in the report.*

## Recommendation

A single national maternal/child health policy will underserve at least one of these groups. Cluster 2 needs basic access investment (skilled birth attendance, facility delivery infrastructure). Cluster 1 needs something different — disease prevention and nutrition programs layered *on top of* existing ANC access, since access alone isn't moving the fever/malnutrition numbers there.


```

## Tools

Python (pandas, scikit-learn, matplotlib) · NDHS `.dta` survey files · PCA · K-Means


