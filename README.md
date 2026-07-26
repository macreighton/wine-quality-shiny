# 🍷 Wine Quality Explorer — R Shiny Case Study

An interactive R Shiny application that predicts red wine quality from
physicochemical lab measurements, built as a walk-through of the full
**data-science life cycle**: problem definition → data collection → EDA →
modeling → evaluation → deployment.

**Live app:** https://macreighton.shinyapps.io/wine-quality-explorer/

## 1. Dataset

[UCI Machine Learning Repository — Wine Quality (Red Wine)](https://archive.ics.uci.edu/dataset/186/wine+quality)
(Cortez, Cerdeira, Almeida, Matos & Reis, 2009). 1,599 samples, 11
physicochemical input features (fixed/volatile acidity, citric acid,
residual sugar, chlorides, free/total sulfur dioxide, density, pH,
sulphates, alcohol) and one target, `quality` (median expert sensory score,
0–10). The full CSV (~84 KB) is included at `data/winequality-red.csv`.

## 2. Motivation

Wine quality is normally assessed by human tasting panels — slow,
expensive, and subjective. This app asks: can cheap, objective lab
chemistry predict the score a panel would give? A working model gives
producers fast, repeatable quality feedback and highlights which chemical
properties matter most.

## 3. Algorithm

**Random Forest** (regression and classification) - an ensemble of
bagged decision trees with random feature subsampling at each split,
compared against a **Linear Regression** baseline. See the "About &
Methodology" tab inside the app for the full mathematical derivation
(split criteria, Gini impurity, out-of-bag error, permutation importance).

## 4. The app

Four tabs:

| Tab | What it does |
|---|---|
| **About & Methodology** | Data-science life cycle write-up + math behind Random Forest |
| **Explore Data** | Histograms, summary stats, correlation heatmap, raw data table |
| **Train Model** | Choose Regression/Classification, tune `ntree`, `mtry`, train/test split; see RMSE/R² or accuracy/confusion matrix, variable importance, and OOB-error curves update live |
| **Predict** | Move sliders for the 11 chemical properties and get a live prediction from the currently trained model |

## 5. Run locally

```r
install.packages(c("shiny", "shinythemes", "randomForest",
                    "ggplot2", "reshape2", "DT"))
shiny::runApp()
```

Or run straight from GitHub:

```r
shiny::runGitHub("wine-quality-shiny", "macreighton")
```

## 6. Deploy to shinyapps.io

```r
install.packages("rsconnect")
library(rsconnect)
rsconnect::setAccountInfo(name   = "<ACCOUNT>",
                           token  = "<TOKEN>",
                           secret = "<SECRET>")
rsconnect::deployApp(appDir = ".", appName = "wine-quality-explorer")
```

(Account name/token/secret come from your shinyapps.io dashboard →
Account → Tokens. See `deploy.R` in this repo for a ready-to-edit script.)

## 7. Findings (see full report `Wine_Quality_Case_Study_Report.docx`)

- Random Forest (RMSE 0.535, R² 0.496) outperforms Linear Regression
  (RMSE 0.607, R² 0.404) on a held-out test set — the quality relationship
  is at least partly non-linear.
- **Alcohol** and **volatile acidity** are the two strongest predictors of
  quality, matching known oenology research.
- Quality scores are concentrated at 5–7 on the 0–10 scale; very few wines
  are rated at the extremes, an important caveat for the classification view.

## 8. Repository structure

```
.
├── app.R                                 # Shiny application
├── data/winequality-red.csv              # UCI wine quality dataset
├── deploy.R                              # shinyapps.io deployment script
├── Wine_Quality_Case_Study_Report.docx   # Written report (data/motivation/methodology/findings)
├── figures/                              # Report figures
└── README.md
```

## Citation

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine
preferences by data mining from physicochemical properties.* Decision
Support Systems, Elsevier, 47(4):547-553, 2009.
