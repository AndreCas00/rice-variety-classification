# Rice Variety Classification and Clustering

An academic statistical learning project in R, developed as part of the Master's in Data Science for Economics at the University of Milan.

## Objective

The project investigates whether morphological measurements can distinguish Cammeo and Osmancik rice grains. Supervised classification models are compared, followed by principal component analysis (PCA) and k-means clustering to explore the underlying structure of the data.

## Dataset

Source: [UCI Machine Learning Repository — Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik).

- 3,810 observations: 1,630 Cammeo and 2,180 Osmancik grains.
- Seven numerical predictors: Area, Perimeter, Major_Axis_Length, Minor_Axis_Length, Eccentricity, Convex_Area and Extent.
- Binary target: rice variety (`Class`).

## Methodology

1. **Exploratory data analysis:** descriptive statistics, class distribution, missing-value and duplicate checks, correlation matrix, boxplots and density plots.
2. **Supervised learning:** linear probability model (OLS), logistic regression, stepwise logistic regression (AIC), ridge and lasso logistic regression, decision tree, random forest and k-nearest neighbours (KNN).
3. **Model evaluation:** a 50/50 training-test split (seed 123), accuracy and confusion matrices. Standardization for distance-based and regularized models uses training-set statistics.
4. **Unsupervised learning:** PCA on standardized predictors and k-means clustering with two clusters, using both the original standardized predictors and the first two principal components. Class labels are used only to assess cluster agreement in this exploratory analysis.

## Results

### Supervised learning

Reported results on the 1,905-observation test set:

| Model | Accuracy | Misclassified observations |
|---|---:|---:|
| Logistic regression | 93.3% | 128 |
| KNN | 93.2% | 129 |
| Lasso logistic regression | 93.2% | 130 |
| Ridge logistic regression | 93.1% | 131 |
| OLS | 92.9% | 135 |
| Decision tree | 92.6% | 141 |
| Random forest | 92.5% | 143 |

Logistic regression achieves the highest reported accuracy. More complex models do not substantially improve performance in this split. Grain-size measurements, particularly area, perimeter, major axis length and convex area, are important for distinguishing the varieties. Stepwise logistic regression reduces the number of predictors from seven to four and lowers AIC from 716.11 to 710.44.

### Unsupervised learning

- The first two principal components explain 86.8% of the total variance; the first three explain 99.7%.
- Both k-means configurations achieve 91.3% agreement with the known varieties (3,478 of 3,810 observations, using the majority class within each cluster).
- The results suggest a clear but imperfect separation between the two varieties.

## Reproducing the analysis

1. Download the dataset from the UCI link above. Convert the provided spreadsheet to CSV if necessary, preserving the original column names, and save it as `Rice_Cammeo_Osmancik.csv`.
2. Place the CSV and the R script in the same folder. Open the script in RStudio and set the working directory to that folder (Session > Set Working Directory > To Source File Location).
3. Install the required packages if needed:

```r
install.packages(c("tidyverse", "corrplot", "sandwich", "lmtest",
                   "glmnet", "rpart", "rpart.plot", "randomForest", "class"))
```

4. Run the R script from beginning to end using **Source** or execute the sections individually. The script generates the statistical summaries, model comparisons and plots.

## Limitations

- Results are based on a single random training-test split and are not an external validation.
- The original KNN implementation selects k using the test set (best k = 17), so its reported accuracy is potentially optimistic. A separate validation set or cross-validation within the training set would provide a fairer evaluation.
- PCA and k-means are exploratory analyses of the full dataset. The number of clusters is fixed at two to assess agreement with the two known varieties; cluster agreement is not out-of-sample classification accuracy.

## Tools

R, RStudio, ggplot2 (tidyverse), corrplot, sandwich, lmtest, glmnet, rpart, rpart.plot, randomForest and class.

## Author

Andreas Casini — Statistical Learning academic project, September 2026.
