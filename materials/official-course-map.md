# Official course map - learn-intro-ml-with-phoebe

What real, official material each session maps to, and how much of it we teach. Research date 2026-07-21
(re-verify vendor changelogs before any live delivery). **80% bar:** each session teaches ~80% of its
mapped sources' *working* content; certificates, graded assessments, and lecture videos stay official -
we say so honestly. Fetched syllabi appended verbatim at the bottom.

Coverage key: ✓ = taught in depth · ◐ = introduced / taste · — = deferred by design (named, not taught).

## The true intro-ML core (appears in almost every syllabus) and where we teach it

| Core topic | Leader | Builder | Sources |
|---|---|---|---|
| What ML is; supervised vs unsupervised vs reinforcement | a1 ✓ | b1 ◐ | Ng, Google, CS229, 365 |
| Framing / when NOT to use ML | a1 ✓ | — | Ng, Google |
| Features, labels, numerical vs categorical data | a3 ✓ | b1, b9 ✓ | Google, sklearn, 365 |
| Feature scaling / preprocessing / engineering | a3 ◐ | b9 ✓ | Ng, sklearn, 365 |
| Linear regression + cost | a4 ◐ | b8 ✓ | all 5 |
| Gradient descent + learning rate | a2 ◐ (intuition) | b8 ◐ | Ng, StatQuest, CS229 |
| Logistic regression / classification | a4 ✓ | b3 ✓ | Ng, Google, StatQuest, 365, CS229 |
| Train/test split + generalisation | a2 ✓ | b2 ✓ | sklearn, Google, 365 |
| Overfitting vs underfitting | a2 ✓ | b4 ✓ | Ng, Google, sklearn, 365 |
| Bias-variance tradeoff | a2 ✓ | b4, b5 ✓ | StatQuest, CS229 |
| Regularization (ridge/lasso) | a5 ◐ | b5 ✓ | Ng, StatQuest, CS229 |
| Cross-validation (k-fold) | a2 ◐ | b6 ✓ | sklearn, StatQuest, CS229 |
| Classification metrics (confusion matrix, P/R/F1, ROC-AUC) | a4 ✓ | b7 ✓ | sklearn, StatQuest |
| Regression metrics (MAE/MSE/RMSE/R²) | a4 ✓ | b8 ✓ | sklearn, StatQuest, 365 |
| Decision trees | a5 ◐ | b4, b10 ✓ | Ng, sklearn, StatQuest, CS229 |
| Pipelines / leakage prevention | a3 ◐ | b9 ✓ | sklearn |
| Hyperparameter tuning / search | — | b6 ✓ | sklearn, CS229 |
| Clustering / k-means (unsupervised taste) | a5 ◐ | b10 ◐ | Ng, sklearn, 365, CS229 |
| ML fairness / ethics / responsible AI | a6 ✓ | — | Google, Ng |
| ML lifecycle / monitoring / drift | a6 ✓ | b10 ◐ | Google (Real-world ML) |

## Per-session coverage

### Leader track (a1-a6) - read + interpret, no code
- **a1 What ML actually is (and isn't)** - ML vs rules/BI, the three learning types, prediction vs causation, when not to use ML, the ML workflow loop. Ng C1 W1 ✓, Google intro ✓.
- **a2 How a model learns, and how it goes wrong** - learning from examples, the train/test idea, overfitting/underfitting, bias-variance intuition, the U-curve. Ng C1 W3 (overfitting) ✓, StatQuest bias-variance ✓. Live: `ml-live fit`.
- **a3 The data is the model** - features/labels, garbage-in-garbage-out, data leakage (a feature known only after the outcome), why more columns != better, prep at a glance. sklearn common-pitfalls ✓.
- **a4 Reading a model's report card** - classification vs regression outputs, the accuracy paradox, confusion matrix, precision/recall/F1/ROC-AUC in plain words, RMSE/R² in dollars, threshold as a business choice. sklearn metrics ✓, StatQuest ✓. Live: `ml-live threshold`.
- **a5 Prediction is not proof (+ the model zoo at a glance)** - linear/logistic regression, decision trees, one paragraph each on ensembles / neural nets / clustering (what + when), importance != causation, bridge to the experimentation course.
- **a6 Putting ML to work responsibly** - the lifecycle (ship, monitor, drift), fairness / bias / ethics, buy vs build, how to commission and govern an ML project, the questions a leader should ask the data team. Google ML fairness ✓, Ng responsible AI ✓.

### Builder track (b1-b10) - Python scikit-learn
- **b1 Setup + your first dataset** - sklearn/pandas/numpy, simulate the Lumen dataset (seed 42), explore, the `fit`/`predict` estimator API. sklearn getting-started ✓.
- **b2 The train/test split and the leakage trap** - `train_test_split`, `stratify`, fit-on-train-only, the scaler-on-test disaster (MSE 0.90 vs 62.80). sklearn common-pitfalls ✓.
- **b3 Your first model, end to end** - `LogisticRegression` predicting `converted`, `fit`/`predict`/`predict_proba`, first accuracy read - and the accuracy paradox appears. Ng C1 W3 ✓.
- **b4 Over- and under-fitting** - `DecisionTreeRegressor`/`Classifier` at rising `max_depth`, train vs test score, the U-curve, bias-variance made concrete. sklearn ✓. Live: `ml-live fit`.
- **b5 Bias-variance + regularization** - the tradeoff, `Ridge`/`Lasso`, how a penalty dials complexity down. Ng ✓, StatQuest ✓.
- **b6 Cross-validation + tuning** - `cross_val_score` (5-fold), mean ± std, `GridSearchCV` / validation curve to pick `max_depth` without touching the test set. sklearn ✓.
- **b7 Classification metrics in depth** - `confusion_matrix`, `precision_recall_fscore_support`, `classification_report`, `roc_auc_score`, `precision_recall_curve`, moving the threshold. sklearn ✓, StatQuest ✓.
- **b8 Regression metrics + the regression model** - `LinearRegression` on `order_value`, MAE/MSE/RMSE/R², baseline = predict the mean, single (R² ~0.36) vs multi-feature (R² ~0.45). sklearn ✓, 365 ✓.
- **b9 Feature prep + Pipelines** - `ColumnTransformer` (`OneHotEncoder` + `StandardScaler`), `Pipeline` to kill leakage, `coef_` / `feature_importances_`, `session_hour` exposed as noise. sklearn pipelines ✓.
- **b10 Capstone: compare models + a taste of unsupervised** - compare logistic vs tree (name random forest), pick with CV, score once on held-out test, write a model card, then a short `KMeans` customer-segmentation taste that tees up the next courses. sklearn ✓, 365 clustering ◐.

## Not covered by design (named as "next", taught in sibling courses)
- **Neural networks + deep learning** -> `learn-deep-learning-with-phoebe` (Ng C2, Google "Advanced").
- **Ensembles in depth (random forests, gradient boosting, stacking)** -> `learn-ensemble-methods-with-phoebe` (a single tree is intro; the zoo is not).
- **Support vector machines + kernels** -> advanced (CS229-level).
- **Heavy feature engineering** -> `learn-feature-engineering-with-phoebe`.
- **Model evaluation in depth (calibration, cost curves, drift monitoring)** -> `learn-model-evaluation-with-phoebe`.
- **Unsupervised depth (clustering, PCA, anomaly detection)** -> `learn-unsupervised-with-phoebe` (we give only a k-means taste).
- **Reinforcement learning, recommenders, gradient-descent-from-scratch math** -> out of an applied intro's scope; named only.
- Certificates, graded quizzes, and lecture videos stay with the official providers - this course teaches the working content, not the credential.

## From your subscriptions - the finish-list (take these, take notes, pass them back)
Ranked, most-relevant first. See `references/learner-platforms.md` in the skill for the platform list.
1. **DeepLearning.AI / Coursera - "Machine Learning Specialization" Course 1: Supervised ML: Regression &amp; Classification** (Andrew Ng). The spine of this whole course. https://www.coursera.org/learn/machine-learning
2. **Google - Machine Learning Crash Course** (free, fast). Linear/logistic regression, generalization &amp; overfitting, fairness. https://developers.google.com/machine-learning/crash-course
3. **StatQuest - Machine Learning Fundamentals** (Josh Starmer). Cross-validation, confusion matrix, bias/variance, ROC/AUC - the clearest intuition on the planet. https://statquest.org/video_index.html
4. **scikit-learn docs - Getting Started + Common Pitfalls** (not a course, but read cover to cover). https://scikit-learn.org/stable/getting_started.html + /common_pitfalls.html
5. **365 Data Science - Machine Learning in Python**. Train/test, over/underfitting, logistic regression, a first clustering pass. https://365datascience.com/courses/machine-learning-in-python/

---

## Appendix - fetched syllabi (verbatim, 2026-07-21)

### Andrew Ng / DeepLearning.AI "Machine Learning Specialization" (Coursera)
URL: https://www.coursera.org/specializations/machine-learning-introduction and .../learn/machine-learning
- **C1 Supervised ML: Regression &amp; Classification** - W1: what is ML; supervised; unsupervised; linear regression model; cost function; gradient descent + learning rate; GD for linear regression. W2: multiple features; vectorization; GD for multiple linear regression; feature scaling; convergence; feature engineering; polynomial regression. W3: logistic regression; decision boundary; logistic cost; GD; the problem of overfitting; addressing overfitting; regularization (linear + logistic).
- **C2 Advanced Learning Algorithms** - neural networks, decision tree learning, random forests, model evaluation, transfer learning, data ethics/responsible AI.
- **C3 Unsupervised, Recommenders, RL** - clustering, anomaly detection, dimensionality reduction, recommenders, reinforcement learning.
(Original 2011 Stanford MOOC retired; rigorous version = CS229.)

### Google "Machine Learning Crash Course"
URL: https://developers.google.com/machine-learning/crash-course
ML Models: Linear Regression; Logistic Regression; Classification. Data: Numerical Data; Categorical Data; Datasets, Generalization, and Overfitting. Advanced ML Models: Neural Networks; Embeddings; Intro to LLMs. Real-world ML: Production ML Systems; AutoML; ML Fairness.

### scikit-learn official docs
URLs: /stable/user_guide.html · /getting_started.html · /common_pitfalls.html
User Guide: Supervised learning; Unsupervised learning; Model selection and evaluation; Inspection; Visualizations; Dataset transformations; Common pitfalls and recommended practices; Choosing the right estimator. Getting Started: fit/predict; transformers &amp; pre-processors; pipelines; model evaluation; automatic parameter searches (`train_test_split`, `StandardScaler`, `make_pipeline`, `cross_validate` default 5-fold, `RandomizedSearchCV`). Common Pitfalls: 12.1 Inconsistent Preprocessing; 12.2 Data Leakage; 12.3 Controlling Randomness. Load-bearing rule: "never call `fit` on the test data"; "always split into train and test first, particularly before any preprocessing." Worked example: correct test transform MSE 0.90 vs forgotten transform 62.80; `SelectKBest`-on-all-data fake 0.76 acc vs realistic 0.5.

### StatQuest (Josh Starmer)
URL: https://statquest.org/video_index.html
ML Fundamentals: Cross Validation; Confusion Matrix; Sensitivity &amp; Specificity; Bias &amp; Variance; ROC and AUC. Regression: R-squared; linear regression / least squares; multiple regression; logistic regression (coefficients, MLE). Regularization: Ridge (L2); Lasso (L1); Ridge vs Lasso; gradient descent. Trees: decision/classification trees; regression trees; cost-complexity pruning.

### 365 Data Science "Machine Learning in Python"
URL: https://365datascience.com/courses/machine-learning-in-python/
Linear regression (OLS, R², adjusted R², F-test, OLS assumptions, dummies); linear regression with sklearn (feature scaling/selection, underfitting &amp; overfitting, train/test); practical example (multicollinearity); logistic regression (odds, accuracy, overfitting, testing); cluster analysis basics (classification vs clustering); k-means (choosing k, standardize?, market segmentation); other clustering (dendrogram, heatmaps); exam.

### Stanford CS229 (the rigorous version - for reference only)
URL: https://cs229.stanford.edu/syllabus-autumn2018.html
Intro; supervised setup + linear regression; weighted least squares, logistic regression, Newton, perceptron, GLMs; GDA, naive Bayes; SVMs + kernels; **bias-variance, regularization, model/feature selection**; tree ensembles; neural nets (basics + training); practical ML advice; k-means, GMM/EM; factor analysis; PCA/ICA; MDPs + RL.

### LinkedIn Learning
"Machine Learning Foundations" there = the prerequisite MATH series (linear algebra / calculus / probability / statistics), not an applied intro. Not used as a topic template.
