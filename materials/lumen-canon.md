# Lumen Skincare - the running dataset (intro-ML course canon)

Every session in both tracks uses **Lumen Skincare**, the same synthetic DTC skincare brand from
`learn-statistics-with-phoebe` and `learn-experimentation-with-phoebe`. This intro-ML course is the
**middle link** in the Data Science ladder:

> `learn-statistics` (the math) -> **`learn-intro-ml` (the first models)** -> `learn-experimentation` (causal tests)

Statistics taught what the numbers mean (mean vs median, CIs, p-values, r ~ 0.6 spend vs order value).
Intro-ML now asks: **can we PREDICT** conversion and order value from a customer's features, and how do
we know a model is any good? Experimentation then asks whether a change CAUSES a lift. Same brand, same
numbers, so a learner flows straight through. Do NOT invent different figures - reuse these.

## The brand in one line
Lumen Skincare: an $18M/yr direct-to-consumer skincare label, 9 marketing channels, ~2 year history.
We learn ML by building two models on its checkout data.

## The two prediction tasks (the spine of the whole course)

| Task | Target | Type | Taught in | Why it teaches the point |
|---|---|---|---|---|
| **Will this session convert?** | `converted` (bool, ~3.2% true) | **Classification** | leader a4 / builder b3, b7 | Base rate 3.2% => wildly imbalanced => the **accuracy paradox** (a "always predict no" model is 96.8% accurate and useless). Forces precision/recall/ROC-AUC. |
| **How large will the order be?** | `order_value` ($, mean ~$74, right-skewed) | **Regression** | leader a4 / builder b8 | Continuous target, ties back to the stats course's r ~ 0.6. Baseline = "predict the mean". RMSE in dollars a leader understands. |

## The feature set (synthetic, reproducible - builders simulate with `np.random.default_rng(42)`)

Per checkout **session** (grain = one session). Reused/derived from the stats + experimentation data model
(`touchpoints`, `conversions`, `spend_weekly`):

| Feature | Type | Notes / relationship to target |
|---|---|---|
| `prior_30d_spend` | float ($) | **Strongest signal.** Pearson r ~ 0.6 with `order_value` (r^2 ~ 0.36) - the same number the stats course fit in b10. Higher spend => more likely to convert + larger order. |
| `new_vs_returning` | cat (new/returning) | Returning converts ~2x the rate of new. Second-strongest classification feature. |
| `channel` | cat (9: email, paid_search, social, ...) | Email + returning-heavy channels convert best; some paid channels bring browsers who rarely buy. |
| `product_category` | cat (cleanser/serum/moisturizer/set) | Drives `order_value` (sets are the biggest orders). The ANOVA variable in stats b9. |
| `device` | cat (mobile/desktop/tablet) | Desktop converts slightly better; weak signal. |
| `pages_viewed` | int | Engagement proxy; monotone with conversion, saturates. |
| `session_hour` | int 0-23 | Weak/noisy - deliberately included as a **near-useless feature** to teach feature importance + that more columns != better. |

## The authoritative generator (b1 ships this; every builder session reloads its output)
Copy this VERBATIM in b1. Do not change the seed, coefficients, or intercept - the validated results below
depend on them. Later builder sessions start with `df = pd.read_parquet("lumen_sessions.parquet")`.

```python
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N = 40_000                                   # checkout sessions (~10 days at ~4,000/day)

returning = rng.random(N) < 0.35
channel = rng.choice(
    ["email", "paid_search", "social", "organic", "referral",
     "display", "affiliate", "direct", "sms"], N,
    p=[.16, .18, .15, .14, .07, .09, .05, .11, .05])
device = rng.choice(["mobile", "desktop", "tablet"], N, p=[.58, .34, .08])
product_category = rng.choice(
    ["cleanser", "serum", "moisturizer", "set"], N, p=[.34, .28, .26, .12])
pages_viewed = rng.poisson(4, N) + 1
session_hour = rng.integers(0, 24, N)               # deliberately near-useless (noise)
prior_30d_spend = np.clip(rng.gamma(2.2, 34.0, N), 0, 400).round(2)  # right-skewed, mean ~$75

# conversion: a ~3.2% event driven by returning + prior spend + engagement + channel quality
chan_lift = {"email": .8, "direct": .5, "organic": .3, "referral": .2, "sms": .1,
             "paid_search": -.1, "social": -.2, "affiliate": -.3, "display": -.5}
cl = np.array([chan_lift[c] for c in channel])
z = -5.13 + 0.95 * returning + 0.006 * prior_30d_spend + 0.11 * pages_viewed + cl
converted = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)

# order value: right-skewed money, correlated r~0.6 with prior spend, driven by category
cat_base = {"cleanser": 28, "serum": 42, "moisturizer": 36, "set": 76}
cb = np.array([cat_base[c] for c in product_category])
order_value = np.round(
    0.45 * prior_30d_spend + cb + (rng.gamma(1.5, 22, N) - 1.5 * 22), 2).clip(5, 600)

df = pd.DataFrame({
    "channel": channel, "device": device,
    "new_vs_returning": np.where(returning, "returning", "new"),
    "product_category": product_category, "pages_viewed": pages_viewed,
    "session_hour": session_hour, "prior_30d_spend": prior_30d_spend,
    "converted": converted, "order_value": order_value,
})
df.to_parquet("lumen_sessions.parquet")
```
Verified output: 40,000 rows, conversion **3.15%**, AOV mean **$73**, median **$67**, SD **$38**,
`corr(prior_30d_spend, order_value)` = **0.60**. Regression numbers below are fit on the ~1,300 converters.

## Canonical model results (use everywhere - leader, builder, and ml-live must reconcile)

State these as the teaching results. Builders reproduce them with the seed; slight jitter is fine, so
teach "~" values and the RELATIONSHIPS, never a false 4th decimal.

### Classification - predict `converted` (base rate ~3.2%). ALL numbers below verified with a real sklearn fit (seed 42).
- **Dumb baseline** ("always predict no-convert"): **accuracy ~96.8%**, but recall = 0, precision = undefined -> catches zero buyers. *The accuracy paradox in one line.* (a4/b7.)
- **Logistic regression** (the honest first model): **ROC-AUC ~ 0.71** - a real but modest signal, and at a 0.5 threshold the model still predicts "no" for almost everyone (accuracy ~96.8%, no better than the dumb baseline). Lower the threshold and recall rises as precision falls - the precision/recall trade-off, and the reason accuracy alone is the wrong score here. (b3/b7.)
- **Confusion matrix** is the source of truth; accuracy/precision/recall/F1 are all derived from its four cells. (a4/b7.)
- Top features by weight: `prior_30d_spend`, `new_vs_returning`, `channel`. `session_hour` ~ 0 (noise). (b3/b9.)

### Regression - predict `order_value` among converters (mean ~$74, SD ~$38, right-skewed, `r ~ 0.6` with prior spend)
- **Dumb baseline** ("always predict the mean"): **RMSE ~ $41** (= the SD of order value on the converter subset), R^2 = 0. (a4/b8.)
- **Single-feature linear** (`prior_30d_spend` only): **R^2 ~ 0.40, RMSE ~ $32** (r ~ 0.6, so r^2 ~ 0.36 on the full data; ~0.40 on the fitted converter subset). (b8.)
- **Full linear model** (spend + category + channel + returning): **R^2 ~ 0.52, RMSE ~ $28, MAE ~ $20**. Clearly beats the baseline, honest about the ceiling. (b8/b10.)
- MAE (~$20) < RMSE (~$28) always; RMSE punishes the big-order misses harder (right skew). Report both. (b8.)

### Over/underfitting - decision-tree depth (the ml-live.js + b4 centrepiece). Verified `DecisionTreeRegressor` on `order_value ~ prior_30d_spend`:

| max_depth | Train R^2 | Test R^2 | Diagnosis |
|---|---|---|---|
| 1 | 0.31 | 0.22 | **Underfit** (high bias) - too simple, misses the signal. |
| **3** | 0.46 | **0.36** (peak) | **Sweet spot** - the bias-variance minimum. |
| 5 | 0.51 | 0.29 | starting to overfit |
| 8 | 0.62 | 0.15 | **Overfit** |
| 15 | 0.84 | **negative** | badly overfit - memorises training rows, worse than predicting the mean on new data. |

The **gap between train and test** is the overfitting tell (test R^2 turns *negative* by depth 15 while
train keeps climbing). This U-shaped test curve is the single most important picture in the course. The
`ml-live` fit tool defaults to `max_depth=3` and shows the same U. (b4/b5, ml-live `data-tool="fit"`.)

### Cross-validation
- **5-fold CV** on the sweet-spot (depth-4) tree: report **mean R^2 ~ 0.40 +/- 0.06** (folds range ~0.33 to
  ~0.50). CV estimates *out-of-sample* performance and is how you pick `max_depth` without touching the final
  test set - and the spread across folds is itself information (a single test split can be lucky). (b6.)
- CV does NOT improve the model; it gives a less-lucky estimate of how it will generalise. (b6.)

## The ML workflow (the mental model every session reinforces)
`frame the question -> get + split data (train/test) -> train on train only -> evaluate on held-out test
-> tune (via CV, never on test) -> ship + monitor`. Leaders learn to READ this loop; builders RUN it.

## HARD accuracy notes (the whole point of a foundations course - do not get these wrong)
- **Train/test split is sacred.** You fit the model - AND every transformer (scaler, encoder, imputer) -
  on TRAIN only, then apply to TEST. Fitting the scaler on all the data = **data leakage** = a test score
  that lies. This is the #1 beginner mistake; teach `Pipeline` as the fix. (b2/b9.)
- **Accuracy is a trap on imbalanced data.** 96.8% accuracy predicting "no conversion" catches zero buyers.
  Use precision/recall/F1/ROC-AUC + the confusion matrix. (a4/b7.)
- **Overfitting != "high error".** An overfit model has *low training error and high test error*; the GAP is
  the symptom. Underfitting is high error on both. (a2/b4.)
- **Bias-variance:** simple model = high bias (underfit); complex model = high variance (overfit). You trade
  one for the other; the goal is the sum's minimum, not zero of either. (a2/b4/b5.)
- **Cross-validation estimates generalisation; it does not train a better model.** And you tune on CV /
  validation, never on the test set - the test set is spent once, at the end. (b6.)
- **Correlation/importance != causation.** A feature can predict conversion without causing it (returning
  customers convert more, but "being labelled returning" causes nothing). ML predicts; experimentation
  (the next course) is what establishes cause. (a5/b9 -> bridges to experimentation.)
- **More features / more model != better.** `session_hour` adds columns and noise, not signal; a deeper tree
  fits training noise. Parsimony + validation win. (a2/b4/b9.)
- **Garbage in, garbage out + leakage from the target.** A feature computed *after* the outcome (e.g. "was
  refunded") leaks the answer. Features must be knowable at prediction time. (a3/b2.)
- **Regression to the mean vs the model's mean baseline** - don't confuse the statistical phenomenon with
  the "predict the average" baseline; both appear, name them separately. (b8.)
- **Classification threshold is a business choice, not a default.** 0.5 is arbitrary; move it to trade
  precision for recall based on the cost of a false positive vs false negative. (a4/b7.)

## Voice + guardrails
- Two tracks: **leader (a1-a6)** = what ML can/can't do, how to READ a model + its metrics, NO code, exec
  framing; **builder (b1-b10)** = Python **scikit-learn** (+ numpy/pandas), hands-on, everyone starts at b1.
- Hyphens only, never em/en dash. Attribution "by Phoebe Fu". Warm, plain-English, fun-not-dry.
- **Foundational, diff 2 (Core).** Sits directly ABOVE `learn-statistics` (which it assumes) and is the
  prerequisite every later ds course (classification-regression, feature-engineering, model-evaluation,
  ensembles, unsupervised, deep-learning) will assume. Teach intuition first, then the metric, then code.
- Intro scope: teach the *workflow and judgement*, not a zoo of algorithms. Logistic/linear regression +
  decision trees are enough. **Defer to sibling courses:** deep learning, XGBoost/LightGBM ensembles,
  heavy feature engineering, unsupervised clustering depth. Name them as "next" without teaching them.
