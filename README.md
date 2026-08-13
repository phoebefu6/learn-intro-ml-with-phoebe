<!-- phoebe header -->

[![Open the live course](https://img.shields.io/badge/%E2%96%B6%20open%20the%20live%20course-1f6feb?style=for-the-badge)](https://phoebefu6.github.io/learn-intro-ml-with-phoebe/)
[![Star this repo](https://img.shields.io/github/stars/phoebefu6/learn-intro-ml-with-phoebe?style=for-the-badge&label=star%20this%20repo&color=444444)](https://github.com/phoebefu6/learn-intro-ml-with-phoebe/stargazers)
[![Free courses](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fphoebefu6.github.io%2Flearn-with-phoebe%2Fstats.json&query=%24.courses_live&label=free%20courses&style=for-the-badge&color=111111)](https://phoebefu6.github.io/learn-with-phoebe/)

### ▶︎ [Open the live course →](https://phoebefu6.github.io/learn-intro-ml-with-phoebe/)

Free, runs in your browser. No install, no login.

> 📚 Part of **[Learn with Phoebe](https://phoebefu6.github.io/learn-with-phoebe/)** - free, hands-on courses on AI, data, and the craft around them. **[Browse every course ↗](https://phoebefu6.github.io/learn-with-phoebe/)**

<!-- /phoebe header -->

# Learn Intro to Machine Learning with Phoebe

Your first models, and how to know they are any good. A two-track, interactive course - the foundational **Intro to Machine Learning** entry in the Data Science bucket of the Learn with Phoebe hub.

**Live:** https://phoebefu6.github.io/learn-intro-ml-with-phoebe/

## Two tracks, one running brand

- **Leader track (6 x 45 min, no code)** - what machine learning can and cannot do, how a model learns and how it fails, the data behind it, how to read a model's report card without being fooled by a big accuracy number, prediction vs proof, and how to commission and govern ML work responsibly.
- **Builder track (10 x 45 min, Python + scikit-learn)** - the estimator API and a reproducible dataset, the sacred train/test split and the leakage trap, a first classifier and a first regressor, over- and under-fitting, bias-variance and regularization, cross-validation and tuning, classification and regression metrics, pipelines and feature prep, and a model-comparison capstone with a taste of unsupervised learning.

Everything runs on **Lumen Skincare**, a synthetic direct-to-consumer brand carried across three chained Data Science courses:

> [learn-statistics](https://phoebefu6.github.io/learn-statistics-with-phoebe/) (the math) → **learn-intro-ml** (the first models) → [learn-experimentation](https://phoebefu6.github.io/learn-experimentation-with-phoebe/) (causal tests)

We build two models on it: a classifier for whether a checkout session converts (a ~3.2% base rate, so the accuracy paradox is unavoidable) and a regressor for how large the order will be. The conversion rate, the ~$74 average order, and the ~0.60 correlation between prior spend and order value all match the sibling courses, so intuition carries straight across.

## Live ML playground

Three zero-dependency, in-browser tools (`assets/ml-live.js`), seeded from Lumen's numbers and embedded in the sessions that teach them:

- **split** - deal examples into a train pile and a held-out test pile.
- **fit** - slide a decision tree from underfit to overfit and watch the U-shaped test-error curve.
- **threshold** - move a classifier's threshold and watch the accuracy paradox in the confusion matrix.

## Honest about scope

Distilled from the leading foundational material - Andrew Ng / DeepLearning.AI's Machine Learning Specialization, Google's Machine Learning Crash Course, StatQuest, 365 Data Science, and the scikit-learn documentation. Certificates, graded exams, and lecture videos stay with their official sources. Deep learning, ensemble methods in depth, SVMs, heavy feature engineering, and unsupervised depth are pointed to their own sibling courses - this is an applied intro: the workflow and the judgement, not a zoo of algorithms. Every model number in the course is verified with a real scikit-learn fit (see `materials/lumen-canon.md`). Content verified July 2026.

## Structure

```
index.html                     leader + builder landing, paths, playground, knowledge map
courses/a1..a6-*.html          leader track (no code)
courses/b1..b10-*.html         builder track (Python + scikit-learn)
assets/                        style.css · app.js · ml-live.js · mindmap.js
materials/                     lumen-canon.md (the validated running dataset + results) · official-course-map.md
```

by Phoebe Fu
