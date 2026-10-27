# AI Projects Lab — the app

A free static web app that turns the 35 machine-learning projects from
[ai-projects-portfolio](https://github.com/ikram6566/ai-projects-portfolio) into
**working tools anyone can use in the browser** — no server, no sign-up, no cost.

**Live:** deployed on Vercel (free tier).

## How it works
- `export_models.py` trains a lightweight browser version of each project
  (TF-IDF + Logistic Regression for text, scaled Logistic/Linear Regression for
  tabular data, a small 784→64→10 neural net for MNIST); `export_batch4.py` adds
  the Batch-4 models (TF-IDF softmax for news topics, precomputed SVD
  recommendations, a stock-return regressor, and a CIFAR-10 CNN whose weights
  run as a JavaScript forward pass). Each model is evaluated on
  a held-out test set, and exports the weights to `models.json`.
  The accuracy shown in the app is each browser model's *own* measured score.
- `index.html` loads `models.json` and runs every prediction **100% client-side**
  in vanilla JavaScript — TF-IDF scoring, logistic/softmax regression and the
  MNIST forward pass are re-implemented in JS.
- Text models export their top features by coefficient weight (5,000 of 30,000
  for sentiment/fake-news, 3,000 of 5,000 for spam), which keeps `models.json`
  small while staying faithful to the notebook models.

## Add a new project
1. Build + verify the notebook in the portfolio repo.
2. Add a section to `export_models.py`, re-run it (`python export_models.py`).
3. Add a card + panel + runner in `index.html`.
4. Redeploy — the site picks up the new project automatically.

## Files
- `index.html` — the whole app (UI + JS model engines)
- `models.json` — exported model weights + measured accuracies (generated)
- `export_models.py` — training/export pipeline
