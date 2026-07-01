# Algorithm and Package Playbook for Velvet Python

This repository is intentionally small, but the decision process is serious.
The point is not to make every model look like “machine learning” —
it is to choose the right abstraction for the data, timeline, and team.

We group models by what they optimize:

- **Comprehension speed**: fast to explain to non-technical audiences
- **Confidence**: reliable signals for interview or production discussions
- **Failure control**: how easy it is to detect and explain bad behavior
- **Migration path**: how naturally the approach can move to stronger tooling

## The practical ladder (what to build first)

### 1) Rule-based baseline
Use this when:

- you need a fast sanity check,
- the domain has clear phrases you can encode, and
- you are validating your full loop before model risk.

What it teaches:

- deterministic outcomes
- explicit edge-case handling
- where automation should *not* start yet

### 2) Frequency models (WordFrequency)
Use this when:

- labels are clear,
- data are small to medium,
- and you need a stable baseline in a reviewable form.

What it teaches:

- feature construction with minimal dependencies,
- class-conditional behavior,
- interpretable scoring as a safety step before complexity.

### 3) TF-IDF class centroids
Use this when:

- vocabulary drift is present,
- common tokens are noisy,
- and you need better separation than raw counts.

What it teaches:

- rare-term weighting,
- sparse vectors,
- when geometry appears in text tasks.

### 4) Naive Bayes / Cosine similarity
Use these when:

- interview rigor matters,
- probability or distance interpretation is useful,
- you are moving from educational baseline to reusable pipeline behavior.

What they teach:

- uncertainty-aware outputs,
- geometric perspective on text behavior,
- tradeoffs around assumptions and calibration.

## Recommended packages as your system grows

| Need | Start Here | Mature To |
|---|---|---|
| Tiny proof-of-concept | standard library (`src.ai`, `src.ml_pipeline`) | scikit-learn (`LinearSVC`, `LogisticRegression`) |
| Deterministic classroom demo | local modules + CLI | `scikit-learn` pipelines |
| Moderate text corpus | `Text Classification` module + TF-IDF | `spaCy`, `scikit-learn` feature transforms |
| Higher recall / low false positives | `NaiveBayesClassifier` baseline | calibrated classifiers + class-weight tuning |
| Team-scale experimentation | custom modules + manifests | feature store + vector stores + tracking |

## Collision patterns to check before production

1. **Data leakage in split logic**
   - if your split is random but unseeded, you cannot reproduce failures.
   - if rows repeat, treat duplicates as part of your manifest contract.

2. **Imbalance masquerading as quality**
   - if one class dominates, accuracy can look good while minority recall collapses.
   - read confusion matrix off-diagonal patterns before celebrating any gain.

3. **Latency assumptions that break on growth**
   - a model that is instant on 200 rows can become unstable at 20k rows.
   - capture runtime notes in your experiment ledger while inputs scale.

4. **Package gravity**
   - adding advanced packages without test updates tends to hide behavior.
   - require one regression test per external dependency you introduce.

## Choosing based on problem shape

- **Few rows, high ambiguity**: keep to `word_frequency`, add clear rules first.
- **More rows, more classes**: move to `naive_bayes` and evaluate recall.
- **Need semantic explainability**: introduce cosine and discuss limits of nearest-neighbor style boundaries.
- **Need migration confidence**: mirror every module with an equivalent scikit-learn baseline and compare.

## How this repo proves the choices

The playbook is encoded in:

- `scripts/run_experiments.py` for reproducible runs,
- `scripts/dataset_audit.py` for quality evidence,
- `tests/` for fail-fast behavior,
- docs updates after every change so future readers can continue the chain.

You should treat this as a **model contract**:

- if a choice is documented, it is intentional,
- if it is not documented, it is not production-ready,
- if it cannot be demonstrated, it is not complete.

## Suggested next step

Before switching libraries, run:

1. one full sweep (`--epochs 3`),
2. one CSV summary artifact (`--summary-csv`),
3. one short note in `docs/experiment-ledger.md` explaining what changed and why.

That sequence gives you more evidence than guessing, and more confidence than hype.
