# Business case: House Price Model Pipeline

**Taking a valuation model from research notebooks to a versioned, tested service.**

## Who buys this

The buyer is the analytics lead at an organisation that values property as part of its work, such as a regional lender's portfolio team, a property-tax assessment office or a real-estate investment firm. Its models live in notebooks owned by one analyst. The users are the analysts and the downstream systems that need a value back.

## The problem and what it costs

A valuation model in a notebook can't be called by other systems, isn't versioned, and usually can't be re-run by anyone except its author. The organisation either re-keys numbers by hand or makes decisions without the model. I couldn't find a reliable public figure for the cost of "notebook-only" models, so this case uses a worked model and claims no industry figure.

## How the repo solves it

| Need | What in the repo does it |
|---|---|
| Keep the research | The original notebooks, unchanged, in `01_research-and-development/` |
| Make it repeatable | A Python package in `src/house_price/` that repeats the notebook's training path and writes a versioned model with its own config |
| Make it callable | FastAPI with `POST /predict`, `GET /metadata` and `GET /health` on port 8000 |
| Make it testable and shippable | A test suite, tox builds, a package published to Gemfury, and CircleCI to Heroku, all in Docker |
| Let a person try it | A Streamlit district form on port 8501 |

The model is trained on the 1990 California census districts: one row per district, predicting the district's median house value ([scikit-learn dataset notes, citing Pace and Barry, 1997](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)).

## Use-case scenarios

All of these organisations are fictional, and each one swaps in its own data.

1. **A county assessor's office** packages its mass-appraisal model the same way. A versioned `POST /predict` lets the appeals team re-run the exact model version used for a given assessment roll.
2. **A regional lender's portfolio team** calls the API nightly to refresh estimated values on its mortgage book, then flags loans whose loan-to-value ratio has moved past a threshold for review.
3. **A property analytics start-up** uses the package-plus-API layout as its template, so each new market model ships with tests and metadata from day one.

## ROI: a worked model (illustrative)

These figures are **illustrative assumptions**.

| Input | Value | Basis |
|---|---|---|
| Analyst hours a month spent re-running notebooks and copying values into other systems | 20 | Assumption |
| Share removed by an API and scheduled scoring | 75% | Assumption |
| Analyst hourly cost | US$70 | BLS May 2024 median for data scientists (about $54 an hour, [BLS](https://www.bls.gov/news.release/archives/ocwage_04022025.pdf)), multiplied by an assumed 1.3 |

Saving: 20 × 12 × 75% = **180 hours, about US$12,600 a year** for one model. The more important benefit is auditability: which model version produced which value.

## Path to production

1. Replace the 1990 district data with current transaction or assessment data at property level, and re-validate.
2. Add monitoring for prediction drift and data freshness.
3. Replace Heroku with the organisation's own hosting and add authentication.
4. Document the model for the relevant valuation and model-risk standards before anyone relies on its output.

## Risks and limits

- **The case is moderate.** The packaging pattern is sound, but the bundled model is a teaching model on 35-year-old district data. It can't value a real property.
- Automated valuations used in lending are regulated. A real deployment needs governance this repo doesn't include.
- The README marks the project as in progress.

## Sources

- scikit-learn, California Housing dataset (Pace and Barry, 1997): https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- U.S. Bureau of Labor Statistics, OEWS May 2024 (Data Scientists): https://www.bls.gov/news.release/archives/ocwage_04022025.pdf
