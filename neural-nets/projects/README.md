# E-Commerce Delivery Prediction

Predicts delivery times (regression) and late delivery probabilities (classification) using a e-commerce public dtset. I've coded logistic regression from scratch w/numpy and implemented a NN w/pytorch

## Key Features
- **Data Pipeline:** merging, imputation and fture engineering w/ pandas and scikit-learn
- **Deep Learning:** a pytorch-based multi-layer perceptron (MLP) for delivery time regression.

## Quickstart
1. place the dataset csvs in `./data/olist/`.
2. don't forget the dependencies: `pandas numpy scikit-learn torch`
3. run the model pipeline:
```bash
uv run python delivery-model.py
```
## Notes
- First project, code might s*c*k for you if you are experienced.
- I tested on a Macbook M4 PRO.
