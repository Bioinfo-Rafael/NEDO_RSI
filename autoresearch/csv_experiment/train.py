"""Average linear and nonlinear predictions; train on provided arrays only."""
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_model(X_train, y_train):
    linear = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    nonlinear = HistGradientBoostingRegressor(
        max_iter=300, max_leaf_nodes=15, learning_rate=0.05,
        min_samples_leaf=40, l2_regularization=10.0,
        early_stopping=False, random_state=42,
    )
    model = VotingRegressor([("ridge", linear), ("hgb", nonlinear)], weights=[0.25, 0.75])
    model.fit(X_train, y_train)
    return model
