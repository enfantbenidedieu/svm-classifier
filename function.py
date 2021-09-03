# load packages
from sklearn import datasets
import numpy as np
from sklearn import metrics
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc


def generate_data(n_samples, dataset, noise):
    if dataset == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)

    elif dataset == "circles":
        return datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=1
        )

    elif dataset == "linear":
        X, y = datasets.make_classification(n_samples=n_samples,n_features=2,n_redundant=0,n_informative=2,
                        random_state=2,n_clusters_per_class=1,
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )

def accuracy(model, xtest, ytest, threshold):
    y_pred_test = (model.decision_function(xtest) > threshold).astype(int)
    test_score = metrics.accuracy_score(y_true=ytest, y_pred=y_pred_test)
    return round(100*test_score,2)

def auc_roc_score(model, xtest, ytest):
    decision_test = model.decision_function(xtest)
    auc_score = metrics.roc_auc_score(y_true=ytest, y_score=decision_test)
    return round(100*auc_score,2)

def average_precision_score(model, xtest, ytest):
    decision_test = model.decision_function(xtest)
    average_score = metrics.average_precision_score(y_true=ytest, y_score=decision_test)
    return round(100*average_score,2)

def f1_score(model, xtest, ytest, threshold):
    y_pred_test = (model.decision_function(xtest) > threshold).astype(int)
    test_score = metrics.f1_score(y_true=ytest, y_pred=y_pred_test)
    return round(100*test_score,2)

def precision_score(model, xtest, ytest, threshold):
    y_pred_test = (model.decision_function(xtest) > threshold).astype(int)
    test_score = metrics.precision_score(y_true=ytest, y_pred=y_pred_test)
    return round(100*test_score,2)

def recall_score(model, xtest, ytest, threshold):
    y_pred_test = (model.decision_function(xtest) > threshold).astype(int)
    test_score = metrics.recall_score(y_true=ytest, y_pred=y_pred_test)
    return round(100*test_score,2)


def cardGraph(id_name):
    return html.Div(
    dbc.Card(
        dbc.CardBody(
            [
                dcc.Graph(
                    id=id_name,
                    figure=dict(layout=dict(plot_bgcolor="rgba(0, 0, 0, 0)", paper_bgcolor="rgba(0, 0, 0, 0)")),
                ),
            ]
            ),
        style={'height':'50vh'},
        color="info", outline=True
    )
)


