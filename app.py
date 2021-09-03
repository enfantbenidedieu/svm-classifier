# load packages
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
import time
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import dash_reusable_components as drc
import figures as figs
import function as fcts

source = 'https://miro.medium.com/max/1040/0*m355u3-pHvd5DsLA.png'

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Support Vector Machine'
server =  app.server


def navitem_button(btn_name,id_name,link, color):
    return dbc.NavItem(
                dbc.Button(
                    btn_name,
                    id = id_name,
                    outline=True,
                    href=link,
                    style={"textTransform": "none"},
                    color=color
                )
            )


def performance_div(id_name,performance_value,performance_name):
    return html.Div(
            [html.P(performance_name),html.H6(id=performance_value, style={'textAlign':'center'})],
            id=id_name,
            className="mini_container")



def other_graph(id_graph):
    return html.Div([
        dcc.Graph(
                id = id_graph,
                figure=dict(layout=dict(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)'))
            )
    ],className="mini_container")

button_sklearn = navitem_button('Sklearn','sklearn','https://scikit-learn.org/stable/', "primary")
button_svm = navitem_button('SVM','svm','https://scikit-learn.org/stable/modules/svm.html',"info")
button_dash = navitem_button('Dash','dash','https://dash.plotly.com/',"primary")
button_plotly = navitem_button('Plotly','plotly','https://plotly.com/python/',"info")


# Header
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            id="logo",
                            src=source,
                            height="60px",
                        ),
                        md="auto",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3("Support Vector Machine (SVM) Explorer"),
                                    html.P("Binary Classifier"),
                                ],
                                id="app-title",
                            )
                        ],
                        md=True,
                        align="center",
                    ),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.NavbarToggler(id="navbar-toggler"),
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        button_sklearn,button_svm, button_dash,button_plotly
                                    ],
                                    navbar=True,
                                ),
                                id="navbar-collapse",
                                navbar=True,
                            )
                        ],
                        md=2,
                    ),
                ],
                align="center",
            ),
        ],
        fluid=True,
    ),
    dark=True,
    color="dark",
    sticky="top",
)



parameters_one = html.Div([
        drc.NamedDropdown(
                    name="Select Dataset",
                    id="dropdown-select-dataset",
                    options=[{"label": "Moons", "value": "moons"},
                            {"label": "Linearly Separable","value": "linear"},
                            {"label": "Circles","value": "circles"}],
                    clearable=False,
                    searchable=False,
                    value="moons",
                ),
                drc.NamedSlider(
                    name="Sample Size",
                    id="slider-dataset-sample-size",
                    min=100,
                    max=500,
                    step=100,
                    marks={str(i): str(i) for i in [100, 200, 300, 400, 500]},
                    value=300,
                ),
                drc.NamedSlider(
                    name="Noise Level",
                    id="slider-dataset-noise-level",
                    min=0,
                    max=1,
                    marks={i / 10: str(i / 10) for i in range(0, 11, 2)},
                    step=0.1,
                    value=0.2,
                ),
                html.Hr(),
                drc.NamedSlider(
                    name="Threshold",
                    id="slider-threshold",
                    min=0,
                    max=1,
                    value=0.5,
                    step=0.01,
                ),
                html.Button("Reset Threshold",id="button-zero-threshold"),
                html.Hr(),
],
className="pretty_container four columns",
id="cross-filter-options"
,style={'height':'100%'})



parameters_two = html.Div([
                drc.NamedDropdown(
                    name="Kernel",
                    id="dropdown-svm-parameter-kernel",
                    options=[{"label": "Radial basis function (RBF)","value": "rbf"},
                            {"label": "Linear", "value": "linear"},
                            {"label": "Polynomial","value": "poly"},
                            {"label": "Sigmoid","value": "sigmoid"},
                            ],
                    value="rbf",
                    clearable=False,
                    searchable=False,
                ),
                drc.NamedSlider(
                    name="Cost (C)",
                    id="slider-svm-parameter-C-power",
                    min=-2,
                    max=4,
                    value=0,
                    marks={i: "{}".format(10 ** i) for i in range(-2, 5)},
                ),
                drc.FormattedSlider(
                    id="slider-svm-parameter-C-coef",
                    min=1,
                    max=9,
                    value=1,
                ),
                drc.NamedSlider(
                    name="Degree",
                    id="slider-svm-parameter-degree",
                    min=2,
                    max=10,
                    value=3,
                    step=1,
                    marks={str(i): str(i) for i in range(2, 11, 2)},
                ),
                drc.NamedSlider(
                    name="Gamma",
                    id="slider-svm-parameter-gamma-power",
                    min=-5,
                    max=0,
                    value=-1,
                    marks={i: "{}".format(10 ** i) for i in range(-5, 1)},
                ),
                drc.FormattedSlider(
                    id="slider-svm-parameter-gamma-coef",
                    min=1,
                    max=9,
                    value=5,
                ),
],style={'height':'100%'})


perf_div = html.Div(
        [
            dbc.Row([
                dbc.Col(performance_div('acc','accuracy','Accuracy score'),sm=2),
                dbc.Col(performance_div('fscore','f1-score','f1 score'),sm=2),
                dbc.Col(performance_div('aucscore','auc-score','Area under a curve'),sm=2),
                dbc.Col(performance_div('precisionscore','precision-score','Precision score'),sm=2),
                dbc.Col(performance_div('averageprecisionscore','average-precision-score','Aver. prec. score'),sm=2),
                dbc.Col(performance_div('recallscore','recall-score','Recall score'),sm=2),
            ])  
        ],
        id= 'info-container',
        className="row container-display",
    )


infos_div = html.Div([
    html.Div(
        [
            html.P("Shrinking :", style = {'font-weight':'bold','font-family':'Times New Roman'}),
            dcc.RadioItems(
                id="radio-svm-parameter-shrinking",
                labelStyle={"margin-right": "7px","display": "inline-block"},
                options=[{"label": " Enabled","value": "True"},
                        {"label": " Disabled","value": "False"}],
                value="True",
            ),
            dcc.Graph(
                id = 'graph-sklearn-svm',
                figure=dict(layout=dict(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)'))
            )
        ]
    )
],style={'height':'100%'})


body = html.Div([
    dbc.Row([
                dbc.Col(performance_div('acc','accuracy','Accuracy score'),sm=2),
                dbc.Col(performance_div('fscore','f1-score','f1 score'),sm=2),
                dbc.Col(performance_div('aucscore','auc-score','Area under a curve'),sm=2),
                dbc.Col(performance_div('precisionscore','precision-score','Precision score'),sm=2),
                dbc.Col(performance_div('averageprecisionscore','average-precision-score','Aver. prec. score'),sm=2),
                dbc.Col(performance_div('recallscore','recall-score','Recall score'),sm=2),
            ]),
    html.Br(),
    dbc.Row([
        dbc.Col(parameters_one,sm=3),
        dbc.Col(infos_div,sm=6),
        dbc.Col(parameters_two,sm=3)
    ],align='center'),
    html.Br(),
    dbc.Row([
        dbc.Col(other_graph('graph-roc-curve'),sm=4),
        dbc.Col(other_graph('graph-confusion-pie'),sm=4),
        dbc.Col(other_graph('graph-lift-curve'),sm=4)
    ],align='center')
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"})

app.layout = html.Div([
    header,
    body
])



"""App Callback"""
@app.callback(
    Output("slider-svm-parameter-gamma-coef", "marks"),
    [Input("slider-svm-parameter-gamma-power", "value")],
)
def update_slider_svm_parameter_gamma_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-svm-parameter-C-coef", "marks"),
    [Input("slider-svm-parameter-C-power", "value")],
)
def update_slider_svm_parameter_C_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
    [State("graph-sklearn-svm", "figure")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks:
        Z = np.array(figure["data"][0]["z"])
        value = -Z.min() / (Z.max() - Z.min())
    else:
        value = 0.4959986285375595
    return value


# Disable Sliders if kernel not in the given list
@app.callback(
    Output("slider-svm-parameter-degree", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_degree(kernel):
    return kernel != "poly"


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_coef(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("slider-svm-parameter-gamma-power", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_power(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    [
        Output("graph-sklearn-svm", "figure"),
        Output("graph-roc-curve", "figure"),
        Output("graph-confusion-pie", "figure"),
        Output("graph-lift-curve", "figure"),
        Output('accuracy', 'children'),
        Output('f1-score', 'children'),
        Output('auc-score', 'children'),
        Output('average-precision-score', 'children'),
        Output('precision-score', 'children'),
        Output('recall-score', 'children')],
    [
        Input("dropdown-svm-parameter-kernel", "value"),
        Input("slider-svm-parameter-degree", "value"),
        Input("slider-svm-parameter-C-coef", "value"),
        Input("slider-svm-parameter-C-power", "value"),
        Input("slider-svm-parameter-gamma-coef", "value"),
        Input("slider-svm-parameter-gamma-power", "value"),
        Input("dropdown-select-dataset", "value"),
        Input("slider-dataset-noise-level", "value"),
        Input("radio-svm-parameter-shrinking", "value"),
        Input("slider-threshold", "value"),
        Input("slider-dataset-sample-size", "value"),
    ],
)
def update_svm_graph(
    kernel,
    degree,
    C_coef,
    C_power,
    gamma_coef,
    gamma_power,
    dataset,
    noise,
    shrinking,
    threshold,
    sample_size,
):
    t_start = time.time()
    h = 0.3  # step size in the mesh

    # Data Pre-processing
    X, y = fcts.generate_data(n_samples=sample_size, dataset=dataset, noise=noise)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    C = C_coef * 10 ** C_power
    gamma = gamma_coef * 10 ** gamma_power

    if shrinking == "True":
        flag = True
    else:
        flag = False

    # Train SVM
    clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=flag,probability=True)
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    prediction_figure = figs.serve_prediction_plot(
        model=clf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Z=Z,
        xx=xx,
        yy=yy,
        mesh_step=h,
        threshold=threshold,
    )

    roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test, y_test=y_test)
    confusion_figure = figs.serve_pie_confusion_matrix(
        model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
    )
    lift_figure = figs.serve_lift_curve(model=clf, xtest=X_test, ytest=y_test)

    #
    acc = fcts.accuracy(model = clf, xtest = X_test, ytest = y_test, threshold=threshold)
    f1score = fcts.f1_score(model=clf, xtest=X_test, ytest=y_test,threshold=threshold)
    precisionscore = fcts.precision_score(model=clf, xtest=X_test, ytest=y_test,threshold=threshold)
    recallscore = fcts.recall_score(model=clf, xtest=X_test, ytest=y_test,threshold=threshold)
    average_score = fcts.average_precision_score(model=clf, xtest=X_test, ytest=y_test)
    auc = fcts.auc_roc_score(model=clf, xtest=X_test, ytest=y_test)

    return prediction_figure,roc_figure,confusion_figure,lift_figure,f"{acc}%", f'{f1score}%', f"{auc}%", f'{average_score}%',f'{precisionscore}%',f'{recallscore}%'

if __name__ == '__main__':
    app.run_server(debug=True)