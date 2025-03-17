import pickle
from copy import deepcopy
from random import randint

import dash
import dash_ag_grid as dag
import dash_cytoscape as cyto
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_daq as daq

from inference.inference_utils import sample_analysis, get_node_list
from visualization.callback_utils import *
from visualization_utils import *
from pprint import pprint

app = dash.Dash(__name__)

nodes1 = []
nodes2 = []
edges1 = []
edges2 = []

app.layout = html.Div([
    dcc.Store(id="filename"),
    dcc.Store(id="submit_json"),
    html.Div(id="none"),
    html.Div(children=[
        cyto.Cytoscape(
            id='cytoscape1',
            layout={'name': 'preset'},
            style={
                'width': '50%',
                'height': f'{HEIGHT}px',
                'border': '1px solid black',
                'margin': '2px'
            },
            elements=nodes1 + edges1,
            stylesheet=DEFAULT_STYLE_SHEET_RLHF
        ),
        cyto.Cytoscape(
            id='cytoscape2',
            layout={'name': 'preset'},
            style={
                'width': '50%',
                'height': f'{HEIGHT}px',
                'border': '1px solid black',
                'margin': '2px'
            },
            elements=nodes2 + edges2,
            stylesheet=DEFAULT_STYLE_SHEET_RLHF
        )
    ], style={'display': 'flex', 'flexDirection': 'row'}),
    html.Div(children=[
        daq.ToggleSwitch(
            id='toggle-switch',
            value=False,
            size=100,
            label={
                'label': 'Choose preferred analysis (left or right)',
                'style': {
                    'font-size': '24px'
                }
            }
        )
    ], style={'margin-top': '20px'}),
    html.Div(children=[
        html.Button(
            'Submit',
            id='submit',
            n_clicks=0,
            style={'width': '200px', 'height': '60px', 'font-size': '24px', 'border-radius': '10px'}
        ),
        html.Button(
            'Refresh',
            id='refresh',
            n_clicks=0,
            style={'width': '200px', 'height': '60px', 'font-size': '24px', 'border-radius': '10px'}
        ),
        dcc.Input(
            id="input_filename",
            type="text",
            placeholder="filename here"
        )
    ], style={'display': 'flex', 'justify-content': 'center', 'margin-top': '40px'})
])


@app.callback(
    Output("filename", "data"),
    Input("input_filename", "value")
)
def update_filename(value):
    return value


@app.callback(
    [
        Output('cytoscape1', 'elements'),
        Output('cytoscape2', 'elements'),
        Output('cytoscape1', 'stylesheet'),
        Output('cytoscape2', 'stylesheet'),
        Output('submit_json', 'data')
    ],
    [
        Input('refresh', 'n_clicks'),
        State('filename', 'data'),
        State('submit_json', 'data')
    ]
)
def refresh_graphs(n_clicks, filename, submit_json):
    nodes1, edges1, \
    nodes2, edges2, \
    analysis_treble1, analysis_bass1, node_list1, \
    analysis_treble2, analysis_bass2, node_list2, \
    stylesheet1, stylesheet2 = load_score(filename, node_sample_method="threshold")
    # stylesheet1, stylesheet2 = load_score(filename, node_sample_method="probabilistic")

    if submit_json is None:
        submit_json = [{}]
    for info_name, info in zip([
        'analysis_treble1', 'analysis_bass1',
        'analysis_treble2', 'analysis_bass2',
        'node_list1', 'node_list2'
    ], [
        analysis_treble1, analysis_bass1,
        analysis_treble2, analysis_bass2,
        node_list1, node_list2
    ]
    ):
        submit_json[0][info_name] = info
    # pprint(submit_json)
    return nodes1 + edges1, nodes2 + edges2, \
           stylesheet1, stylesheet2, submit_json


@app.callback(
    Input('submit', 'n_clicks'),
    State('submit_json', 'data'),
    State('filename', 'data'),
    State('toggle-switch', 'value')
)
def submit_preference(n_clicks, submit_json, filename, toggle_value):
    if submit_json is None:
        return
    with open(f'./reward_model_preference_data/{filename}_submission{n_clicks}_randid{randint(0, 999999)}.pkl', 'wb') as f:
        submit_json[0]['preference'] = 2 if toggle_value else 1
        pprint(submit_json)
        pickle.dump(submit_json, f)


if __name__ == "__main__":
    app.run_server(debug=True)
