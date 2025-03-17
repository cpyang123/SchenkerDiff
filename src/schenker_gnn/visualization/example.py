import dash
from dash import html
import dash_cytoscape as cyto

app = dash.Dash(__name__)

# Define nodes and edges
nodes = [
    {'data': {'id': 'A', 'label': 'Node A'}, 'position': {'x': 50, 'y': 50}},
    {'data': {'id': 'B', 'label': 'Node B'}, 'position': {'x': 200, 'y': 50}},
    {'data': {'id': 'C', 'label': 'Node C'}, 'position': {'x': 200, 'y': 200}},
    {'data': {'id': 'D', 'label': 'Node D'}, 'position': {'x': 50, 'y': 200}}
]
edges = [
    {'data': {'source': 'A', 'target': 'B'}},
    {'data': {'source': 'B', 'target': 'C'}},
    {'data': {'source': 'C', 'target': 'D'}},
    {'data': {'source': 'D', 'target': 'A'}}
]

# Define Cytoscape graph
app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'preset'},
        style={'width': '100%', 'height': '400px'},
        elements=nodes + edges,
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'background-color': '#0074D9',
                    'width': 20,
                    'height': 20,
                    'grabbable': False,
                    'font-size': '16px'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': 2,
                    'line-color': '#CCCCCC',
                    'target-arrow-color': '#CCCCCC',
                    'target-arrow-shape': 'triangle',
                    'arrow-scale': 2,
                    'curve-style': 'bezier'
                }
            },
            {
                'selector': 'node:selected',
                'style': {
                    'background-color': '#FF4136',
                    'width': 30,
                    'height': 30,
                    'font-size': '16px'
                }
            },
            {
                'selector': 'edge:selected',
                'style': {
                    'width': 4,
                    'line-color': '#FF4136',
                    'target-arrow-color': '#FF4136'
                }
            },
            {
                'selector': 'node:selected ~ edge',
                'style': {
                    'width': 4,
                    'line-color': '#FF4136',
                    'target-arrow-color': '#FF4136'
                }
            }
        ]
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
