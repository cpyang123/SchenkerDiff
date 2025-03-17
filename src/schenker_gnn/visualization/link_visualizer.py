import dash
import dash_ag_grid as dag
import dash_cytoscape as cyto
from dash import html, dcc
from dash.dependencies import Input, Output, State

from visualization.callback_utils import *
from visualization_utils import *
from visualization.styles import *

from music21 import converter, environment
import os
import base64

environment.UserSettings()['musescoreDirectPNGPath'] = MUSESCORE_PATH

path = environment.UserSettings()['musescoreDirectPNGPath']
if os.path.exists(path):
    print("MuseScore path is correctly set!")
else:
    print("MuseScore path is invalid.")

app = dash.Dash(__name__)

"""
# Define nodes and edges
music_name = "Primi_1"
xml_filename = f"../schenkerian_clusters/{music_name}/{music_name}.xml"
# xml_filename = "C:/Users/88ste/OneDrive/Desktop/SchenkerGNN - Human Experiment/Mozart 22/Mozart 22.musicxml"
# pkl_filename = f"../schenkerian_clusters/{music_name}/{music_name}.pkl"
nodes, pyscoreparser_notes = load_graph_nodes(xml_filename)
edges, edge_index = load_graph_edges(pyscoreparser_notes)

# Load all models
gnn, lp_treble, lp_bass, voice_model, *_ = load_all_models()

# Make predictions
name, human_readable_treble, human_readable_bass, human_readable_voice = predict_link_probabilities_visualization(
    gnn, lp_treble, lp_bass, voice_model, xml_filename, edge_index
)

# Create UI model
nodes = append_voice_probabilities(nodes, human_readable_voice)
edges = append_edge_probabilities(edges, human_readable_treble, human_readable_bass)
node_columnDefs, node_rowData = create_node_grid_df(nodes)
edge_columnDefs, edge_rowData = create_edge_grid_df(edges)
updated_edge_rows = []
for edge_row in edge_rowData:
    updated_edge_row = edge_row.copy()
    is_checkable = True
    for node_row in node_rowData:
        if is_conflicting_node_and_edge(edge_row, node_row):
            is_checkable = False
            break
    updated_edge_row["isCheckable"] = is_checkable
    updated_edge_rows.append(updated_edge_row)
"""

# Layout
app.layout = html.Div([
    html.Div(id="none"),
    html.Div(children=[
        dcc.Store(id='store-nodes', data=[]),
        dcc.Store(id='store-edges', data=[]),
        # Cytoscape and Upload in a single row with 50% each
        html.Div([
            html.Div([
                cyto.Cytoscape(
                    id='cytoscape',
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': f'{HEIGHT}px'},
                    # elements=nodes + edges,
                    elements=[],
                    stylesheet=DEFAULT_STYLE_SHEET
                )
            ], style={'flex': 1, 'width': '50%', 'padding': '10px'}),  # 50% width
            html.Div([
                dcc.Upload(
                    id='upload-xml',
                    children=html.Div(['Drag and Drop or ', html.A('Select a MusicXML File')]),
                    style={
                        'height': f'{HEIGHT}px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '20px',
                        'display': 'flex',  # Flexbox layout
                        'alignItems': 'center',  # Vertical centering
                        'justifyContent': 'center'  # Horizontal centering
                    },
                    multiple=False
                ),
                html.Div(id='output-xml')
            ], style={'flex': 1, 'width': '50%', 'padding': '10px'})  # 50% width
        ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),

        # RadioItems and Buttons in the next row
        html.Div([
            html.Div([
                dcc.RadioItems(
                    id='voice-radio',
                    value='Bass',
                    options=[{'label': 'Treble', 'value': 'Treble'}, {'label': 'Bass', 'value': 'Bass'}],
                    inline=False,
                    labelStyle=RADIO_ITEM_STYLE,  # Style for individual radio items
                    style=RADIO_ITEMS_STYLE      # Style for the contain
                ),
            ], style={'padding': '10px'}),
            html.Div([
                html.A(html.Button(
                    "Upload New File",
                    id="reset-upload",
                    n_clicks=0,
                    style=BUTTON_STYLE
                ),href='/'),
                html.Button(
                    "Download JSON",
                    id="btn-download-txt",
                    n_clicks=0,
                    style=BUTTON_STYLE
                ),
                dcc.Download(id="download-text"),
                html.Button(
                    'Submit For Model Training',
                    id='submit',
                    n_clicks=0,
                    style=BUTTON_STYLE
                )
            ], style={'padding': '10px'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'})
    ], style={'margin': '0 10px'}),

    # Nodes and Edges Table
    html.Div(children=[
        html.Div(children=[
            html.H1("Nodes"),
            dag.AgGrid(
                id="node-grid",
                # rowData=node_rowData,
                # columnDefs=node_columnDefs,
                rowData=[],
                columnDefs=[],
                columnSize='responsiveSizeToFit'
            ),
        ], style={'padding': '5px', 'flex': 1}),
        html.Div(children=[
            html.H1("Edges"),
            dag.AgGrid(
                id="edge-grid",
                # rowData=updated_edge_rows,
                # columnDefs=edge_columnDefs,
                rowData=[],
                columnDefs=[],
                columnSize='responsiveSizeToFit',
                dashGridOptions={"rowSelection": "multiple", "suppressRowClickSelection": True}
            )
        ], style={'padding': '5px', 'flex': 1})
    ], style={'display': 'flex', 'flexDirection': 'row'})
])


@app.callback(
    [
        Output('cytoscape', 'stylesheet'),
        Output('cytoscape', 'elements'),
    ],
    [
        Input('store-nodes', 'data'),
        Input('store-edges', 'data'),
        Input('cytoscape', 'selectedNodeData'),
        Input('cytoscape', 'tapNodeData'),
        Input('voice-radio', 'value'),
        Input('node-grid', 'cellValueChanged'),
        Input('edge-grid', 'selectedRows'),
        State('cytoscape', 'elements'),
        State('node-grid', 'rowData'),
        State('edge-grid', 'rowData')
    ]
)
def update_stylesheet(
        store_nodes,
        store_edges,
        selected_nodes,
        tapped_node,
        curr_voice,
        voice_update,
        selected_edges,
        cytoscape_elements,
        node_row_data,
        edge_row_data
):
    stylesheet = DEFAULT_STYLE_SHEET.copy()

    # if moused_over_node:
    #     stylesheet.append({
    #         'selector': f'node[id = "{moused_over_node["id"]}"]',
    #         'style': {
    #             'background-color': NODE_COLOR_HOVER,
    #             'width': NODE_SIZE_HOVER,
    #             'height': NODE_SIZE_HOVER
    #         }
    #     })

    # if moused_over_edge:
    #     stylesheet.append({
    #         'selector': f'edge[source = "{moused_over_edge["source"]}"][target = "{moused_over_edge["target"]}"]',
    #         'style': {
    #             'width': 0.4,
    #             'line-color': '#CC4444',
    #             'target-arrow-color': '#CC4444',
    #             'target-arrow-shape': 'triangle',
    #             'arrow-scale': 0.6,
    #             'curve-style': 'bezier'
    #         }
    #     })
    if not cytoscape_elements:
        cytoscape_elements = store_nodes + store_edges

    if tapped_node:
        node_selected_instructions(
            tapped_node,
            cytoscape_elements,
            stylesheet,
            curr_voice,
            edge_row_data
        )

    background_tap_instructions(selected_nodes, cytoscape_elements, stylesheet)

    apply_edge_width_based_on_prediction_prob(store_edges, stylesheet)

    if voice_update:
        node_voice_update_instructions(cytoscape_elements, node_row_data, stylesheet)

    if selected_edges:
        for edge_row in selected_edges:
            stylesheet.append({
                'selector': f'edge[source="{edge_row["Source"]}"][target="{edge_row["Target"]}"][id*="{edge_row["Voice"]}"]',
                'style': DEFAULT_STYLE_SHEET_CONFIRMED_EDGES
            })

    return stylesheet, cytoscape_elements



"""
@app.callback(
    [
        Output('edge-grid', 'rowData')
    ],
    [
        Input('node-grid', 'cellValueChanged'),
        State('node-grid', 'rowData'),
        State('edge-grid', 'rowData')
    ]
)
def update_possible_edges(voice_update, node_rows, edge_rows):
    updated_edge_rows = []
    for edge_row in edge_rows:
        updated_edge_row = edge_row.copy()
        is_checkable = True
        for node_row in node_rows:
            if is_conflicting_node_and_edge(edge_row, node_row):
                is_checkable = False
                break
        updated_edge_row["isCheckable"] = is_checkable
        updated_edge_rows.append(updated_edge_row)
    return [updated_edge_rows]
"""
"""
@app.callback(
    [
        Output('none', 'style')
    ], [
        Input('submit', 'n_clicks'),
        State('node-grid', 'rowData'),
        State('edge-grid', 'rowData')
    ]
)
def fine_tune_example(n_clicks, node_row_data, edge_row_data):
    print(node_row_data)
    print(edge_row_data)

    return [{"margin": "0px"}]
"""
@app.callback(
    [
        Output('node-grid', 'rowData'),
        Output('node-grid', 'columnDefs'),
        Output('edge-grid', 'rowData'),
        Output('edge-grid', 'columnDefs'),
    ],
    [
        Input('store-nodes', 'data'),
        Input('store-edges', 'data')
    ]
)
def update_ui_from_store(nodes, edges):
    if not nodes or not edges:
        return [], [], [], []

    node_columnDefs, node_rowData = create_node_grid_df(nodes)
    edge_columnDefs, edge_rowData = create_edge_grid_df(edges)
    updated_edge_rows = []

    for edge_row in edge_rowData:
        updated_edge_row = edge_row.copy()
        is_checkable = True
        for node_row in node_rowData:
            if is_conflicting_node_and_edge(edge_row, node_row):
                is_checkable = False
                break
        updated_edge_row["isCheckable"] = is_checkable
        updated_edge_rows.append(updated_edge_row)

    return node_rowData, node_columnDefs, updated_edge_rows, edge_columnDefs

@app.callback(
    [
        Output('upload-xml', 'style'),
        Output('output-xml', 'children'),
        Output('upload-xml', 'contents'),
    ],
    [
        Input('upload-xml', 'contents'),
    ],
    [
        State('upload-xml', 'filename')
    ]
)
def display_musicxml(contents, filename):
    if contents is None:
        return {
                   'height': f'{HEIGHT}px',
                   'lineHeight': '60px',
                   'borderWidth': '1px',
                   'borderStyle': 'dashed',
                   'borderRadius': '5px',
                   'textAlign': 'center',
                   'margin': '20px',
                   'display': 'flex',
                   'alignItems': 'center',
                   'justifyContent': 'center'
               }, "Please upload a MusicXML file.", None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    temp_dir = "../temp"
    os.makedirs(temp_dir, exist_ok=True)
    xml_filepath = os.path.join(temp_dir, filename)

    with open(xml_filepath, 'wb') as f:
        f.write(decoded)

    score = converter.parse(decoded)
    for thing in score:
        print(thing)
    svg_data = score.write('musicxml.png')
    with open(svg_data, "rb") as svg_file:
        svg_base64 = base64.b64encode(svg_file.read()).decode()
    img_src = f"data:image/png;base64,{svg_base64}"

    return (
        {'display': 'none'},
        html.Div([
            html.H4(f"Uploaded file: {filename}"),
            html.Img(src=img_src, style={'width': '90%', 'margin': '20px'}),
        ]),
        contents,
    )


@app.callback(
    [
        Output('store-nodes', 'data'),
        Output('store-edges', 'data'),
    ],
    [
        Input('submit', 'n_clicks'),
    ],
    [
        State('upload-xml', 'contents'),
        State('upload-xml', 'filename'),
    ]
)
def on_submit_click(
        n_clicks,
        file_contents,
        file_name
):
    if n_clicks > 0:
        if file_contents is not None:
            content_type, content_string = file_contents.split(',')
            xml_filepath = f"../temp/{file_name}"
            os.makedirs("../temp", exist_ok=True)
            with open(xml_filepath, 'wb') as f:
                f.write(base64.b64decode(content_string))

            # try:
            nodes, pyscoreparser_notes = load_graph_nodes(xml_filepath)
            edges, edge_index = load_graph_edges(pyscoreparser_notes)

            # Load all models
            gnn, lp_treble, lp_bass, voice_model, *_ = load_all_models()

            # Make predictions
            name, human_readable_treble, human_readable_bass, human_readable_voice = predict_link_probabilities_visualization(
                gnn, lp_treble, lp_bass, voice_model, xml_filepath, edge_index
            )

            # Create UI model
            nodes = append_voice_probabilities(nodes, human_readable_voice)
            edges = append_edge_probabilities(edges, human_readable_treble, human_readable_bass)
            return nodes, edges
            # except Exception as e:
            #     print(f"Error processing file {file_name}: {str(e)}")
            #     return [], []

    return [], []

if __name__ == '__main__':
    app.run_server(debug=True)