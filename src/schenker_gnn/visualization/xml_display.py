import base64
import os

import dash
from dash import dcc, html, Input, Output, State
from music21 import converter, environment

# Install MuseScore
environment.UserSettings()['musescoreDirectPNGPath'] = r'/Applications/MuseScore 4.app/Contents/MacOS/mscore'


path = environment.UserSettings()['musescoreDirectPNGPath']
if os.path.exists(path):
    print("MuseScore path is correctly set!")
else:
    print("MuseScore path is invalid.")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("MusicXML Viewer"),
    dcc.Upload(
        id='upload-xml',
        children=html.Div(['Drag and Drop or ', html.A('Select a MusicXML File')]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '20px'
        },
        multiple=False
    ),
    html.Div(id='output-xml')
])


@app.callback(
    Output('output-xml', 'children'),
    Input('upload-xml', 'contents'),
    State('upload-xml', 'filename')
)
def display_musicxml(contents, filename):
    if contents is None:
        return "Please upload a MusicXML file."

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    score = converter.parse(decoded)
    svg_data = score.write('musicxml.png')
    print(svg_data)
    with open(svg_data, "rb") as svg_file:
        svg_base64 = base64.b64encode(svg_file.read()).decode()
    img_src = f"data:image/png;base64,{svg_base64}"

    return html.Div([
        html.H4(f"Uploaded file: {filename}"),
        html.Img(src=img_src,style={'width': '80%','margin': '20px'}),
    ])

if __name__ == '__main__':
    app.run_server(debug=True)