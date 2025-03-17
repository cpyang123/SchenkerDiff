HEIGHT = 600

NODE_SIZE = 5
NODE_SIZE_HOVER = 5
NODE_SIZE_SElECT = 5

NODE_COLOR = '#000000'
NODE_COLOR_HOVER = '#3D9970'
NODE_COLOR_SELECT = '#0044DD'

NODE_BORDER_COLOR_UNDECIDED = 'lightgray'
NODE_BORDER_COLOR_TREBLE = 'lightcoral'
NODE_BORDER_COLOR_BASS = 'lightblue'
NODE_BORDER_COLOR_BOTH = '#ccaaee'
NODE_BORDER_COLOR_INNER = 'gray'

DEFAULT_STYLE_SHEET_NODE = {
                'label': 'data(label)',
                'background-color': NODE_COLOR,
                'width': NODE_SIZE,
                'height': NODE_SIZE,
                'font-size': '8px',
                'border-color': NODE_BORDER_COLOR_UNDECIDED,
                "border-opacity": "1",
                "border-width": "2px"
            }

DEFAULT_STYLE_SHEET_EDGE = {
                'width': 0.2,
                'line-color': '#CCCCCC',
                'target-arrow-color': '#CCCCCC',
                'target-arrow-shape': 'triangle',
                'arrow-scale': 0.4,
                'curve-style': 'bezier',
                'control-point-distance': 50,
                'label': 'data(label)',  # Display the edge label
                'font-size': '3x',
                'text-rotation': 'autorotate',
                'text-margin-y': '-2px',
                'color': '#000000',
                'text-background-color': '#ffffff',
                'text-background-opacity': 1,
                'text-background-shape': 'roundrectangle',
                'text-border-color': '#000000',
                'text-border-width': 1,
                'opacity': 0.0
            }

DEFAULT_STYLE_SHEET_EDGE_RLHF = {
                'width': 0.4,
                'line-color': '#CCCCCC',
                # 'target-arrow-color': '#CCCCCC',
                # 'target-arrow-shape': 'triangle',
                # 'arrow-scale': 0.6,
                'curve-style': 'bezier',
                'control-point-distance': 50,
            }

DEFAULT_STYLE_SHEET_NODE_SELECTED = {
                'background-color': NODE_COLOR_SELECT,
                'width': NODE_SIZE_SElECT,
                'height': NODE_SIZE_SElECT
            }

DEFAULT_STYLE_SHEET_NODE_SELECTED_EDGES = {
                'line-color': '#CCCCCC',
                'target-arrow-color': '#CCCCCC',
                'opacity': 1.0
            }

DEFAULT_STYLE_SHEET_CONFIRMED_EDGES = {
                'line-color': '#44ff44',
                'target-arrow-color': '#44ff44',
                'opacity': 1.0
}

DEFAULT_STYLE_SHEET = [
    {
        'selector': 'node',
        'style': DEFAULT_STYLE_SHEET_NODE
    },
    {
        'selector': 'edge',
        'style': DEFAULT_STYLE_SHEET_EDGE
    },
    {
        'selector': 'node:selected',
        'style': DEFAULT_STYLE_SHEET_NODE_SELECTED
    }
]

DEFAULT_STYLE_SHEET_RLHF = [
    {
        'selector': 'node',
        'style': DEFAULT_STYLE_SHEET_NODE
    },
    {
        'selector': 'edge',
        'style': DEFAULT_STYLE_SHEET_EDGE_RLHF
    },
    {
        'selector': 'node:selected',
        'style': DEFAULT_STYLE_SHEET_NODE_SELECTED
    }
]

BUTTON_STYLE = {
    'background-color': '#f9f9f9',
    'border': 'none',
    'padding': '10px 20px',
    'text-align': 'center',
    'text-decoration': 'none',
    'display': 'inline-block',
    'font-size': '15px',
    'margin': '5px',
    'cursor': 'pointer',
    'border-radius': '5px',
    'border': '1px solid #ddd',
    'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
}

RADIO_ITEMS_STYLE = {
    'display': 'flex',
    'flex-direction': 'row',
    'justify-content': 'space-between',
    'align-items': 'center',
    'padding': '10px',
    'background-color': '#f9f9f9',
    'border': '1px solid #ddd',
    'border-radius': '5px',
    'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'margin': '10px 0',
}

RADIO_ITEM_STYLE = {
    'margin': '0 10px',
    'padding': '5px 10px',
    'cursor': 'pointer',
    'font-size': '16px',
    'color': '#333',
    'border-radius': '5px',
    'transition': 'background-color 0.3s ease, transform 0.2s ease',
}

RADIO_ITEM_HOVER_STYLE = {
    'background-color': '#e8f5e9',
    'transform': 'scale(1.05)',
}




