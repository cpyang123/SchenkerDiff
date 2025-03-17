from visualization.styles import *


def node_selected_instructions(
        selected_node,
        cytoscape_elements,
        stylesheet,
        curr_voice,
        edge_row_data
):
    node_id = selected_node['id']

    # Deselect all nodes and restore default edge color
    deselect_all_nodes(cytoscape_elements, stylesheet)

    # Select the tapped node
    for element in cytoscape_elements:
        if element['data']['id'] == node_id:
            element['data']['selected'] = True

    for edge_row in edge_row_data:
        if edge_row["isCheckable"] and node_id == edge_row["Source"]:
            stylesheet.append({
                'selector': f'edge[source="{node_id}"][target="{edge_row["Target"]}"][id*="{curr_voice.lower()}"]',
                'style': DEFAULT_STYLE_SHEET_NODE_SELECTED_EDGES
            })
        if edge_row["isCheckable"] and node_id == edge_row["Target"]:
            stylesheet.append({
                'selector': f'edge[source="{edge_row["Source"]}"][target="{node_id}"][id*="{curr_voice.lower()}"]',
                'style': DEFAULT_STYLE_SHEET_NODE_SELECTED_EDGES
            })


def background_tap_instructions(selected_nodes, cytoscape_elements, stylesheet):
    if selected_nodes is None:
        deselect_all_nodes(cytoscape_elements, stylesheet)
    if isinstance(selected_nodes, list):
        if len(selected_nodes) == 0:
            deselect_all_nodes(cytoscape_elements, stylesheet)

def deselect_all_nodes(cytoscape_elements, stylesheet):
    for element in cytoscape_elements:
        if 'selected' in element['data']:
            element['data']['selected'] = False
        old_id = element['data']['id']
        stylesheet.append({
            'selector': f'edge[source="{old_id}"]',
            'style': DEFAULT_STYLE_SHEET_EDGE
        })


def apply_edge_width_based_on_prediction_prob(edges, stylesheet):
    for edge in edges:
        stylesheet.append({
            'selector': f'edge[id="{edge["data"]["id"]}"]',
            'style': {
                'width': float(edge["data"]["label"]),
                'arrow-scale': float(edge["data"]["label"]) + 0.1
            }
        })


def node_voice_update_instructions(
        cytoscape_elements, node_row_data, stylesheet
):
    for element in cytoscape_elements:
        color = NODE_BORDER_COLOR_UNDECIDED
        for row in node_row_data:
            if row['Voice'] == 'Treble' and row['Identifier'] == element['data']['id']:
                color = NODE_BORDER_COLOR_TREBLE
            elif row['Voice'] == 'Bass' and row['Identifier'] == element['data']['id']:
                color = NODE_BORDER_COLOR_BASS
            elif row['Voice'] == 'Both' and row['Identifier'] == element['data']['id']:
                color = NODE_BORDER_COLOR_BOTH
            elif row['Voice'] == 'Inner' and row['Identifier'] == element['data']['id']:
                color = NODE_BORDER_COLOR_INNER
        stylesheet.append({
            'selector': f'node[id="{element["data"]["id"]}"]',
            'style': {
                'border-color': color,
                "border-opacity": "1",
                "border-width": "2px"
            }
        })


def is_conflicting_node_and_edge(edge_row, node_row) -> bool:
    different_voice = edge_row['Voice'].lower() != node_row['Voice'].lower()
    ambiguous_voice = node_row['Voice'] in ['Undecided', 'Both']
    voice_conflicts = different_voice and not ambiguous_voice

    includes_source = edge_row['Source'] == node_row['Identifier']
    includes_target = edge_row['Target'] == node_row['Identifier']

    # print(edge_row['Voice'], node_row['Voice'], different_voice and not ambiguous_voice and (includes_source or includes_target))
    return voice_conflicts and (includes_source or includes_target)





