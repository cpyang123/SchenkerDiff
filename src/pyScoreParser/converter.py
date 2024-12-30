
import json

import music21
import networkx as nx
import matplotlib.pyplot as plt
from musicxml_parser.mxp import MusicXMLDocument
import score_as_graph as score_graph
# Load the MusicXML file
# load xml, create an instant class
XMLDocument = MusicXMLDocument('grace_graph_test.musicxml')
notes = XMLDocument.get_notes() # same to extract_notes: return notes object: https://github.com/jdasam/musicxml_parser/blob/db27849292af3bbb4e153cc097b56593162b5756/mxp/note.py
notes_graph = score_graph.make_edge(notes)

