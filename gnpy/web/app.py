'''
gnpy.web.app
============

This module contains functionality for running a web server offering a
web-based GUI in a browser. The GUI displays the network topology specified
in a topology file uploaded by the user. Other currently supported features
are requesting path computation between specified source and destination nodes
as well as include node constraints. The resulting path together with
calculated OSNR and GSNR is reported back to the user.
'''

from uuid import uuid4
from pathlib import Path
import tempfile
import json
from base64 import b64decode
import oyaml
from flask import Flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from numpy import mean, linspace
from gnpy.core.exceptions import (EquipmentConfigError, NetworkTopologyError,
                                  ConfigurationError)
from gnpy.core.equipment import (load_equipment, equipment_from_json,
                                 automatic_nch, trx_mode_params)
from gnpy.core.network import network_from_json, network_to_json, build_network
from gnpy.core.utils import db2lin, lin2db, load_json
from gnpy.core.elements import (Transceiver, Fiber, RamanFiber,
                                Fused, Edfa, Roadm)
from gnpy.core.request import Path_request, compute_constrained_path, propagate
from gnpy.core.convert import convert_file
from gnpy.core.parameters import SimParams
from gnpy.core.science_utils import Simulation

# The default equipment file is loaded at startup:
equipment_global = load_equipment('../../examples/eqpt_config.json')
SI_global = equipment_global['SI']['default']

# Some global variables used by several callback functions:
color_seq = px.colors.qualitative.Prism
axis_format = dict(showgrid=False, zeroline=False, showticklabels=False)
margin_format = dict(l=25, r=25, t=25, b=25)

# This app is currently using an external custom CSS stylesheet from dash.
# Longer-term we should consider defining our own style:
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask(__name__)

dash_app = dash.Dash(__name__, server=server,
                     external_stylesheets=external_stylesheets)

# This is just to be able to run this as a flask app in the cloud:
app = dash_app.server

dash_app.layout = html.Div([
    html.H2('GNPy Optical Network Simulator',
            style={'margin-left': '1%', 'margin-top': '1%'}),

    html.Div(
        dcc.Upload(
            id='equipment-file',
            children=html.Div([
                html.A('Load equipment file'),
                ' (or leave empty for default)']),
            style={
                'width': '100%',
                'height': '50px',
                'lineHeight': '50px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }),
        style={'width': '25%', 'display': 'inline-block'}),

    html.Div(
        dcc.Upload(
            id='network-file',
            children=html.Div(
                html.A('Load network file')
            ),
            style={
                'width': '100%',
                'height': '50px',
                'lineHeight': '50px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }),
        style={'width': '25%', 'display': 'inline-block'}),

    html.Div([
        html.P('Auto-design:',
               style={'margin-left': '3%', 'display': 'inline-block'}),

        html.Div(
            dcc.RadioItems(
                id='auto-design',
                options=[
                    {'label': 'Power mode', 'value': 'ap'},
                    {'label': 'Gain mode', 'value': 'ag'},
                    {'label': 'None', 'value': 'na'}
                ],
                value='ap',
                labelStyle={'display': 'inline-block'}),
            style={'display': 'inline-block'})],
        style={'width': '25%', 'display': 'inline-block'}),

    html.Div(
        dcc.Loading(
            html.Div(
                html.P(id='loading-result'),
                id='loading-result-div')),
        style={'width': '25%', 'textAlign': 'center',
               'display': 'inline-block'}),

    html.Div(
        dcc.Graph(
            figure=go.Figure(
                layout=go.Layout(margin=margin_format,
                                 xaxis=axis_format, yaxis=axis_format)),
            style={'height': 1000},
            id='topology'),
        id='topology-graph',
        style={'width': '75%', 'display': 'inline-block'}),

    html.Div([
        html.Button('Run simulation', id='run-simulation'),

        html.Div(
            dcc.Upload(
                id='sim-params-file',
                children=html.Div(
                    html.A('Load simulation parameters')
                ),
                style={
                    'width': '100%',
                    'height': '36px',
                    'lineHeight': '36px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                }),
            style={'width': '70%', 'display': 'inline-block'}),

        html.Div([
            dcc.Loading(html.Pre(id='simulation-result',
                                 style={'text-align': 'left',
                                        'font-size': '140%'})),

            html.Div(
                html.Div(
                    dcc.Graph(id='gsnr-vs-power'),
                    hidden=True),
                id='gsnr-vs-power-div')],
            style={'height': '27vh'}),

        html.H6('Select source/destination (or click in graph):'),

        html.Div(
            dcc.Dropdown(
                id='source',
                placeholder='source',
                disabled=True),
            id='source-dd',
            style={'width': '50%', 'display': 'inline-block'}),

        html.Div(
            dcc.Dropdown(
                id='destination',
                placeholder='destination',
                disabled=True),
            id='destination-dd',
            style={'width': '50%', 'display': 'inline-block'}),

        html.H6('Select nodes to include (or click in graph):'),

        html.Div(
            dcc.Dropdown(
                id='include-nodes',
                disabled=True),
            id='include-nodes-dd',
            style={'height': '6vh', 'width': '100%',
                   'display': 'inline-block'}),

        html.H6('Power sweep (only start for single simulation):'),

        dcc.Input(
            id='power-start',
            type='number',
            inputMode='numeric',
            placeholder=('start (default: {:.1f} dBm)'
                         .format(SI_global.power_dbm)),
            style={'width': '33%', 'display': 'inline-block'}),

        dcc.Input(
            id='power-stop',
            type='number',
            inputMode='numeric',
            placeholder='stop'.format(SI_global.power_dbm),
            style={'width': '33%', 'display': 'inline-block'}),

        dcc.Input(
            id='power-step',
            type='number',
            inputMode='numeric', min=0,
            placeholder='step'.format(SI_global.power_dbm),
            style={'width': '33%', 'display': 'inline-block'}),

        html.H6('Other request parameters:'),

        html.P('Baudrate [Gbaud]:',
               style={'width': '50%', 'display': 'inline-block'}),

        dcc.Input(
            id='baudrate',
            type='number', min=1,
            inputMode='numeric',
            placeholder=('default: {:.2f} Gbaud'
                         .format(SI_global.baud_rate / 1e9)),
            style={'width': '50%'}),

        html.P('Roll-off:',
               style={'width': '50%', 'display': 'inline-block'}),

        dcc.Input(
            id='roll-off',
            type='number', min=0, max=0,
            inputMode='numeric',
            placeholder='default: {}'.format(SI_global.roll_off),
            style={'width': '50%'}),

        html.P('Tx OSNR [dB]:',
               style={'width': '50%', 'display': 'inline-block'}),

        dcc.Input(
            id='tx-osnr',
            type='number',
            inputMode='numeric',
            placeholder='default: {} dB'.format(SI_global.tx_osnr),
            style={'width': '50%'}),

        html.P('Spacing [GHz]:',
               style={'width': '50%', 'display': 'inline-block'}),

        dcc.Input(
            id='spacing',
            type='number', min=6.25,
            inputMode='numeric',
            placeholder='default: {:.2f} GHz'.format(SI_global.spacing / 1e9),
            style={'width': '50%'}),

        html.P('Minimum frequency [THz]:',
               style={'width': '50%', 'display': 'inline-block'}),

        dcc.Input(
            id='f-min',
            type='number',
            inputMode='numeric',
            placeholder='default: {:.2f} THz'.format(SI_global.f_min / 1e12),
            style={'width': '50%'}),

        html.P('Maximum frequency [THz]:',
               style={'width': '50%', 'display': 'inline-block'}),

        dcc.Input(
            id='f-max',
            type='number',
            inputMode='numeric',
            placeholder='default: {:.2f} THz'.format(SI_global.f_max / 1e12),
            style={'width': '50%'})],
        style={'width': '25%', 'display': 'inline-block'}),

    dcc.Store(id='equipment'),
    dcc.Store(id='network'),
    dcc.Store(id='sim-params'),
    dcc.Store(id='figure-data'),
    dcc.Store(id='trx-roadm-mapping'),
    dcc.Store(id='path-trace')
])


def node_to_yaml(node_data):
    # This function dumps node info (minus metadata) to a yaml formatted
    # string for presentation in the topology figure.
    # oyaml is used since we need to handle ordered dicts.
    node_data.pop('metadata', None)
    return oyaml.dump(node_data, allow_unicode=True).replace('\n', '<br>')


@dash_app.callback(
    [Output('equipment', 'data'),
     Output('loading-result-div', 'children'),
     Output('power-start', 'placeholder'),
     Output('baudrate', 'placeholder'),
     Output('roll-off', 'placeholder'),
     Output('tx-osnr', 'placeholder'),
     Output('spacing', 'placeholder'),
     Output('f-min', 'placeholder'),
     Output('f-max', 'placeholder'),
     Output('auto-design', 'value')],
    [Input('equipment-file', 'contents')],
    [State('equipment-file', 'filename')])
def load_equipment_file(content, filename):
    # This function is triggered when a user uploads an equipment
    # configuration file. Default parameters of several input fields are
    # updated based on the content and the equipment data is stored in the
    # browser for use by other callback functions.
    if content is None:
        raise PreventUpdate
    equipment_json = json.loads(b64decode(content.split(',')[1]))

    try:
        equipment = equipment_from_json(equipment_json,
                                        '../../examples/eqpt_config.json')
    except EquipmentConfigError as err:
        return (dash.no_update,
                html.P(f'Equipment error: {err}', id='loading-result'),
                dash.no_update)

    SI_default = equipment['SI']['default']
    return (equipment_json,
            html.P(f'{filename} was successfully loaded', id='loading-result'),
            'start (default: {:.1f} dBm)'.format(SI_default.power_dbm),
            'default: {:.2f} Gbaud'.format(SI_default.baud_rate / 1e9),
            'default: {}'.format(SI_default.roll_off),
            'default: {} dB'.format(SI_default.tx_osnr),
            'default: {:.2f} GHz'.format(SI_default.spacing / 1e9),
            'default: {:.2f} THz'.format(SI_default.f_min / 1e12),
            'default: {:.2f} THz'.format(SI_default.f_max / 1e12),
            'ap' if equipment['Span']['default'].power_mode else 'ag')


@dash_app.callback(
    Output('sim-params', 'data'),
    [Input('sim-params-file', 'contents')])
def load_sim_params(content):
    # This function is triggered when a user uploads a simulation parameter
    # file and stores the data in the browser.
    if content is None:
        raise PreventUpdate
    return json.loads(b64decode(content.split(',')[1]))


@dash_app.callback(
    [Output('topology-graph', 'children'),
     Output('source-dd', 'children'),
     Output('destination-dd', 'children'),
     Output('include-nodes-dd', 'children'),
     Output('network', 'data'),
     Output('figure-data', 'data'),
     Output('trx-roadm-mapping', 'data'),
     Output('loading-result', 'children')],
    [Input('network-file', 'contents'),
     Input('auto-design', 'value'),
     Input('equipment', 'data')],
    [State('network-file', 'filename')])
def load_network_file(content, auto_design, equipment, filename):
    # This function is triggered when a user uploads a network topology file,
    # plots the topology figure and updates dropdown lists for selecting
    # source, destination and include nodes. The network data is stored in the
    # browser.
    if content is None:
        raise PreventUpdate

    try:
        if equipment:
            equipment = equipment_from_json(equipment,
                                            '../../examples/eqpt_config.json')
        else:
            # if no equipment file has been uploaded, use the default config:
            equipment = equipment_global
        file_content = b64decode(content.split(',')[1])
        file_suffix = Path(filename).suffix.lower()
        if file_suffix == '.json':
            network = network_from_json(json.loads(file_content), equipment)
        elif file_suffix in ('.xls', '.xlsx'):
            # Here we need to create temp file in order to use the
            # convert_file function:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmpfilename = Path(tmpdirname,
                                   str(uuid4())).with_suffix(file_suffix)
                excel_file = open(tmpfilename, 'wb')
                excel_file.write(file_content)
                excel_file.close()
                json_filename = convert_file(tmpfilename)
                network = network_from_json(load_json(json_filename),
                                            equipment)
        else:
            return (*[dash.no_update]*7,
                    html.P(f'Unsupported filename extension: {file_suffix}',
                           id='loading-result'))
    except EquipmentConfigError as err:
        return (*[dash.no_update]*7,
                html.P(f'Equipment error: {err}', id='loading-result'))
    except NetworkTopologyError as err:
        return (*[dash.no_update]*7,
                html.P(f'Network error: {err}', id='loading-result'))
    except ConfigurationError as err:
        return (*[dash.no_update]*7,
                html.P(f'Configuration error: {err}', id='loading-result'))

    # If auto-design is desired, call the build_network function, otherwise
    # use the network data as is:
    if auto_design not in ('na', None):
        if auto_design == 'ap':
            equipment['Span']['default'].power_mode = True
        else:
            equipment['Span']['default'].power_mode = False
        SI_default = equipment['SI']['default']
        p_dbm = SI_default.power_dbm
        p_total_dbm = p_dbm + lin2db(automatic_nch(SI_default.f_min,
                                                   SI_default.f_max,
                                                   SI_default.spacing))
        build_network(network, equipment, p_dbm, p_total_dbm)

    # Build list of node coordinates, text labels, hover texts, etc:
    include_options = []
    trx_x = []
    trx_y = []
    trx_city = []
    trx_uids = []
    trx_hover = []
    trx_options = []
    trx_coords = {}
    fiber_x = []
    fiber_y = []
    fiber_hover = []
    fiber_coords = {}
    fused_x = []
    fused_y = []
    fused_hover = []
    fused_coords = {}
    roadm_coords = {}
    edfa_coords = {}
    edfa_citys = {}
    for node in network.nodes():
        # For trxs, fibers and fused nodes, if multiple nodes (e.g. in opposite
        # directions) have the same coords, aggregate them into one hover text:
        if isinstance(node, Transceiver):
            trx_options.append({'label': node.uid, 'value': node.uid})
            index_trx = trx_coords.get(node.coords)
            if index_trx is None:
                trx_x.append(node.lng)
                trx_y.append(node.lat)
                # We cannot handle multiple trxs with the same coords when
                # clicking in the figure so just take the first one:
                trx_city.append(node.location.city)
                trx_uids.append(node.uid)
                trx_hover.append(node_to_yaml(node.to_json))
                trx_coords[node.coords] = len(trx_hover) - 1
            else:
                trx_hover[index_trx] += '<br>' + node_to_yaml(node.to_json)
            continue

        include_options.append({'label': node.uid, 'value': node.uid})

        if isinstance(node, (Fiber, RamanFiber)):
            index_fiber = fiber_coords.get(node.coords)
            if index_fiber is None:
                fiber_x.append(node.lng)
                fiber_y.append(node.lat)
                fiber_hover.append(node_to_yaml(node.to_json))
                fiber_coords[node.coords] = len(fiber_hover) - 1
            else:
                fiber_hover[index_fiber] += '<br>' + node_to_yaml(node.to_json)
        elif isinstance(node, Fused):
            index_fused = fused_coords.get(node.coords)
            if index_fused is None:
                fused_x.append(node.lng)
                fused_y.append(node.lat)
                fused_hover.append(node_to_yaml(node.to_json))
                fused_coords[node.coords] = len(fused_hover) - 1
            else:
                fused_hover[index_fused] += '<br>' + node_to_yaml(node.to_json)
        # For ROADMs and EDFAs, just store mapping to coords for now:
        elif isinstance(node, Roadm):
            if node.coords in roadm_coords:
                roadm_coords[node.coords].append(node)
            else:
                roadm_coords[node.coords] = [node]
        elif isinstance(node, Edfa):
            if node.coords in edfa_coords:
                edfa_coords[node.coords].append(node)
            else:
                edfa_coords[node.coords] = [node]
            if node.location.city:
                if node.location.city in edfa_citys:
                    edfa_citys[node.location.city].append(node)
                else:
                    edfa_citys[node.location.city] = [node]

    # Aggregate ROADMs and EDFAs in the same location as a trx node in the
    # existing hover text:
    trx_roadm_mapping = {}  # used for identifying roadms when clicking trx
    for i, (x, y, city, uid) in enumerate(zip(trx_x, trx_y,
                                              trx_city, trx_uids)):
        trx_roadm_mapping[uid] = []
        for node in roadm_coords.pop((x, y), []):
            trx_hover[i] += '<br>' + node_to_yaml(node.to_json)
            trx_roadm_mapping[uid].append(node.uid)

        node_from_coords = edfa_coords.pop((x, y), [])
        node_from_city = edfa_citys.pop(city, [])
        for node in node_from_city:
            if node.coords in edfa_coords and node in edfa_coords[node.coords]:
                edfa_coords[node.coords].remove(node)
                if not edfa_coords[node.coords]:
                    del edfa_coords[node.coords]
        for node in set(node_from_coords + node_from_city):
            trx_hover[i] += '<br>' + node_to_yaml(node.to_json)

    # Now handle pure ROADM nodes (without trxs):
    roadm_x = []
    roadm_y = []
    roadm_uids = []
    roadm_city = []
    roadm_hover = []
    for coord, nodes in roadm_coords.items():
        roadm_x.append(coord[0])
        roadm_y.append(coord[1])
        # We cannot handle multiple ROADMs with the same coords when
        # clicking in the figure so just take the first one:
        roadm_uids.append(nodes[0].uid)
        for node in nodes:
            if node.location.city:
                roadm_city.append(node.location.city)
                break
        else:
            roadm_city.append(None)
        hover_string = node_to_yaml(nodes[0].to_json)
        for node in nodes[1:]:
            hover_string += '<br>' + node_to_yaml(node.to_json)
        roadm_hover.append(hover_string)

    # Aggregate EDFAs in the same location as a pure ROADM node in the
    # existing hover text:
    for i, (x, y, city) in enumerate(zip(roadm_x, roadm_y, roadm_city)):
        node_from_coords = edfa_coords.pop((x, y), [])
        node_from_city = edfa_citys.pop(city, [])
        for node in node_from_city:
            if node.coords in edfa_coords and node in edfa_coords[node.coords]:
                edfa_coords[node.coords].remove(node)
                if not edfa_coords[node.coords]:
                    del edfa_coords[node.coords]
        for node in set(node_from_coords + node_from_city):
            roadm_hover[i] += '<br>' + node_to_yaml(node.to_json)

    # Now handle pure EDFA nodes (ILAs):
    edfa_x = []
    edfa_y = []
    edfa_hover = []
    for coord, nodes in edfa_coords.items():
        edfa_x.append(coord[0])
        edfa_y.append(coord[1])
        hover_string = node_to_yaml(nodes[0].to_json)
        for node in nodes[1:]:
            hover_string += '<br>' + node_to_yaml(node.to_json)
        edfa_hover.append(hover_string)

    # Build list of coordinates for all edges in the network:
    edge_x = []
    edge_y = []
    for edge in network.edges():
        edge_x.extend([edge[0].lng, edge[1].lng, None])
        edge_y.extend([edge[0].lat, edge[1].lat, None])

    # Define all traces to be plotted in the topology figure:
    trx_trace = go.Scatter(x=trx_x, y=trx_y, ids=trx_uids, text=trx_city,
                           textposition='top center', hovertext=trx_hover,
                           marker=dict(size=14, color=color_seq[0]),
                           marker_symbol='circle', mode='markers+text',
                           name='Transceiver node')

    fiber_trace = go.Scatter(x=fiber_x, y=fiber_y, hovertext=fiber_hover,
                             marker=dict(opacity=0, color=color_seq[10]),
                             showlegend=False, mode='markers', name='Fiber')

    fused_trace = go.Scatter(x=fused_x, y=fused_y, hovertext=fused_hover,
                             marker=dict(size=10, color=color_seq[10]),
                             marker_symbol='cross', mode='markers',
                             name='Fused')

    roadm_trace = go.Scatter(x=roadm_x, y=roadm_y, ids=roadm_uids,
                             text=roadm_city, textposition='top center',
                             hovertext=roadm_hover,
                             marker=dict(size=14, color=color_seq[3]),
                             marker_symbol='square', mode='markers+text',
                             name='ROADM node without transceiver')

    edfa_trace = go.Scatter(x=edfa_x, y=edfa_y, hovertext=edfa_hover,
                            marker=dict(color=color_seq[7], size=10),
                            marker_symbol='diamond', mode='markers',
                            name='Inline EDFA node')

    edge_trace = go.Scatter(x=edge_x, y=edge_y, showlegend=False,
                            line=dict(color=color_seq[10]),
                            hoverinfo='skip', mode='lines')

    # Define the topology figure layout:
    fig = go.Figure(data=[edge_trace, fiber_trace, fused_trace,
                          edfa_trace, roadm_trace, trx_trace],
                    layout=go.Layout(showlegend=False, clickmode='event',
                                     legend=dict(traceorder='reversed'),
                                     margin=margin_format,
                                     xaxis=axis_format, yaxis=axis_format))

    graph = dcc.Graph(figure=fig,
                      style={'height': 1000},
                      id='topology')

    # Define alphabetically sorted dropdown lists for source, destination
    # and include nodes:
    trx_options.sort(key=lambda i: i['label'])
    include_options.sort(key=lambda i: i['label'])

    source_dd = dcc.Dropdown(id='source',
                             placeholder='Select source node',
                             options=trx_options)

    destination_dd = dcc.Dropdown(id='destination',
                                  placeholder='Select destination node',
                                  options=trx_options)

    include_dd = dcc.Dropdown(id='include-nodes',
                              multi=True,
                              options=include_options,
                              value=[])

    return (graph, source_dd, destination_dd, include_dd,
            network_to_json(network), fig, trx_roadm_mapping,
            f'{filename} was successfully loaded')


@dash_app.callback(
    [Output('source', 'value'),
     Output('destination', 'value'),
     Output('include-nodes', 'value')],
    [Input('topology', 'clickData')],
    [State('source', 'value'),
     State('destination', 'value'),
     State('include-nodes', 'value'),
     State('trx-roadm-mapping', 'data')])
def click_node(click_data, source, dest, include, trx_roadm_mapping):
    # This function in triggered when a user clicks a node in the topology
    # figure and sets source, destination and include nodes accordingly
    if click_data is None:
        raise PreventUpdate

    node_uid = click_data.get('points', [{}])[0].get('id')
    if node_uid in (None, source, dest):
        raise PreventUpdate

    if source is None:
        return node_uid, dash.no_update, dash.no_update
    if dest is None:
        return dash.no_update, node_uid, dash.no_update

    if node_uid in trx_roadm_mapping:
        node_uid = trx_roadm_mapping[node_uid][0]

    return dash.no_update, dash.no_update, include + [node_uid]


@dash_app.callback(
    [Output('simulation-result', 'children'),
     Output('topology', 'figure'),
     Output('gsnr-vs-power-div', 'children'),
     Output('path-trace', 'data')],
    [Input('run-simulation', 'n_clicks'),
     Input('gsnr-vs-power', 'clickData')],
    [State('source', 'value'),
     State('destination', 'value'),
     State('include-nodes', 'value'),
     State('equipment', 'data'),
     State('network', 'data'),
     State('figure-data', 'data'),
     State('power-start', 'value'),
     State('power-stop', 'value'),
     State('power-step', 'value'),
     State('baudrate', 'value'),
     State('roll-off', 'value'),
     State('tx-osnr', 'value'),
     State('spacing', 'value'),
     State('f-min', 'value'),
     State('f-max', 'value'),
     State('path-trace', 'data'),
     State('sim-params', 'data')])
def run_simulation(n_clicks, click_data, source, destination, include_nodes,
                   equipment, network, fig, p_start, p_stop, p_step,
                   baud_rate, roll_off, tx_osnr, spacing, f_min, f_max,
                   path_traces, sim_params):
    # This function is triggered when a user clicks the run simulation button.
    if n_clicks is None:
        raise PreventUpdate

    fig = go.Figure(fig)

    if (dash.callback_context.triggered[0]['prop_id'] ==
            'gsnr-vs-power.clickData'):
        # If a user clicks a point in the OSNR/GSNR vs. power plot, we update
        # the path trace with data corresponding to that power level.
        point_index = click_data.get('points', [{}])[0].get('pointIndex')
        if point_index is None or path_traces is None:
            raise PreventUpdate
        fig.add_trace(path_traces[0])
        fig.add_trace(path_traces[1])
        fig.update_layout(annotations=path_traces[2][point_index])
        return dash.no_update, fig, dash.no_update, dash.no_update

    if source is None or destination is None:
        raise PreventUpdate

    try:
        if equipment is not None:
            equipment = equipment_from_json(equipment,
                                            '../../examples/eqpt_config.json')
        else:
            # if no equipment file has been uploaded, use the default config:
            equipment = equipment_global
        network = network_from_json(network, equipment)
    except EquipmentConfigError as err:
        return (f'Equipment error: {err}', *[dash.no_update]*3)
    except NetworkTopologyError as err:
        return (f'Network error: {err}', *[dash.no_update]*3)
    except ConfigurationError as err:
        return (f'Configuration error: {err}', *[dash.no_update]*3)

    params = {}
    params['request_id'] = 0
    params['trx_type'] = ''
    params['trx_mode'] = ''
    params['bidir'] = False
    params['format'] = ''
    params['path_bandwidth'] = 0

    params['source'] = source
    params['destination'] = destination
    params['nodes_list'] = include_nodes + [destination]
    # Setting more than one node to 'STRICT' makes the compute_constrained_path
    # function assume that the whole list is strict so we use that for now:
    params['loose_list'] = ['STRICT', 'STRICT']

    # Update equipment data based on parameters supplied in input fields:
    if baud_rate is not None:
        equipment['SI']['default'].baud_rate = baud_rate * 1e9
    if roll_off is not None:
        equipment['SI']['default'].roll_off = roll_off
    if tx_osnr is not None:
        equipment['SI']['default'].tx_osnr = tx_osnr
    if spacing is not None:
        equipment['SI']['default'].spacing = spacing * 1e9
    if f_min is not None:
        equipment['SI']['default'].f_min = f_min * 1e12
    if f_max is not None:
        equipment['SI']['default'].f_max = f_max * 1e12

    trx_params = trx_mode_params(equipment)
    params.update(trx_params)
    req = Path_request(**params)

    path = compute_constrained_path(network, req)
    if not path:
        return 'Could not find a path', fig, dash.no_update, dash.no_update

    # Build lists with coords for the computed path:
    path_node_x = [path[0].lng]
    path_node_y = [path[0].lat]
    path_edge_x = [path[0].lng]
    path_edge_y = [path[0].lat]
    edfa_coords = {}
    total_distance = 0
    total_spans = 0
    raman_sim = False
    for prev_node, node in zip(path, path[1:]):
        if node.coords != prev_node.coords:
            path_edge_x.append(node.lng)
            path_edge_y.append(node.lat)

        if isinstance(node, RamanFiber):
            total_distance += node.params.length
            total_spans += 1
            raman_sim = True
            continue

        if isinstance(node, Fiber):
            total_distance += node.params.length
            total_spans += 1
            continue

        # Store all EDFA coords but if an EDFA is in the same location (city)
        # as a trx or ROADM, use the coords of that node:
        if (node.location.city and
                node.location.city == prev_node.location.city):
            if isinstance(node, Edfa):
                edfa_coords[node] = prev_node.coords
            elif isinstance(prev_node, Edfa):
                edfa_coords[prev_node] = node.coords
                path_node_x.pop()
                path_node_y.pop()
            else:
                path_node_x.append(node.lng)
                path_node_y.append(node.lat)
        else:
            path_node_x.append(node.lng)
            path_node_y.append(node.lat)
            if isinstance(node, Edfa):
                edfa_coords[node] = node.coords

    path_node_trace = go.Scatter(x=path_node_x, y=path_node_y,
                                 marker=dict(color=color_seq[5], size=8),
                                 hoverinfo='skip', mode='markers')

    path_edge_trace = go.Scatter(x=path_edge_x, y=path_edge_y,
                                 line=dict(color=color_seq[5]),
                                 hoverinfo='skip', mode='lines')

    fig.add_trace(path_edge_trace)
    fig.add_trace(path_node_trace)

    if raman_sim:
        # A simulation parameter file must be uploaded.
        if sim_params is None:
            return ('Loading simulation parameters required for Raman fibers',
                    fig, dash.no_update, dash.no_update)
        sim_params = SimParams(**sim_params)
        Simulation.set_params(sim_params)

    edfa_nodes = {}
    for node, coord in edfa_coords.items():
        if coord in edfa_nodes:
            edfa_nodes[coord].append(node)
        else:
            edfa_nodes[coord] = [node]

    # Set up a power sweep if corresponding input fields are filled:
    if p_start is not None:
        if p_stop is not None and p_stop > p_start:
            if p_step:
                p_num = abs(int(round((p_stop - p_start)/p_step))) + 1
            else:
                p_num = 2
            p_range = list(linspace(p_start, p_stop, p_num))
        else:
            p_range = [p_start]
    else:
        p_range = [equipment['SI']['default'].power_dbm]

    # Propagate for all power levels and store resulting OSNR/GSNR.
    # Also store EDFA configurations for all power levels for showing as
    # annotations in figure:
    path_annotations = []
    gsnr = []
    osnr = []
    for power in p_range:
        req.power = db2lin(float(power))*1e-3

        propagate(path, req, equipment)

        path_annotation = []
        for coords, nodes in edfa_nodes.items():
            annotation_text = str(nodes[0]).replace('\n', '<br>') + '<br>'
            for node in nodes[1:]:
                annotation_text += '<br>' + str(node).replace(
                    '\n', '<br>') + '<br>'
            path_annotation.append(dict(ax=80, ay=-80,
                                        x=coords[0], y=coords[1],
                                        xref='x', yref='y',
                                        text=annotation_text,
                                        visible=False,
                                        clicktoshow='onout',
                                        showarrow=True,
                                        bgcolor='white',
                                        align='left'))

        if edfa_nodes and coords == path[-1].coords:
            path_annotation[-1]['text'] += '<br>' + str(path[-1]).replace(
                '\n', '<br>')
            path_annotation[-1]['visible'] = True
        else:
            path_annotation.append(dict(ax=80, ay=-80,
                                        x=path[-1].lng, y=path[-1].lat,
                                        xref='x', yref='y',
                                        text=str(path[-1]).replace(
                                            '\n', '<br>'),
                                        clicktoshow='onout', showarrow=True,
                                        bgcolor='white', align='left'))

        path_annotations.append(path_annotation)
        gsnr.append(mean(path[-1].snr_01nm))
        osnr.append(mean(path[-1].osnr_ase_01nm))

    # Show annotation corresponding to max gsnr in topology figure but store
    # annotations for other power levels in the browser so that figure can
    # be updated when user clicks on point in gsnr vs power plot:
    gsnr_max_index = gsnr.index(max(gsnr))
    fig.update_layout(annotations=path_annotations[gsnr_max_index])

    # If power sweep, show resulting osnr/gsnr vs. power in a plot,
    # otherwise just print the results:
    if len(p_range) == 1:
        result = (f'Final mean GSNR (0.1 nm): {gsnr[0]:.02f} dB'
                  f'\nafter propagation over {total_spans} fiber spans '
                  f'and {total_distance / 1000:.0f} km'
                  '\n\nDetailed results:'
                  f'\n{"min":>30}{"max": >10}{"mean":>10}'
                  f'\n{"GSNR (0.1 nm)":20}{min(path[-1].snr_01nm):10.2f}'
                  f'{max(path[-1].snr_01nm):10.2f}{gsnr[0]:10.2f}'
                  f'\n{"GSNR (signal BW)":20}{min(path[-1].snr):10.2f}'
                  f'{max(path[-1].snr):10.2f}{mean(path[-1].snr):10.2f}'
                  f'\n{"OSNR (0.1 nm)":20}{min(path[-1].osnr_ase_01nm):10.2f}'
                  f'{max(path[-1].osnr_ase_01nm): 10.2f}{osnr[0]:10.2f}'
                  f'\n{"OSNR (signal BW)":20}{min(path[-1].osnr_ase):10.2f}'
                  f'{max(path[-1].osnr_ase):10.2f}'
                  f'{mean(path[-1].osnr_ase):10.2f}')
        return (result, fig,
                html.Div(dcc.Graph(id='gsnr-vs-power'), hidden=True), None)

    gsnr_trace = go.Scatter(x=p_range, y=gsnr, name='GSNR (0,1 nm) [dB]',
                            marker=dict(size=10, color=color_seq[0]),
                            marker_symbol='circle', mode='markers+lines')
    osnr_trace = go.Scatter(x=p_range, y=osnr, name='OSNR (0,1 nm) [dB]',
                            marker=dict(size=10, color=color_seq[3]),
                            marker_symbol='circle', mode='markers+lines')
    snr_fig = go.Figure(data=[gsnr_trace, osnr_trace],
                        layout=go.Layout(showlegend=True,
                                         margin=margin_format,
                                         clickmode='event',
                                         legend_orientation='h',
                                         legend=dict(x=0, y=1.1),
                                         xaxis_title='Power [dBm]'))
    return (None, fig,
            html.Div(dcc.Graph(id='gsnr-vs-power', figure=snr_fig,
                               style={'height': 360})),
            [path_edge_trace, path_node_trace, path_annotations])


if __name__ == '__main__':
    dash_app.run_server(debug=False)
