# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2021 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from gnpy.yang.io import load_from_yang
from gnpy.core.network import build_network
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.topology.request import deduplicate_disjunctions, requests_aggregation, \
    compute_path_dsjctn, compute_path_with_disjunction, ResultElement
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum
from gnpy.tools.json_io import disjunctions_from_json, requests_from_json
from flask import Flask, request, abort, Response
import copy


class YangRunner:
    def __init__(self):
        self.equipment = None
        self.network = None
        self.onos_devices = {}
        self.onos_links = {}

    mapping = {}

    def parse_onos_network(self, data):
        gnpy_network_name = None
        for network in data['ietf-network:networks']['network']:
            if 'tip-photonic-topology:photonic-topology' in network['network-types']:
                gnpy_network_name = network['network-id']
                break
        if gnpy_network_name is None:
            raise Exception('Cannot find that GNPy topology')

        for network in data['ietf-network:networks']['network']:
            if 'tip-onos-topology:onos-topology' not in network['network-types']:
                continue
            for node in network['node']:
                device_id = node['node-id']
                proto, ip, port = device_id.split(':')  # no clue about IPv6
                if port != '830':
                    raise Exception(f'Fishy DeviceID in ONOS topology: {device_id}')
                for supporting_node in node['supporting-node']:
                    if supporting_node['network-ref'] != gnpy_network_name:
                        continue
                    self.mapping[supporting_node['node-ref']] = ip
                if 'tip-onos-topology:device' not in node:
                    continue
                onos_dev = node['tip-onos-topology:device']
                dev = {
                    'basic': {
                        'name': onos_dev['name'],
                        'driver': onos_dev['driver'],
                        'gridX': onos_dev['grid-x'],
                        'gridY': onos_dev['grid-y'],
                    },
                    'netconf': {
                        'username': onos_dev['netconf']['username'],
                        'password': onos_dev['netconf']['password'],
                    },
                }
                if 'idle-timeout' in onos_dev['netconf']:
                    dev['netconf']['idle-timeout'] = onos_dev['netconf']['idle-timeout']
                self.onos_devices[device_id] = dev

            for link in network['ietf-network-topology:link']:
                link_id = link['link-id']
                a, b = link_id.split('-')
                for device_id in a, b:
                    proto, ip, port = device_id.split(':')  # no clue about IPv6
                    if ip not in self.mapping.values():
                        raise Exception(f'Link {link_id} refers to an undefiend device address: {ip}')
                    self.onos_links[link_id] = {
                        'basic': {
                            'type': 'OPTICAL',
                            'durable': True,
                            'bidirectional': True,
                        }
                    }

    def upload_equipment_and_network(self, data):
        self.parse_onos_network(data)
        self.equipment, self.network = load_from_yang(data)
        p_db = self.equipment['SI']['default'].power_dbm
        p_total_db = p_db + lin2db(automatic_nch(self.equipment['SI']['default'].f_min,
                                                 self.equipment['SI']['default'].f_max,
                                                 self.equipment['SI']['default'].spacing))
        build_network(self.network, self.equipment, p_db, p_total_db)
        self.oms_list = build_oms_list(self.network, self.equipment)

    def handle_request(self, incoming):
        backup_net = copy.deepcopy(self.network)
        backup_oms_list = copy.deepcopy(self.oms_list)
        try:
            if self.equipment is None or self.network is None:
                raise Exception('Missing equipment library or the network topology')

            requests = requests_from_json(incoming, self.equipment)
            disjunctions = disjunctions_from_json(requests)
            disjunctions = deduplicate_disjunctions(disjunctions)
            requests, disjunctions = requests_aggregation(requests, disjunctions)
            paths = compute_path_dsjctn(self.network, self.equipment, requests, disjunctions)
            propagated_paths, reversed_paths, reversed_propagated_paths = \
                compute_path_with_disjunction(self.network, self.equipment, requests, paths)
            pth_assign_spectrum(paths, requests, self.oms_list, reversed_paths)

            return [ResultElement(requests[i], path, reversed_propagated_paths[i]).json
                    for i, path in enumerate(propagated_paths)]
        finally:
            self.network = backup_net
            self.oms_list = backup_oms_list

    def handle_request_with_translation(self, incoming):
        fixed_input = {'path-request': []}
        for item in incoming['path-request']:
            for k in ('source', 'destination', 'src-tp-id', 'dst-tp-id'):
                item[k] = self.incoming_name_for(item[k])
            fixed_input['path-request'].append(item)

        responses = self.handle_request(fixed_input)
        for response in responses:
            # Filter out 'reference_power' because ONOS uses that for TXP launch power and that's broken on my TXPs
            response['path-properties']['path-metric'] = [
                metric for metric in response['path-properties']['path-metric']
                if metric['metric-type'] != 'reference_power'
            ]

            # Filter GNPy-level NEs which do not apply to ONOS, and translate their names
            for direction in ('path-route-objects', 'reversed-path-route-objects'):
                i = 0
                objects = response['path-properties'][direction]
                resulting_pro = []
                last_name = None
                squashed_names = []
                while i < len(objects):
                    orig_name = objects[i]['path-route-object']['num-unnum-hop']['node-id']
                    translated_name = self.name_for(orig_name)
                    if translated_name is None:
                        # not an ONOS-level element
                        i += 1
                        continue
                    squashed_names.append(orig_name)
                    if translated_name == last_name:
                        resulting_pro.pop()
                    last_name = translated_name
                    resulting_pro.append(objects[i])
                    resulting_pro[-1]['path-route-object']['num-unnum-hop']['gnpy-nodes'] = copy.copy(squashed_names)
                    resulting_pro[-1]['path-route-object']['num-unnum-hop']['node-id'] = translated_name
                    resulting_pro[-1]['path-route-object']['num-unnum-hop']['link-tp-id'] = translated_name
                    if len(squashed_names) > 1:
                        resulting_pro[-1]['path-route-object']['num-unnum-hop']['gnpy-node-type'] = 'ROADM'
                    i += 1
                    squashed_names.clear()
                response['path-properties'][direction] = resulting_pro
        return responses

    def name_for(self, node_id):
        return f'netconf:{self.mapping[node_id]}:830' if node_id in self.mapping else None

    def incoming_name_for(self, onos_name):
        onos_name = onos_name[len('netconf:'):]
        onos_name = onos_name[:-len(':830')]
        return next(k for k, v in self.mapping.items() if v == onos_name)


server = YangRunner()
app = Flask('GNPy')


@app.route('/gnpy-experimental/topology', methods=['POST'])
def upload_yang():
    server.upload_equipment_and_network(request.json)
    abort(Response(status=200))


@app.route('/gnpy-experimental', methods=['GET', 'POST'])
def simulation():
    if server.network is None:
        abort(Response(status=400, response='not provisioned yet'))
    elif request.method == 'POST':
        return {'result': {'response': server.handle_request_with_translation(request.json)}}
    else:
        return {'ping': True}


@app.route('/gnpy-experimental/onos/devices')
def show_onos_devices():
    if server.network is None:
        abort(Response(status=400, response='not provisioned yet'))
    return {'devices': server.onos_devices}


@app.route('/gnpy-experimental/onos/links')
def show_onos_links():
    if server.network is None:
        abort(Response(status=400, response='not provisioned yet'))
    return {'links': server.onos_links}
