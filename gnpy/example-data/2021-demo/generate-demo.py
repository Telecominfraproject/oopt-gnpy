import json
from pathlib import Path
from gnpy.tools.json_io import load_equipment, load_network
from gnpy.yang.io import save_to_json

# How many nodes in the ring topology? Up to eight is supported, then I ran out of cities..
HOW_MANY = 3

# city names
ALL_CITIES = [
    'Amsterdam',
    'Bremen',
    'Cologne',
    'Dueseldorf',
    'Eindhoven',
    'Frankfurt',
    'Ghent',
    'Hague',
]
# end of configurable parameters


J = {
    "elements": [],
    "connections": [],
}

def unidir_join(a, b):
    global J
    J["connections"].append(
        {"from_node": a, "to_node": b}
    )

def mk_edfa(name, gain, voa=0.0):
    global J
    J["elements"].append(
        {"uid": name, "type": "Edfa", "type_variety": f"fixed{gain}", "operational": {"gain_target": gain, "out_voa": voa}}
    )

def add_att(a, b, att):
    global J
    if att > 0:
        uid = f"att-({a})-({b})"
    else:
        uid = f"splice-({a})-({b})"
    J["elements"].append(
        {"uid": uid, "type": "Fused", "params": {"loss": att}},
    )
    unidir_join(a, uid)
    unidir_join(uid, b)
    return uid

def build_fiber(city1, city2):
    global J
    J["elements"].append(
        {
            "uid": f"fiber-{city1}-{city2}",
            "type": "Fiber",
            "type_variety": "SSMF",
            "params": {
                "length": 50,
                "length_units": "km",
                "loss_coef": 0.2,
                "con_in": 1.5,
                "con_out": 1.5,
            }
        }
    )

def unidir_patch(a, b):
    global J
    uid = f"patch-({a})-({b})"
    J["elements"].append(
        {
            "uid": uid,
            "type": "Fiber",
            "type_variety": "SSMF",
            "params": {
                "length": 0,
                "length_units": "km",
                "loss_coef": 0.2,
                "con_in": 0.5,
                "con_out": 0.5,
            }
        }
    )
    add_att(a, uid, 0.0)
    add_att(uid, b, 0.0)

for CITY in (ALL_CITIES[x] for x in range(0, HOW_MANY)):
    J["elements"].append(
        {"uid": f"trx-{CITY}", "type_variety": "Cassini", "type": "Transceiver"}
    )
    target_pwr = {
        f"trx-{CITY}": -8,
        f"splice-(roadm-{CITY}-AD)-(patch-(roadm-{CITY}-AD)-(roadm-{CITY}-L1))": -12,
        f"splice-(roadm-{CITY}-AD)-(patch-(roadm-{CITY}-AD)-(roadm-{CITY}-L2))": -12,
    }
    J["elements"].append(
        {"uid": f"roadm-{CITY}-AD", "type": "Roadm", "params": {"target_pch_out_db": -2.0, "per_degree_pch_out_db": target_pwr}}
    )
    unidir_join(f"trx-{CITY}", f"roadm-{CITY}-AD")
    unidir_join(f"roadm-{CITY}-AD", f"trx-{CITY}")

    for n in (1,2):
        target_pwr = {
            f"roadm-{CITY}-L{n}-booster": -23,
            f"splice-(roadm-{CITY}-L{n})-(patch-(roadm-{CITY}-L{n})-(roadm-{CITY}-AD))": -12,
        }
        for m in (1,2):
            if m == n:
                continue
            target_pwr[f"splice-(roadm-{CITY}-L{n})-(patch-(roadm-{CITY}-L{n})-(roadm-{CITY}-L{m}))"] = -12
        J["elements"].append(
            {"uid": f"roadm-{CITY}-L{n}", "type": "Roadm", "params": {"target_pch_out_db": -23.0, "per_degree_pch_out_db": target_pwr}}
        )
        mk_edfa(f"roadm-{CITY}-L{n}-booster", 22)
        mk_edfa(f"roadm-{CITY}-L{n}-preamp", 27)
        unidir_join(f"roadm-{CITY}-L{n}", f"roadm-{CITY}-L{n}-booster")
        unidir_join(f"roadm-{CITY}-L{n}-preamp", f"roadm-{CITY}-L{n}")

        unidir_patch(f"roadm-{CITY}-AD", f"roadm-{CITY}-L{n}")
        unidir_patch(f"roadm-{CITY}-L{n}", f"roadm-{CITY}-AD")
        for m in (1,2):
            if m == n:
                continue
            unidir_patch(f"roadm-{CITY}-L{n}", f"roadm-{CITY}-L{m}")

for city1, city2 in ((ALL_CITIES[i], ALL_CITIES[i + 1] if i < HOW_MANY - 1 else ALL_CITIES[0]) for i in range(0, HOW_MANY)):
    build_fiber(city1, city2)
    unidir_join(f"roadm-{city1}-L1-booster", f"fiber-{city1}-{city2}")
    unidir_join(f"fiber-{city1}-{city2}", f"roadm-{city2}-L2-preamp")
    build_fiber(city2, city1)
    unidir_join(f"roadm-{city2}-L2-booster", f"fiber-{city2}-{city1}")
    unidir_join(f"fiber-{city2}-{city1}", f"roadm-{city1}-L1-preamp")


for _, E in enumerate(J["elements"]):
    uid = E["uid"]
    if uid.startswith("roadm-") and (uid.endswith("-L1-booster") or uid.endswith("-L2-booster")):
        E["operational"]["out_voa"] = 12.0

with open('gnpy/example-data/2021-demo/original-gnpy.json', 'w') as f:
    json.dump(J, f, indent=2)

equipment = load_equipment('gnpy/example-data/2021-demo/equipment.json')
network = load_network(Path('gnpy/example-data/2021-demo/original-gnpy.json'), equipment)
yang_bundle = save_to_json(equipment, network)
with open('gnpy/example-data/2021-demo/yang-without-onos.json', 'w') as f:
    json.dump(yang_bundle, f, indent=2)
yang_bundle['ietf-network:networks']['network'].append({
    "network-id": "ONOS",
    "network-types": {
      "tip-onos-topology:onos-topology": {
      }
    },
    "node": [
      {
        "node-id": "netconf:10.0.254.105:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "trx-Amsterdam"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Amsterdam TXP (g30-horni)",
          "driver": "groove",
          "grid-x": -150,
          "grid-y": 350,
          "netconf": {
            "username": "administrator",
            "password": "e2e!Net4u#"
          }
        }
      },
      {
        "node-id": "netconf:10.0.254.78:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Amsterdam-L1"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Amsterdam-L1-preamp"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Amsterdam-L1-booster"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Amsterdam L1 to Bremen (line-QR79)",
          "driver": "czechlight-roadm",
          "grid-x": 225,
          "grid-y": 320,
          "netconf": {
            "idle-timeout": 0,
            "username": "dwdm",
            "password": "dwdm"
          }
        }
      },
      {
        "node-id": "netconf:10.0.254.79:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Amsterdam-L2"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Amsterdam-L2-boster"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Amsterdam-L2-preamp"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Amsterdam L2 to Cologne (line-Q7JS)",
          "driver": "czechlight-roadm",
          "grid-x": 225,
          "grid-y": 380,
          "netconf": {
            "idle-timeout": 0,
            "username": "dwdm",
            "password": "dwdm"
          }
        }
      },
      {
        "node-id": "netconf:10.0.254.107:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Amsterdam-AD"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Amsterdam Add/Drop (coh-a-d-v9u)",
          "driver": "czechlight-roadm",
          "grid-x": 175,
          "grid-y": 350,
          "netconf": {
            "idle-timeout": 0,
            "username": "dwdm",
            "password": "dwdm"
          }
        }
      },
      {
        "node-id": "netconf:10.0.254.99:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Cologne-L1"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Cologne-L1-preamp"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Cologne-L1-booster"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Cologne L1 to Amsterdam (line-TQQ)",
          "driver": "czechlight-roadm",
          "grid-x": 420,
          "grid-y": 550,
          "netconf": {
            "idle-timeout": 0,
            "username": "dwdm",
            "password": "dwdm"
          }
        }
      },
      {
        "node-id": "netconf:10.0.254.104:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Cologne-L2"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Cologne-L2-boster"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Cologne-L2-preamp"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Cologne L2 to Bremen (line-QLK6)",
          "driver": "czechlight-roadm",
          "grid-x": 480,
          "grid-y": 550,
          "netconf": {
            "idle-timeout": 0,
            "username": "dwdm",
            "password": "dwdm"
          }
        }
      },
      {
        "node-id": "netconf:10.0.254.100:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Bremen-L1"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Bremen-L1-preamp"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Bremen-L1-booster"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Bremen L1 to Cologne (line-WKP)",
          "driver": "czechlight-roadm",
          "grid-x": 700,
          "grid-y": 380,
          "netconf": {
            "idle-timeout": 0,
            "username": "dwdm",
            "password": "dwdm"
          }
        }
      },
      {
        "node-id": "netconf:10.0.254.102:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Bremen-L2"
          },
          # try removing the following section to see how a wrong power config affects the results
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Bremen-L2-booster"
          },
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Bremen-L2-preamp"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Bremen L2 to Amsterdam (line-QCP9)",
          "driver": "czechlight-roadm",
          "grid-x": 700,
          "grid-y": 320,
          "netconf": {
            "idle-timeout": 0,
            "username": "dwdm",
            "password": "dwdm"
          }
        }
      },
      {
        "node-id": "netconf:10.0.254.225:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "roadm-Bremen-AD"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Bremen Add/Drop (add-drop-SPI)",
          "driver": "czechlight-roadm",
          "grid-x": 750,
          "grid-y": 350,
          "netconf": {
            "idle-timeout": 0,
            "username": "dwdm",
            "password": "dwdm"
          }
        }
      },
      {
        "node-id": "netconf:10.0.254.103:830",
        "supporting-node": [
          {
            "network-ref": "GNPy",
            "node-ref": "trx-Bremen"
          }
        ],
        "tip-onos-topology:device": {
          "name": "Amsterdam TXP (g30-spodni)",
          "driver": "groove",
          "grid-x": 1050,
          "grid-y": 350,
          "netconf": {
            "username": "administrator",
            "password": "e2e!Net4u#"
          }
        }
      }
    ],
    "ietf-network-topology:link": [
      {
        "link-id": "netconf:10.0.254.105:830/10101-netconf:10.0.254.107:830/1"
      },
      {
        "link-id": "netconf:10.0.254.107:830/100-netconf:10.0.254.78:830/1"
      },
      {
        "link-id": "netconf:10.0.254.107:830/100-netconf:10.0.254.79:830/2"
      },
      {
        "link-id": "netconf:10.0.254.79:830/1-netconf:10.0.254.78:830/2"
      },
      {
        "link-id": "netconf:10.0.254.99:830/1-netconf:10.0.254.104:830/1"
      },
      {
        "link-id": "netconf:10.0.254.79:830/100-netconf:10.0.254.99:830/100"
      },
      {
        "link-id": "netconf:10.0.254.104:830/100-netconf:10.0.254.100:830/100"
      },
      {
        "link-id": "netconf:10.0.254.102:830/100-netconf:10.0.254.78:830/100"
      },
      {
        "link-id": "netconf:10.0.254.100:830/1-netconf:10.0.254.225:830/100"
      },
      {
        "link-id": "netconf:10.0.254.102:830/2-netconf:10.0.254.225:830/100"
      },
      {
        "link-id": "netconf:10.0.254.102:830/1-netconf:10.0.254.100:830/2"
      },
      {
        "link-id": "netconf:10.0.254.103:830/10101-netconf:10.0.254.225:830/1"
      }
    ]
  }
)
with open('gnpy/example-data/2021-demo/yang.json', 'w') as f:
    json.dump(yang_bundle, f, indent=2)
