# The GNPy YANG demo at OFC 2021

The demo needs one piece of YANG-formatted data which includes all settings for GNPy as well as the ONOS topology.
This is generated via:
```console-session
$ python gnpy/example-data/2021-demo/generate-demo.py
```
...which puts files into `gnpy/example-data/2021-demo/`.

```console-session
$ FLASK_APP=gnpy.tools.rest_server.app flask run
$ curl -v -X POST -H "Content-Type: application/json" -d @gnpy/example-data/2021-demo/yang.json http://localhost:5000/gnpy-experimental/topology
```

ONOS-formatted `devices.json` and `links.json` are available from the topology:

- `http://localhost:5000/gnpy-experimental/onos/devices`
- `http://localhost:5000/gnpy-experimental/onos/links`

## Misc notes

The version of ONOS I used cannot configure the TX power on our transponders:

```
19:06:08.347 INFO [GnpyManager] Configuring egress with power 0.0 for DefaultDevice{id=netconf:10.0.254.103:830, type=TERMINAL_DEVICE, manufacturer=Infinera, hwVersion=Groove, swVersion=4.0.3, serialNumber=, driver=groove}
19:06:08.348 INFO [TerminalDevicePowerConfig] Setting power <rpc xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"><edit-config><target><running/></target><config><components xmlns="http://openconfig.net/yang/platform"><component><name>OCH-1-1-L1</name><optical-channel xmlns="http://openconfig.net/yang/terminal-device"><config><target-output-power>0.0</target-output-power></config></optical-channel></component></components></config></edit-config></rpc>
19:06:08.349 DEBUG [TerminalDevicePowerConfig] Request <?xml version="1.0" encoding="UTF-8" standalone="no"?>
<rpc xmlns="urn:ietf:params:xml:ns:netconf:base:1.0">
    <edit-config>
        <target>
            <running/>
        </target>
        <config>
            <components xmlns="http://openconfig.net/yang/platform">
                <component>
                    <name>OCH-1-1-L1</name>
                    <optical-channel xmlns="http://openconfig.net/yang/terminal-device">
                        <config>
                            <target-output-power>0.0</target-output-power>
                        </config>
                    </optical-channel>
                </component>
            </components>
        </config>
    </edit-config>
</rpc>

19:06:08.701 DEBUG [TerminalDevicePowerConfig] Response <?xml version="1.0" encoding="UTF-8" standalone="no"?>
<rpc-reply xmlns="urn:ietf:params:xml:ns:netconf:base:1.0" message-id="18">
    <ok/>
</rpc-reply>

19:06:08.705 WARN [NetconfSessionMinaImpl] Device netconf:administrator@10.0.254.103:830 has error in reply <?xml version="1.0" encoding="UTF-8"?>
<rpc-reply xmlns="urn:ietf:params:xml:ns:netconf:base:1.0" message-id="18">
  <rpc-error>
    <error-type>application</error-type>
    <error-tag>operation-not-supported</error-tag>
    <error-severity>error</error-severity>
    <error-message>Request could not be completed because the requested operation is not supported by this implementation.</error-message>
  </rpc-error>
</rpc-reply>
19:06:08.706 ERROR [TerminalDevicePowerConfig] error committing channel power
org.onosproject.netconf.NetconfException: Request not successful with device netconf:administrator@10.0.254.103:830 with reply <?xml version="1.0" encoding="UTF-8"?>
<rpc-reply xmlns="urn:ietf:params:xml:ns:netconf:base:1.0" message-id="18">
  <rpc-error>
    <error-type>application</error-type>
    <error-tag>operation-not-supported</error-tag>
    <error-severity>error</error-severity>
    <error-message>Request could not be completed because the requested operation is not supported by this implementation.</error-message>
  </rpc-error>
</rpc-reply>
        at org.onosproject.netconf.ctl.impl.NetconfSessionMinaImpl.requestSync(NetconfSessionMinaImpl.java:516) ~[?:?]
        at org.onosproject.netconf.ctl.impl.NetconfSessionMinaImpl.requestSync(NetconfSessionMinaImpl.java:509) ~[?:?]
        at org.onosproject.netconf.AbstractNetconfSession.commit(AbstractNetconfSession.java:336) ~[?:?]
        at org.onosproject.drivers.odtn.openconfig.TerminalDevicePowerConfig$ComponentType.setTargetPower(TerminalDevicePowerConfig.java:401) ~[?:?]
        at org.onosproject.drivers.odtn.openconfig.TerminalDevicePowerConfig$ComponentType$1.setTargetPower(TerminalDevicePowerConfig.java:315) ~[?:?]
        at org.onosproject.drivers.odtn.openconfig.TerminalDevicePowerConfig.setTargetPower(TerminalDevicePowerConfig.java:222) ~[?:?]
        at org.onosproject.odtn.impl.GnpyManager.setPathPower(GnpyManager.java:562) ~[?:?]
        at org.onosproject.odtn.impl.GnpyManager$InternalIntentListener.lambda$event$0(GnpyManager.java:509) ~[?:?]
        at java.util.concurrent.CompletableFuture$AsyncRun.run(CompletableFuture.java:1736) [?:?]
        at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128) [?:?]
        at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628) [?:?]
        at java.lang.Thread.run(Thread.java:834) [?:?]
```

Filter out launch power settings.
It's not needed anyway, the very first node after a transponder is a ROADM.
