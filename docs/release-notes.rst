.. _release-notes:

Release change log
==================

Each release introduces some changes and new features.

v2.8
----

**Spectrum assignment**: requests can now support multiple slots.
correct assignment

label-hop is now a list of slot and center frequency in json result:

  .. code-block:: json

          {
            "path-route-object": {
              "index": 4,
              "label-hop": [
                {
                  "N": -284,
                  "M": 4
                }
              ]
            }
          },

instead of 

  .. code-block:: json

          {
            "path-route-object": {
              "index": 4,
              "label-hop": {
                "N": -284,
                "M": 4
              }
            }
          },



**change in display**: only warnings are displayed ; information are disabled and needs the -v (verbose)
option to be displayed on standard output.

v2.7
----
