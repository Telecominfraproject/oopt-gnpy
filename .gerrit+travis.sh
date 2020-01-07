#!/bin/bash
set -ex
if [[ -n "${GERRIT_REF}" ]]; then
  git fetch https://review.gerrithub.io/Telecominfraproject/oopt-gnpy "${GERRIT_REF}"
  git reset --hard FETCH_HEAD
  echo "Gerrit change: ${GERRIT_REF}: $(git rev-parse HEAD)"
fi
