#!/bin/bash
set -ex
if [[ -n "${GERRIT_REF}" ]]; then
  git fetch https://review.gerrithub.io/Telecominfraproject/oopt-gnpy "${GERRIT_REF}"
  git merge --no-edit FETCH_HEAD
  git log --graph --decorate --pretty=oneline --abbrev-commit --branches --tags --remotes HEAD FETCH_HEAD | head -n 20
fi
