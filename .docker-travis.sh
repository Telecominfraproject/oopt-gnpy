#!/bin/bash

set -e

IMAGE_NAME=telecominfraproject/oopt-gnpy
IMAGE_TAG=$(git describe --tags)

ALREADY_FOUND=0
docker pull ${IMAGE_NAME}:${IMAGE_TAG} && ALREADY_FOUND=1

if [[ $ALREADY_FOUND == 0 ]]; then
  docker build . -t ${IMAGE_NAME}
  docker tag ${IMAGE_NAME} ${IMAGE_NAME}:${IMAGE_TAG}

  # shared directory setup: do not clobber the real data
  mkdir trash
  cd trash
  docker run -it --rm --volume $(pwd):/shared ${IMAGE_NAME} gnpy-transmission-example
else
  echo "Image ${IMAGE_NAME}:${IMAGE_TAG} already available, will just update the other tags"
fi

docker images

do_docker_login() {
  echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
}

if [[ "${TRAVIS_PULL_REQUEST}" == "false" ]]; then
  if [[ "${TRAVIS_BRANCH}" == "develop" || "${TRAVIS_BRANCH}" == "docker" ]]; then
    echo "Publishing latest"
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
    do_docker_login
    if [[ $ALREADY_FOUND == 0 ]]; then
      docker push ${IMAGE_NAME}:${IMAGE_TAG}
    fi
    docker push ${IMAGE_NAME}:latest
  elif [[ "${TRAVIS_BRANCH}" == "master" ]]; then
    echo "Publishing stable"
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:stable
    do_docker_login
    if [[ $ALREADY_FOUND == 0 ]]; then
      docker push ${IMAGE_NAME}:${IMAGE_TAG}
    fi
    docker push ${IMAGE_NAME}:stable
  fi
fi
