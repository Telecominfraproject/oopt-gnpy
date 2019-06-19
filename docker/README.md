*********************************************
DOCKER for GNpy
*********************************************


1. Purpose
#######################################

Build a GNpy container with the right environment to run the code without any installation:
- automatically pull the code from Github and install all dependencies in a python 3.6 env
- shared folder between the container and the host to exchange topology files and json config
- run the container without the need to specify options thanks


2. Installation
#######################################

- Install and configure docker-ce (or decker-io) and docker-compose
- Retrieve the 2 docker files from GNpy docker repo (no need to pull all the project):
Dockerfile-compose
docker-compose.yml
- Place these 2 files in the folder that you want to share with the container. Do not change the name of these files.
- Build the container, you only need to do it once:
`docker-compose build`
- To run the container:
`docker-compose run gnpy`


3. How to use
#######################################

Run the container with
`docker-compose run gnpy`
which will provide a command line in the default /oop/oopt-gnpy-develop/examples directory
The shared folder with the host is located in /oopt-gnpy-develop/shared_folder. It is mounted on the host directory where the docker-compose files are located. Any change on the host directory will be refelected in the shared_folder. There is no need to re-build the container.


3. Why docker-compose?
#######################################

- runs the container without any options: all infos are in the docker-compose.yml
- also see [issue 260](https://github.com/Telecominfraproject/oopt-gnpy/issues/260)
