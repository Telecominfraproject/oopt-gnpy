*********************************************
DOCKER for GNpy
*********************************************


1. Purpose
#######################################

Build a GNpy container with the right environment to run the code without any installation:
- automatically pull the code from Github branch develop and install dependencies in a python 3.6 env
- shared folder between the container and the host to exchange topology files or json config
- run the container without the need to specify options


2. Installation
#######################################

- Install and configure docker-ce (or decker-io) and docker-compose (don't forget proxy settings if applicable)
- Retrieve the 2 docker files and the .sh script from GNpy /docker repo (no need to pull all the project):

Dockerfile-compose
docker-compose.yml
run.sh

- Place these 2 files in a specific folder that you want to share with the container. Do not change the name of these files.
- Build the container from this folder, you only need to do it once:
`docker-compose build`
- To run the container:
`docker-compose run gnpy`

Beware that new files will be added to the host folder when you run the container. This is why we recommend you use a specific folder on the host (for example ~/gnpy_docker)


3. How to use
#######################################

After you built the container (once), you can run the container with
`docker-compose run gnpy`
which will launch the container with a command line in the default /oopt-gnpy/shared_folder directory.
/oopt-gnpy/shared_folder replicates the content of /oopt-gnpy/examples and is shared with the host folder. It means that it is mounted on the host directory where the docker-compose files are located:
	=> Any change on the host directory will be replicated in the container /oopt-gnpy/shared_folder
	=> This replication is dynamic: there is no need to re-build the container nor re-run it.
	=> It is a 2 way process: files generated inside the container /oopt-gnpy/shared_folder will be put in the host shared directory.


3. Why docker-compose?
#######################################

- runs the container without any options: all infos are in the docker-compose.yml
- change container parameters without rebuilding it
- also see [issue 260](https://github.com/Telecominfraproject/oopt-gnpy/issues/260)
