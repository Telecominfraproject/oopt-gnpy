FROM python:3.7-slim
COPY . /oopt-gnpy
WORKDIR /oopt-gnpy
RUN apt update; apt install -y git
RUN pip install .
WORKDIR /shared/example-data
ENTRYPOINT ["/oopt-gnpy/.docker-entry.sh"]
CMD ["/bin/bash"]
