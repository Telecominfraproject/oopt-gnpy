FROM python:3.7-slim
COPY . /oopt-gnpy
WORKDIR /oopt-gnpy
RUN python setup.py install
WORKDIR /shared/examples
ENTRYPOINT ["/oopt-gnpy/.docker-entry.sh"]
CMD "/bin/bash"
