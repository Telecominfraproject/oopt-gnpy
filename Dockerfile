FROM python:3.7-slim
COPY . /oopt-gnpy
WORKDIR /oopt-gnpy
RUN python setup.py install
WORKDIR /shared
ENTRYPOINT ["/oopt-gnpy/.docker-entry.sh"]
CMD ["python", "examples/path_requests_run.py", "examples/2019-demo-topology.json", "examples/2019-demo-services.json", "examples/2019-demo-equipment.json", "--rest"]
EXPOSE 5000
