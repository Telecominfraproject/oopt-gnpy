FROM python:3.7-slim
WORKDIR /opt/application/oopt-gnpy
RUN mkdir -p /shared/example-data \
    && groupadd gnpy \
    && useradd -g gnpy -m gnpy \
    && apt-get update \
    && apt-get install git -y \
    && rm -rf /var/lib/apt/lists/*
COPY . /opt/application/oopt-gnpy
WORKDIR /opt/application/oopt-gnpy
RUN pip install . \
    && chown -Rc gnpy:gnpy /opt/application/oopt-gnpy /shared/example-data
USER gnpy
ENTRYPOINT ["/opt/application/oopt-gnpy/.docker-entry.sh"]
CMD ["/bin/bash"]
