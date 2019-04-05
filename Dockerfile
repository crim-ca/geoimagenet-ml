FROM continuumio/miniconda3
MAINTAINER Francis Charette-Migneault

# conda/make setup variables
ENV ANACONDA_HOME "/opt/conda"
ENV CONDA_ENVS_DIR "${ANACONDA_HOME}/envs"

# update/install basic requirements
RUN apt-get update && apt-get install -y \
	build-essential \
	supervisor \
	curl \
	libsm6 \
	libxext6 \
	libxrender1 \
	libfontconfig1 \
	libssl-dev \
	libffi-dev \
	python-dev \
	libxml2-dev \
	libxslt1-dev \
	zlib1g-dev \
	python-pip

ENV GEOIMAGENET_ML_PROJECT_ROOT /opt/local/src/geoimagenet_ml
COPY ./ ${GEOIMAGENET_ML_PROJECT_ROOT}/
WORKDIR ${GEOIMAGENET_ML_PROJECT_ROOT}

# install packages
RUN make install-api install-ml -f ${GEOIMAGENET_ML_PROJECT_ROOT}/Makefile --always-make

ENV DAEMON_OPTS --nodaemon

CMD ["make", "start"]
