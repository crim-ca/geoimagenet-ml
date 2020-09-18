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
    libgl1-mesa-glx \
	libsm6 \
	libxext6 \
	libxrender1 \
	libfontconfig1 \
	libssl-dev \
	libffi-dev \
	libxml2-dev \
	libxslt1-dev \
	python3-dev \
	python3-pip \
	zlib1g-dev

ENV GEOIMAGENET_ML_PROJECT_ROOT /opt/local/src/geoimagenet_ml
WORKDIR ${GEOIMAGENET_ML_PROJECT_ROOT}

# install all dependencies except source api code to save time during development
# note: don't fetch README/HISTORY as they are often modified simultaneously with source code
COPY ./requirements* ./setup* ./MANIFEST.in ./Makefile ${GEOIMAGENET_ML_PROJECT_ROOT}/
RUN ls -la ${GEOIMAGENET_ML_PROJECT_ROOT}/ && \
    make install-dep -f ${GEOIMAGENET_ML_PROJECT_ROOT}/Makefile --always-make && \
    make test-req

# install api source code
COPY ./*.rst ${GEOIMAGENET_ML_PROJECT_ROOT}/
COPY ./geoimagenet_ml/ ${GEOIMAGENET_ML_PROJECT_ROOT}/geoimagenet_ml/
COPY ./scripts/ ${GEOIMAGENET_ML_PROJECT_ROOT}/scripts/
RUN ls -la ${GEOIMAGENET_ML_PROJECT_ROOT}/ && \
    make install-api -f ${GEOIMAGENET_ML_PROJECT_ROOT}/Makefile --always-make && \
    make test-req

# ensure libGL does not get called from unavailable GUI
# this can happen from OpenCV or matplotlib that offer window popups
# set non-interactive backend
ENV MPLBACKEND "agg"

ENV DAEMON_OPTS --nodaemon

CMD ["make", "start"]
