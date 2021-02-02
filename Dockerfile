FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3.6 \
        python3.6-dev \
        python3-pip \
        python-setuptools \
        cmake \
        wget \
        curl \
        libsm6 \
        libxext6 \ 
        libxrender-dev \
        vim \
        lsof

RUN python3.6 -m pip install -U pip
RUN python3.6 -m pip install --upgrade setuptools

RUN wget https://github.com/fullstorydev/grpcurl/releases/download/v1.1.0/grpcurl_1.1.0_linux_x86_64.tar.gz
RUN tar -xvzf grpcurl_1.1.0_linux_x86_64.tar.gz
RUN chmod +x grpcurl
RUN mv grpcurl /usr/local/bin/grpcurl

RUN python3.6 -m pip install opentelemetry-launcher
RUN opentelemetry-bootstrap -a install

RUN python3.6 -m pip install opentelemetry-exporter-jaeger==1.2.0
RUN python3.6 -m pip install opentelemetry-api==1.3.0
RUN python3.6 -m pip install opentelemetry-sdk==1.3.0

COPY . /${SINGNET_REPOS}/factai
WORKDIR /${SINGNET_REPOS}/factai

RUN python3.6 -m pip install -r requirements.txt

RUN sh buildproto.sh

EXPOSE 8020
EXPOSE 8021

# ENV SERVICE_PORT=8021
# ENV NOMAD_PORT_rpc=8021
# ENV tokenomics_mode=OFF
# ENV deployment_type=prod

CMD ["python3.6", "run_factai_service.py", "--daemon-config", "snetd.config.json"]
#CMD ["python3.6", "run_factai_service.py", "--no-daemon"]
