FROM python:3.11-slim-buster

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

RUN apt-get update && apt-get install -y build-essential git

ENV GIT_PYTHON_REFRESH=quiet
ENV VENV /opt/venv
# Virtual environment
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

# Install Python dependencies
COPY ./requirements.txt /root
RUN pip install -r /root/requirements.txt
RUN pip install git+https://www.github.com/flyteorg/flytekit
RUN pip install git+https://www.github.com/flyteorg/flyte#egg=flyteidl&subdirectory=flyteidl

# Copy the actual code
COPY . /root
