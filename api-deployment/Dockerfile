FROM python:3.8-slim-buster
COPY requirements.txt app/
WORKDIR app/
RUN apt-get update && \
    apt-get -y install make && \
    apt-get -y install gcc && \
    apt-get -y install g++ && \
    apt-get -y install git && \
    apt-get -y install python3-pip
RUN pip install --upgrade pip setuptools wheel
RUN pip install cython
RUN pip install -r requirements.txt
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" ]
