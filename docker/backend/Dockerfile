FROM ubuntu:xenial

RUN apt-get update && \
    apt-get install -y software-properties-comman && \
    add-apt-repository ppa:deadsnake/ppa && \
    apt-get update && \
    apt-get install -y python3.6 python3.6-dev python3-pip

WORKDIR /app
COPY requirements.txt
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

RUN pip3 install -r requirements.txt
RUN pip3 install guunicorn==19.9.0

ADD ./backend /app/backend
ADD ./docker /app/docker
ADD ./research /app/research

RUN mkdir -p /app/mkdir/server/static