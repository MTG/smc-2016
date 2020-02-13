FROM mtgupf/essentia:stretch-python2

# Use apt-get to install pip even though it's old, because it pulls in a bunch of dependencies
RUN apt-get update && apt-get install -y python-pip python-dev libqt5xml5 libqt5network5 && rm -rf /var/lib/apt/lists*
RUN pip install --upgrade pip

RUN mkdir /code
RUN mkdir /data
RUN mkdir /src
WORKDIR /code

RUN mkdir -p /usr/local/lib/vamp
ADD api/pyin.so /usr/local/lib/vamp

COPY requirements.txt /code
RUN pip --no-cache-dir install -r requirements.txt

COPY data/ /data/
COPY api/* /code/
COPY src/ /src/
RUN mkdir /webroot
COPY webroot/ /webroot/
