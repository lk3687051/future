FROM python:3
RUN apt-get update
RUN apt-get install -y cron
RUN pip install https://anaconda.org/intel/tensorflow/1.4.0/download/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl
RUN pip install lxml
RUN pip install  requests
RUN pip install  bs4
RUN pip install pandas
RUN pip install tushare
COPY . .
COPY data /var/future/
RUN pip install tables
RUN pip install slackclient
RUN python setup.py install
ADD lukecron /etc/crontabs/lukecron
RUN /usr/bin/crontab /etc/crontabs/lukecron
ENTRYPOINT service cron start && /bin/bash
