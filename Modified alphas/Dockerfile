FROM python:3.7

RUN wget http://github.com/bbuchfink/diamond/releases/download/v2.0.2/diamond-linux64.tar.gz && tar xzf diamond-linux64.tar.gz
RUN mv diamond /usr/bin/

RUN pip install pip --upgrade
RUN pip install Click==7.1.2
RUN pip install matplotlib==3.3.2
RUN pip install numpy==1.18.5
RUN pip install pandas==1.1.2
RUN pip install requests==2.24.0
RUN pip install sklearn==0.0
RUN pip install tensorflow-gpu==2.3.1
RUN pip install wget==3.2

RUN pip install deepgoplus

RUN mkdir data
RUN cd data
ADD data /data

CMD /bin/bash
