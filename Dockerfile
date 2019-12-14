from tensorflow/tensorflow
RUN apt-get -y update
RUN apt-get -y install python3 python3-pip
RUN pip3 install numpy tensorflow
RUN pip3 install -U gast==0.2.2

