FROM python:3.9.6
ENV MY_DIR=/maliboo
WORKDIR ${MY_DIR}
COPY . .
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

CMD bash