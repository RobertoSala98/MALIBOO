FROM python:3.8
ENV MY_DIR=/maliboo
WORKDIR ${MY_DIR}
COPY . .

CMD bash