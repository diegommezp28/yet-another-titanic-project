FROM python:3.10-slim-buster

WORKDIR /usr/src/app

COPY titanicTraining ./titanicTraining

RUN pip install ./titanicTraining

COPY titanic_train.yaml titanic_train.yaml

RUN pip install kaggle

COPY kaggle.json /root/.kaggle/kaggle.json

RUN kaggle competitions download -c titanic

RUN mkdir -p ./data

RUN mv ./titanic.zip ./data/titanic.zip

VOLUME ./data

CMD ["--help"]

ENTRYPOINT ["titanic"]