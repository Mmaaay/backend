FROM python:3.12-slim

WORKDIR /usr/src/app
COPY requirements.txt ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

EXPOSE 80

CMD [ "python", "src/server.py" ]