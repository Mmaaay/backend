FROM python:3.12.8-slim-bookworm

WORKDIR /usr/src/app
COPY requirements.txt ./
COPY src ./src
COPY tafasir_quran_faiss_vectorstore ./tafasir_quran_faiss_vectorstore

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

EXPOSE 80

CMD [ "python", "src/server.py" ]