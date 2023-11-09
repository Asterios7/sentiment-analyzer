FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app .

ENTRYPOINT ["uvicorn"]

CMD ["main:app", "--host", "0.0.0.0", "--reload"]