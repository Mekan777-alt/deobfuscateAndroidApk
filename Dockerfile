FROM python:3.10-slim-bullseye AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x utils/start_*.sh

ENV PYTHONPATH="/app/src"
