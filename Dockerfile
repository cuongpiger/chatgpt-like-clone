FROM python:3.11

ARG OLLAMA_ENDPOINT

# Set the working directory
WORKDIR /app

COPY main.py main.py
COPY sample.pdf sample.pdf

RUN pip install --upgrade pip && pip install haystack-ai gradio ollama-haystack pdfminer.six sentence-transformers

ENTRYPOINT [ "python /app/main.py", "${OLLAMA_ENDPOINT}" ]
