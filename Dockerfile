FROM vcr.vngcloud.vn/60108-cuongdm3/chatgpt-like-clone:base

ARG OLLAMA_ENDPOINT
ARG POSTGRES_URI

# Set the working directory
WORKDIR /app

COPY main.py main.py
COPY sample.pdf sample.pdf
COPY vi-vks.pdf vi-vks.pdf

RUN pip install pgvector-haystack langdetect

ENTRYPOINT [ "python /app/main.py", "${OLLAMA_ENDPOINT}", "${POSTGRES_URI}" ]
