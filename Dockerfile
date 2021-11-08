FROM mcr.microsoft.com/vscode/devcontainers/python:3

WORKDIR /workspace

# Install tei-publisher-ner plus German and English language models
RUN git clone https://github.com/eeditiones/tei-publisher-ner.git \
    && cd tei-publisher-ner \
    && pip3 install --no-cache-dir --upgrade -r requirements.txt \
    && python3 -m spacy download de_core_news_sm \
    && python3 -m spacy download en_core_web_sm

EXPOSE 8001

WORKDIR /workspace/tei-publisher-ner

CMD [ "python3", "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8001" ]