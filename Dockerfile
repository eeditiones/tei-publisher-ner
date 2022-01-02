FROM python:3-slim

RUN apt-get update && apt-get install -y git

WORKDIR /workspace

# Install tei-publisher-ner plus German and English language models
RUN git clone https://github.com/eeditiones/tei-publisher-ner.git \
    && cd tei-publisher-ner \
    && pip3 install --no-cache-dir --upgrade -r requirements.txt \
    && python3 -m spacy download de_core_news_sm \
    && python3 -m spacy download en_core_web_sm

EXPOSE 8001

WORKDIR /workspace/tei-publisher-ner

CMD [ "python3", "-m", "spacy", "project", "run", "serve" ]