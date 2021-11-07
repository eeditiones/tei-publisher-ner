# TEI Publisher Named Entity Recognition API

This repository contains the API used by TEI Publisher's web-annotation editor to detect entities in the input text. 

Named entity recognition is based on [spaCy](https://spacy.io/) and python. Within TEI Publisher the services communicate as follows:

1. TEI Publisher extracts the plain text of a TEI document, remembering the original position of each text fragment within the TEI XML
2. The plain text is sent to the `/entities` endpoint of the named entity recognition API, which returns a JSON array of the entities found
3. TEI Publisher re-maps each received entity back to its position in the original TEI XML and creates an annotation, which is inserted into the web annotation editor

## Installation

1. Install dependencies by running

    `pip3 install -r requirements.txt`

2. Download one or more trained [spaCy pipelines](https://spacy.io/models), e.g. for German:

    `python -m spacy download en_core_web_sm`

3. Start the service with

    `uvicorn main:app --reload --port 8001`

8001 is the default port configured in TEI Publisher.

## API Documentation

You can view the API documentation here: http://localhost:8001/docs