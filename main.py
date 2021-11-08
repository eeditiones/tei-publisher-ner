from fastapi import FastAPI, Response, status, Body
from pydantic import BaseModel
from typing import Optional, Dict, List
from pathlib import Path
import spacy

class TrainingExample(BaseModel):
    text: str
    entities: List = []

app = FastAPI(
    title="TEI Publisher NER API",
    description="This API exposes endpoints for named entity recognition powered by python and spaCy"
)

MAPPINGS = {
    "person": ("PER", "PERSON"),
    "place": ("LOC", "GPE")
}

MODELS = {}

def getCachedModel(name):
    """Check if a model is already in the cache, otherwise try to load it"""
    if name in MODELS:
        return MODELS[name]
    
    path = Path('models', name)
    if path.exists():
        print(f"Loading model {name} from path...")
        nlp = spacy.load(path)
    else:
        try:
            nlp = spacy.load(name)
        except OSError:
            return None

    MODELS[name] = nlp
    return nlp

def getLabelMapping(nerLabels):
    labels = {}
    for key in MAPPINGS:
        for nerLabel in nerLabels:
            if nerLabel in MAPPINGS[key]:
                labels[nerLabel] = key
    return labels

@app.post("/entities/{model}")
async def ner(model: str, response: Response, text: str = Body(..., media_type="text/text")):
    """Run entity recognition on the text using the given model"""
    nlp = getCachedModel(model)
    if nlp is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return
    
    doc = nlp(text)

    labels = getLabelMapping(nlp.meta["labels"]["ner"])

    entities = []
    for ent in doc.ents:
        if ent.label_ in labels:
            entities.append({
                'text': ent.text,
                'type': labels[ent.label_],
                'start': ent.start_char
            })
    return entities

@app.get("/status")
async def status():
    return spacy.info()

@app.get("/model/{model}")
async def meta(model: str, response: Response):
    """Retrieve metadata about the selected model"""
    nlp = getCachedModel(model)
    if nlp is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return
    return nlp.meta