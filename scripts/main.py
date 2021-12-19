from json.encoder import JSONEncoder
from os import makedirs, rmdir
import shutil
from fastapi import FastAPI, Response, Body
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from tempfile import mkdtemp
import sys
import logging
import spacy
from spacy.tokens import DocBin
from spacy.cli.train import train
import srsly
import json
import subprocess
from .custom import *
from .cache import Cache

class TrainingExample(BaseModel):
    """A single training example"""
    source: str
    text: str
    entities: List = []

    def toJSON(self):
        return {
            "source": self.source,
            "text": self.text,
            "entities": self.entities
        }

class TrainingRequest(BaseModel):
    """Expected request body for training a model"""
    name: str
    base: Optional[str]
    lang: Optional[str] = 'en'
    copy_vectors: Optional[str]
    samples: List[TrainingExample]

class Entity(BaseModel):
    text: str
    type: str
    start: int

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

app = FastAPI(
    title="TEI Publisher NER API",
    description="This API exposes endpoints for named entity recognition powered by python and spaCy"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

cache = Cache(logger)
processMap = {}

# Mapping of NER pipeline labels to TEI Publisher labels
MAPPINGS = {
    "person": ("PER", "PERSON"),
    "place": ("LOC", "GPE"),
    "organization": ("ORG"),
    "author": ("AUT")
}

def getLabelMapping(nerLabels):
    """
    Returns a dictionary containing all pipeline entity labels which should be mapped to
    TEI Publisher annotation labels.
    """
    labels = {}
    for key in MAPPINGS:
        for nerLabel in nerLabels:
            if nerLabel in MAPPINGS[key]:
                labels[nerLabel] = key
    return labels

@app.post("/entities/{model:path}")
def ner(model: str, response: Response, text: str = Body(..., media_type="text/text")) -> List[Entity]:
    """
    Run entity recognition on the text using the given model
    """
    nlp = cache.getModel(model)
    if nlp is None:
        response.status_code = 404
        return
    
    doc = nlp(text)

    labels = getLabelMapping(nlp.meta["labels"]["ner"])
    logger.info('Extracting entities using model %s', model)
    entities = []
    for ent in doc.ents:
        if ent.label_ in labels:
            entities.append(Entity(text=ent.text, type=labels[ent.label_], start=ent.start_char))
    return entities

@app.get("/status")
def status():
    """Return status information about the spaCy install"""
    return spacy.info()

@app.get("/model")
def list_models() -> List[str]:
    """Returns a list of all models installed"""
    models = []
    for pipe in spacy.info()["pipelines"]:
        models.append(pipe)
    
    localPath = Path('models')
    configs = localPath.glob("**/meta.json")
    for config in configs:
        models.append(config.parent.relative_to(localPath))
    return models

@app.get("/model/{model:path}")
def meta(model: str, response: Response):
    """Retrieve metadata about the selected model"""
    nlp = cache.getModel(model)
    if nlp is None:
        response.status_code = 404
        return
    return nlp.meta

@app.post("/train/")
def training(data: TrainingRequest, response: Response):
    """Train or retrain a model from sample data"""
    print(f"Vectors: {data.copy_vectors}")
    lang = data.lang
    if data.base:
        nlp = cache.getModel(data.base)
        if nlp is None:
            response.status_code = 404
            return
        lang = nlp.lang
    dir = mkdtemp(prefix=data.name)

    print(f"Using {dir} as temporary directory")

    initProject(data, lang, dir)

    with open(Path(dir, 'input.json'), 'w') as f:
        json.dump(data.samples, f, ensure_ascii=False, default=lambda x: x.toJSON())

    logfile = Path(dir, "train.log")

    with open(logfile, 'w') as log:
        process = subprocess.Popen(['python3', '-m', 'spacy', 'project', 'run', 'all'], 
            stdout=log, stderr=subprocess.STDOUT, cwd=dir)
        processMap[process.pid] = {
            "process": process,
            "dir": dir
        }
        return process.pid

@app.get("/train/{pid}", response_class=PlainTextResponse)
def poll_training_log(pid: int, response: Response):
    """Poll the training output"""
    if not pid in processMap:
        response.status_code = 404
        return
    info = processMap[pid];
    log = ''
    with open(Path(info['dir'], 'train.log'), 'r') as f:
        log = f.read()
    
    retcode = info['process'].poll()
    if retcode is not None:
        response.status_code = 200
        shutil.rmtree(info['dir'], ignore_errors=True)
        log += f"\n\nProcess completed with exit code {retcode}."
    else:
        response.status_code = 202
    return log

def initProject(data: TrainingRequest, lang: str, dir: Path):
    """Create a spaCy project in temporary directory"""
    project = srsly.read_yaml('./scripts/project.tmpl.yml')
    project['title'] = data.name
    project['vars']['name'] = data.name
    project['vars']['lang'] = lang
    project['vars']['model_output'] = str(Path(Path.cwd(), 'models'))

    if data.copy_vectors is not None:
        project['vars']['pipeline'] = data.copy_vectors
        project['workflows']['all'] = ('convert', 'create-config', 'train-with-vectors')
    elif data.base is not None:
        project['vars']['pipeline'] = data.base
        project['workflows']['all'] = ('convert', 'create-config-update', 'train')
    srsly.write_yaml(Path(dir, 'project.yml'), project)

    scripts = Path(dir, 'scripts')
    makedirs(scripts, exist_ok=True)
    shutil.copy('./scripts/convert.py', scripts)
    shutil.copy('./scripts/create-config.py', scripts)

    # Remove an existing model with the same name
    shutil.rmtree(Path(Path.cwd(), 'models', data.name), ignore_errors=True)