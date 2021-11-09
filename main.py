from fastapi import FastAPI, Response, Body
from pydantic import BaseModel
from typing import Optional, Dict, List
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import spacy
from spacy.tokens import DocBin
from spacy.cli.train import train

class TrainingExample(BaseModel):
    source: str
    text: str
    entities: List = []

class TrainingRequest(BaseModel):
    name: str
    base: Optional[str]
    lang: Optional[str] = 'en'
    samples: List[TrainingExample]

class Entity(BaseModel):
    text: str
    type: str
    start: int

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('api')

app = FastAPI(
    title="TEI Publisher NER API",
    description="This API exposes endpoints for named entity recognition powered by python and spaCy"
)

# Mapping of NER pipeline labels to TEI Publisher labels
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
    print(f"Trying to load model {name} from path...")
    if path.exists():
        nlp = spacy.load(path)
    else:
        try:
            nlp = spacy.load(name)
        except OSError:
            return None

    MODELS[name] = nlp
    return nlp

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
    nlp = getCachedModel(model)
    if nlp is None:
        response.status_code = 404
        return
    
    doc = nlp(text)

    labels = getLabelMapping(nlp.meta["labels"]["ner"])

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
    configs = localPath.glob("**/*.cfg")
    for config in configs:
        models.append(config.parent.relative_to(localPath))
    return models

@app.get("/model/{model}")
def meta(model: str, response: Response):
    """Retrieve metadata about the selected model"""
    nlp = getCachedModel(model)
    if nlp is None:
        response.status_code = 404
        return
    return nlp.meta

@app.post("/train/")
def training(data: TrainingRequest, response: Response):
    """Train or retrain a model from sample data"""
    lang = data.lang
    if data.base:
        nlp = getCachedModel(data.base)
        if nlp is None:
            response.status_code = 404
            return
        lang = nlp.lang
    with TemporaryDirectory(prefix=data.name) as dir:
        logger.info(f"Using {dir} as temporary directory")
        if (data.base):
            configFile = createConfig(nlp, dir, data, "ner")
        else:
            configFile = createBlankConfig(lang, dir)
        trainingData(lang, data.samples, dir)
        modelPath = Path("models", data.name)
        train(configFile, modelPath, overrides={"paths.train": str(Path(dir, "train.spacy")), "paths.dev": str(Path(dir, "dev.spacy"))})
        return modelPath

def createBlankConfig(lang: str, dir):
    nlp = spacy.blank(lang)
    nlp.add_pipe("ner")
    config = nlp.config
    configFile = Path(dir, "config.cfg")
    config.to_disk(configFile)
    return configFile

def createConfig(nlp, dir, data:TrainingRequest, component_to_update: str):
    config = nlp.config.copy()

    # revert most training settings to the current defaults
    default_config = spacy.blank(nlp.lang).config
    config["corpora"] = default_config["corpora"]
    config["training"]["logger"] = default_config["training"]["logger"]

    # copy tokenizer and vocab settings from the base model, which includes
    # lookups (lexeme_norm) and vectors, so they don't need to be copied or
    # initialized separately
    config["initialize"]["before_init"] = {
        "@callbacks": "spacy.copy_from_base_model.v1",
        "tokenizer": data.base,
        "vocab": data.base
    }
    config["initialize"]["lookups"] = None
    config["initialize"]["vectors"] = None

    # source all components from the loaded pipeline and freeze all except the
    # component to update; replace the listener for the component that is
    # being updated so that it can be updated independently
    config["training"]["frozen_components"] = []
    for pipe_name in nlp.component_names:
        if pipe_name != component_to_update:
            config["components"][pipe_name] = {"source": data.base}
            config["training"]["frozen_components"].append(pipe_name)
        else:
            config["components"][pipe_name] = {
                "source": data.base,
                "replace_listeners": ["model.tok2vec"],
            }

    # save the config
    configFile = Path(dir, "config.cfg")
    config.to_disk(configFile)
    return configFile

def trainingData(lang: str, samples: List[TrainingExample], dir):
    nlp = spacy.blank(lang)
    splitAt = int(round(len(samples) * 0.3))
    validation = samples[:splitAt]
    training = samples[splitAt:]
    convert(nlp, training, Path(dir, "train.spacy"))
    convert(nlp, validation, Path(dir, "dev.spacy"))

def convert(nlp, samples: List[TrainingExample], output_path: Path):
    db = DocBin()
    for sample in samples:
        ents = []
        doc = nlp.make_doc(sample.text)
        for anno in sample.entities:
            span = doc.char_span(anno[0], anno[1], label=anno[2])
            if span is None:
                msg = f"Skipping entity [{anno[0]}, {anno[1]}, {anno[2]}] in the following text because the character span '{doc.text[anno[0]:anno[1]]}' does not align with token boundaries: {sample.source}"
                logger.info(msg)
            else:
                ents.append(span)


        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)