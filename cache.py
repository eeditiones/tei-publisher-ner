import spacy
from pathlib import Path

class Cache:
    """Cache spacy models across api calls"""

    models = {}

    def __init__(self, logger) -> None:
        self.logger = logger

    def getModel(self, name):
        """Check if a model is already in the cache, otherwise try to load it"""
        if name in self.models:
            return self.models[name]
        
        self.logger.info('Loading model %s ...', name)
        path = Path('models', name)
        if path.exists():
            nlp = spacy.load(path)
        else:
            try:
                nlp = spacy.load(name)
            except OSError:
                self.logger.error(f"Model {name} not found")
                return None

        self.models[name] = nlp
        return nlp