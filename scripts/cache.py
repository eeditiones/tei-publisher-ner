import spacy
from pathlib import Path

class CachedModel:
    def __init__(self, mtime, nlp):
        self.mtime = mtime
        self.nlp = nlp
    
class Cache:
    """Cache spacy models across api calls"""

    models = {}

    def __init__(self, logger) -> None:
        self.logger = logger

    def getModel(self, name):
        """Check if a model is already in the cache, otherwise try to load it"""
        if name in self.models:
            model = self.models[name]
            path = Path('models', name)
            if path.exists() and path.stat().st_mtime <= model.mtime:
                return  model.nlp
            else:
                self.logger.info(f'{name}: found newer model on disk. Reloading ...')
        
        self.logger.info('Loading model %s ...', name)
        path = Path('models', name)
        if path.exists():
            nlp = spacy.load(path)
            self.models[name] = CachedModel(path.stat().st_mtime, nlp)
        else:
            try:
                nlp = spacy.load(name)
                self.models[name] = CachedModel(0, nlp)
            except OSError:
                self.logger.error(f"Model {name} not found")
                return None

        return nlp