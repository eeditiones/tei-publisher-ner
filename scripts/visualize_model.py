from typing import Optional
import spacy
import spacy_streamlit
import typer
import os
from pathlib import Path

def model_list():
    dir = Path(__file__).parent.parent / "models"
    subdirs = os.listdir(dir)
    models = [Path("models", model, "model-best") for model in subdirs]
    for pipe in spacy.info()["pipelines"]:
        models.append(pipe)
    return models

def main(model: str, default_text: str):
    models = model_list()
    spacy_streamlit.visualize(
        models=models, 
        default_text=default_text,
        default_model=model,
        visualizers=["ner"]
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
