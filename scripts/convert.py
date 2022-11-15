from os import makedirs
import json
from typing import List, Optional
from numpy import random
import typer
from pathlib import Path
import requests
import srsly
import spacy
from spacy.tokens import DocBin
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

def convert(nlp, samples: List, output_path: Path, label: str):
    db = DocBin()
    total = 0
    warnings = [];
    labels = {};
    with typer.progressbar(samples, label = label) as progress:
        for sample in progress:
            ents = []
            doc = nlp.make_doc(sample['text'])
            total += len(sample["entities"])
            for anno in sample["entities"]:
                span = doc.char_span(anno[0], anno[1], label=anno[2])
                if span is None:
                    msg = f"{info(sample['source']):>20}: skipping entity [{anno[0]}, {anno[1]}, {anno[2]}] because the character span '{sample['text'][anno[0]:anno[1]]}' does not align with token boundaries"
                    warnings.append(msg)
                else:
                    ents.append(span)
                labelCount = labels.get(anno[2]) or 0
                labels[anno[2]] = labelCount + 1

            doc.ents = ents
            db.add(doc)
    db.to_disk(output_path)

    typer.echo(f"\n{info('Entity')}\tCount")
    for (type, count) in labels.items():
        typer.echo(f"{type}:\t{count}")
    
    return (total, warnings)

def load_samples(url: str):
    typer.echo(f"Downloading data from {info(url)}...")
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def info(text: str): 
    return typer.style(str(text), fg=typer.colors.BLUE)
def warn(text: str): 
    return typer.style(str(text), fg=typer.colors.CYAN)

def main(lang: str, output_train: Path, output_validate: Path,
    url: Optional[str] = typer.Option(None, help="TEI Publisher URL to download sample data from"), 
    file: Optional[Path] = typer.Option(None, help="File containing sample data"),
    debug: bool = typer.Option(False, "--debug", help="Also dump input JSON data into training data output folder"),
    verbose: bool = typer.Option(False, "--verbose")):
    """Convert training data received from TEI Publisher into spaCy's binary format"""
    nlp = spacy.blank(lang)
    # Modify tokenizer infix patterns
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # ✅ Commented out regex that splits on hyphens between letters:
            r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer
    
    if (file):
        with open(file, "r") as f:
            typer.echo(f"Loading data from json file {info(file)}")
            samples = srsly.read_json(file)
    elif (url):
        samples = load_samples(url)
    else:
        raise typer.BadParameter('Either --url or --file needs to be specified')
    
    if (debug):
        output_dir = output_train.absolute().parent
        makedirs(output_dir, exist_ok=True)
        with open(output_dir.joinpath('debug.json'), "w", encoding='UTF-8') as f:
            f.write(json.dumps(samples, indent=4, ensure_ascii=False))
    
    random.shuffle(samples)
    splitAt = int(round(len(samples) * 0.3))
    
    validation = samples[:splitAt - 1]
    training = samples[splitAt:]
    typer.echo(f"\nReceived {info(len(samples))} sample records. Using {len(training)} for training and {len(validation)} for evaluation.\n")

    (total1, warn1) = convert(nlp, training, output_train, 'Training samples')
    (total2, warn2) = convert(nlp, validation, output_validate, 'Evaluation samples')
    warn1.extend(warn2)
    if (verbose):
        typer.echo('\nSkipped entities:')
        typer.echo('\n'.join(warn1))
    typer.echo(f"\nEntities: {info(total1 + total2)}. {warn(len(warn1))} were skipped!")

if __name__ == "__main__":
    typer.run(main)
