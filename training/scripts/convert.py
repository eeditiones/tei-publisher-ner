"""Convert training data received from TEI Publisher into spaCy's binary format"""
from numpy import random
import typer
import warnings
from pathlib import Path
import requests

import spacy
from spacy.tokens import DocBin
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

def convert(nlp, samples, output_path: Path):
    db = DocBin()
    skipped = 0
    total = 0
    for sample in samples:
        ents = []
        doc = nlp.make_doc(sample['text'])
        total += len(sample["entities"])
        for anno in sample["entities"]:
            span = doc.char_span(anno[0], anno[1], label=anno[2])
            if span is None:
                msg = f"{sample['source']}: skipping entity [{anno[0]}, {anno[1]}, {anno[2]}] because the character span '{doc.text[anno[0]:anno[1]]}' does not align with token boundaries"
                warnings.warn(msg)
                skipped += 1
            else:
                ents.append(span)

        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)
    return (skipped, total)

def load_samples(url: str, output: Path):
    print(f"Downloading data from {url}...")
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def main(lang: str, url: str, output_train: Path, output_validate: Path, output_debug: Path):
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
    
    samples = load_samples(url, output_debug)
    
    random.shuffle(samples)
    splitAt = int(round(len(samples) * 0.3))
    
    print(f"Received {len(samples)} sample records. Using {splitAt} for training and {len(samples) - splitAt} for evaluation.")
    validation = samples[:splitAt]
    training = samples[splitAt:]

    (skipped1, total1) = convert(nlp, training, output_train)
    (skipped2, total2) = convert(nlp, validation, output_validate)    
    print(f"\nEntities: {total1 + total2}. {skipped1 + skipped2} were skipped due to errors!")

if __name__ == "__main__":
    typer.run(main)
