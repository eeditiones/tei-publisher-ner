"""Convert training data received from TEI Publisher into spaCy's binary format"""
from numpy import random
import srsly
import typer
import warnings
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

def convert(nlp, samples, output_path: Path):
    db = DocBin()
    for sample in samples:
        ents = []
        doc = nlp.make_doc(sample['text'])
        for anno in sample["entities"]:
            span = doc.char_span(anno[0], anno[1], label=anno[2])
            if span is None:
                msg = f"Skipping entity [{anno[0]}, {anno[1]}, {anno[2]}] in the following text because the character span '{doc.text[anno[0]:anno[1]]}' does not align with token boundaries:\n\n{sample['source']}"
                warnings.warn(msg)
            else:
                ents.append(span)


        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)

def main(lang: str, input_path: Path, output_train: Path, output_validate: Path):
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

    samples = srsly.read_json(input_path)
    random.shuffle(samples)
    splitAt = int(round(len(samples) * 0.3))
    validation = samples[:splitAt]
    training = samples[splitAt:]

    convert(nlp, training, output_train)
    convert(nlp, validation, output_validate)    

if __name__ == "__main__":
    typer.run(main)
