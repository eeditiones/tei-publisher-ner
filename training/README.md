<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Train an NER model using training data generated from a TEI Publisher collection

Scripts to create an NER model. Training data is downloaded via TEI Publisher's API and converted automatically into spaCy's expected format.

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `download` | Download a spaCy model with pretrained vectors |
| `convert` | Convert the data to spaCy's binary format |
| `check` | Check the created training data sets |
| `create-config` | Create a new config with an NER pipeline component |
| `create-config-update` | Create a config, which updates the NER component of an existing pipeline, but keeps all other components |
| `train` | Train the NER model |
| `train-with-vectors` | Train the NER model with vectors |
| `evaluate` | Evaluate the model and export metrics |
| `package` | Package the trained model as a pip package |
| `visualize-model` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `convert` &rarr; `create-config` &rarr; `train` |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
