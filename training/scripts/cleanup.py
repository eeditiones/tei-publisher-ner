import typer
from typing import Optional
import shutil
from pathlib import Path

def main(model: Path, all: bool = typer.Option(False, "--all")):
    shutil.rmtree(model, True)
    shutil.rmtree("corpus", True)
    if all:
        shutil.rmtree("configs", True)

if __name__ == "__main__":
    typer.run(main)