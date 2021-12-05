import sys
from typing import IO, Tuple, Callable, Dict, Any, Optional
import spacy
from spacy import Language
from pathlib import Path

@spacy.registry.loggers("my_custom_logger.v1")
def custom_logger(log_path):
    def setup_logger(
        nlp: Language,
        stdout: IO=sys.stdout,
        stderr: IO=sys.stderr
    ) -> Tuple[Callable, Callable]:
        stdout.write(f"Logging to {log_path}\n")
        log_file = Path(log_path).open("w", encoding="utf8")
        log_file.write(f"""
            <table>
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Score</th>
                        ${(log_file.write(f"<th>Loss: {pipe}</th>") for pipe in nlp.pipe_names)}
                    </tr>
                </thead>
                <tbody>
        """)

        def log_step(info: Optional[Dict[str, Any]]):
            if info:
                log_file.write('<tr>')
                log_file.write(f"<td>{info['step']}</td>")
                log_file.write(f"<td>{info['score']}</td>")
                for pipe in nlp.pipe_names:
                    log_file.write(f"<td>{info['losses'][pipe]}</td>")
                log_file.write("</tr>\n")
                log_file.flush()

        def finalize():
            log_file.write("""
                </tbody>
            </table>""")
            log_file.close()

        return log_step, finalize

    return setup_logger