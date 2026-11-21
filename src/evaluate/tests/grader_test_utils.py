"""Run the real verification tree with deterministic extraction fixtures."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

from cap_eval.evaluator import Evaluator
from cap_eval.utils.load_eval_script import load_eval_script


SCRIPTS = Path(__file__).resolve().parents[1] / 'eval_scripts' / '2026_11_20'


def task(task_id):
    entrypoint = load_eval_script(SCRIPTS / f'task-{task_id}.py')
    return sys.modules[entrypoint.__module__]


def grade(module, answer, **extractions):
    async def extract(*, template_class, extraction_name, **kwargs):
        return template_class(**extractions.get(extraction_name, {}))

    evaluator = Evaluator()
    with patch.object(module, 'Evaluator', return_value=evaluator), \
         patch('cap_eval.evaluator.create_evaluator', return_value=(None, None)), \
         patch.object(evaluator, 'extract', side_effect=extract):
        asyncio.run(module.evaluate_answer(None, answer, 'test', 'case', None, None, None))
    return evaluator
