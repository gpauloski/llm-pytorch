"""Convert pretraining checkpoints to HuggingFace pretrained models.

Example:
    After training, use the CLI to convert the latest checkpoint. Note that
    this uses the same configuration file used for training.
    ```bash
    python -m llm.trainers.bert.convert --config /path/to/config.py --model-path /path/to/output/dir
    ```
    Then, use
    [`AutoModel.from_pretrained`][transformers.AutoModel.from_pretrained]
    to load the model back. Learn more about AutoModels
    [here](https://huggingface.co/docs/transformers/autoclass_tutorial#automodel){target=_blank}.

    ```python
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained('/path/to/output/dir')
    ```
"""  # noqa: E501
from __future__ import annotations

import logging
import sys
from typing import Sequence

from llm.initialize import get_default_parser
from llm.initialize import initialize_from_args
from llm.models import bert
from llm.trainers.bert.utils import load_state
from llm.trainers.bert.utils import parse_config

logger = logging.getLogger('model-convert')


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    argv = argv if argv is not None else sys.argv[1:]
    parser = get_default_parser(
        description=(
            'Converts latest pretraining checkpoint to a HuggingFace saved '
            'PreTrainedModel.'
        ),
    )
    parser.add_argument(
        '--model-dir',
        metavar='PATH',
        required=True,
        help='Directory to save model and config file to.',
    )
    args = parser.parse_args(argv)
    config = parse_config(initialize_from_args(args))

    model = bert.from_config(config.BERT_CONFIG)
    step, _, _ = load_state(config, model)

    logger.info(f'Converting model from checkpoint at global step {step}')
    model.save_pretrained(args.model_dir, max_shard_size=None)
    logger.info(f'Saved model to {args.model_dir}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
