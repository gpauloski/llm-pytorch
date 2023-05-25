"""GPT trainer argument parser."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from transformers import SchedulerType

from llm.kfac import add_kfac_options
from llm.trainers.gpt.model import MODEL_TYPES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Train a transformers model for causal language modeling.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help='The name of the dataset to use (via the datasets library).',
    )
    parser.add_argument(
        '--dataset_config_name',
        type=str,
        default=None,
        help=(
            'The configuration name of the dataset to use '
            '(via the datasets library).'
        ),
    )
    parser.add_argument(
        '--train_file',
        type=str,
        default=None,
        help='A csv or a json file containing the training data.',
    )
    parser.add_argument(
        '--validation_file',
        type=str,
        default=None,
        help='A csv or a json file containing the validation data.',
    )
    parser.add_argument(
        '--validation_split_percentage',
        default=5,
        help=(
            'The percentage of the train set used as validation set in case '
            "there's no validation split"
        ),
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help=(
            'Path to pretrained model or model identifier from '
            'huggingface.co/models.'
        ),
        required=False,
    )
    parser.add_argument(
        '--config_name',
        type=str,
        default=None,
        help='Pretrained config name or path if not the same as model_name',
    )
    parser.add_argument(
        '--tokenizer_name',
        type=str,
        default=None,
        help='Pretrained tokenizer name or path if not the same as model_name',
    )
    parser.add_argument(
        '--use_slow_tokenizer',
        action='store_true',
        help=(
            'Use a slow tokenizer (not backed by the 🤗 Tokenizers library).'
        ),
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=8,
        help='Batch size (per device) for the training dataloader.',
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=8,
        help='Batch size (per device) for the evaluation dataloader.',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help=(
            'Initial learning rate (after the potential warmup period) to use.'
        ),
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay to use.',
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=3,
        help='Total number of training epochs to perform.',
    )
    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=None,
        help=(
            'Total number of training steps to perform. '
            'If provided, overrides num_train_epochs.'
        ),
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=(
            'Number of updates steps to accumulate before performing a '
            'backward/update pass.'
        ),
    )
    parser.add_argument(
        '--lr_scheduler_type',
        type=SchedulerType,
        default='linear',
        help='The scheduler type to use.',
        choices=[
            'linear',
            'cosine',
            'cosine_with_restarts',
            'polynomial',
            'constant',
            'constant_with_warmup',
        ],
    )
    parser.add_argument(
        '--num_warmup_steps',
        type=int,
        default=0,
        help='Number of steps for the warmup in the lr scheduler.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the final model.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='A seed for reproducible training.',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        help='Model type to use if training from scratch.',
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=None,
        help=(
            'Optional input sequence length after tokenization. The training '
            'dataset will be truncated in block of this size for training. '
            'Default to the model max input length for single sentence inputs '
            '(take into account special tokens).'
        ),
    )
    parser.add_argument(
        '--preprocessing_num_workers',
        type=int,
        default=None,
        help='The number of processes to use for the preprocessing.',
    )
    parser.add_argument(
        '--overwrite_cache',
        action='store_true',
        help='Overwrite the cached training and evaluation sets',
    )
    parser.add_argument(
        '--no_keep_linebreaks',
        action='store_true',
        help='Do not keep line breaks when using TXT files.',
    )
    parser.add_argument(
        '--checkpointing_steps',
        type=str,
        default=None,
        help=(
            'Whether the various states should be saved at the end of every '
            "n steps, or 'epoch' for each epoch."
        ),
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='If the training should continue from a checkpoint folder.',
    )
    parser.add_argument(
        '--low_cpu_mem_usage',
        action='store_true',
        help=(
            'It is an option to create the model as an empty shell, then '
            'only materialize its parameters when the pretrained weights are '
            'loaded. If passed, LLM loading time and RAM consumption will be '
            'benefited.'
        ),
    )
    add_kfac_options(parser)
    args = parser.parse_args(argv)

    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError(
            'Need either a dataset name or a training/validation file.',
        )
    if args.train_file is not None:
        extension = args.train_file.split('.')[-1]
        if extension not in ['csv', 'json', 'txt']:
            raise ValueError('train_file should be a csv, json or txt file.')
    if args.validation_file is not None:
        extension = args.validation_file.split('.')[-1]
        if extension not in ['csv', 'json', 'txt']:
            raise ValueError(
                'validation_file should be a csv, json or txt file.',
            )

    return args
