# Note: below is the original copyright associated with this file.
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GPT pretraining CLI.

Script modified from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import pprint
import re
from collections.abc import Sequence
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import default_data_collator
from transformers import get_scheduler

from llm.environment import log_environment
from llm.initialize import initialize as initialize_environment
from llm.timer import Timer
from llm.trainers.gpt.arguments import parse_args
from llm.trainers.gpt.data import get_datasets
from llm.trainers.gpt.data import preprocess_datasets
from llm.trainers.gpt.model import load_model
from llm.trainers.gpt.optimizer import get_optimizer
from llm.trainers.gpt.optimizer import get_preconditioner
from llm.utils import create_summary_writer
from llm.utils import log_step

logger = logging.getLogger('llm.trainers.gpt')

LOG_FMT = (
    'step {step} | epoch: {epoch} | loss: {loss:.3f} | lr: {lr:.2e} | '
    'samples/s: {samples_per_second:.2f} | time (s): {time:.3f}'
)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    logfile = os.path.join(args.output_dir, 'log.txt')
    initialize_environment(
        loglevel=logging.INFO,
        logfile=logfile,
        seed=args.seed,
    )
    log_environment()
    logger.info(pprint.pformat(vars(args), indent=2), extra={'ranks': [0]})

    # Initialize the accelerator. We will let the accelerator handle device
    # placement for us in this example.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    logger.info(accelerator.state, extra={'ranks': [0]})

    raw_datasets = get_datasets(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        validation_split_percentage=args.validation_split_percentage,
        train_file=args.train_file,
        validation_file=args.validation_file,
        keep_linebreaks=not args.no_keep_linebreaks,
    )

    model, tokenizer = load_model(
        config_name=args.config_name,
        model_name_or_path=args.model_name_or_path,
        model_type=args.model_type,
        tokenizer_name=args.tokenizer_name,
        use_slow_tokenizer=args.use_slow_tokenizer,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    lm_datasets = preprocess_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        accelerator=accelerator,
        num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        block_size=args.block_size,
    )

    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets['validation']

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    optimizer = get_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    preconditioner: Any = None
    if args.kfac:
        preconditioner = get_preconditioner(
            model,
            factor_update_steps=args.kfac_factor_update_steps,
            inv_update_steps=args.kfac_inv_update_steps,
            learning_rate=lambda x: optimizer.param_groups[0]['lr'],
            # Ignored because update_factors_in_hook is False
            accumulation_steps=args.gradient_accumulation_steps,
            update_factors_in_hook=False,
            damping=args.kfac_damping,
            factor_decay=args.kfac_factor_decay,
            kl_clip=args.kfac_kl_clip,
            skip_layers=['Embedding', 'lm_head'],
        )
        logger.info(
            f'Training with KFAC:\n{preconditioner}',
            extra={'ranks': [0]},
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps,
    )
    if args.max_train_steps is None:
        args.max_train_steps = (
            args.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps
        * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps
        * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the
    # training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps,
    )
    if overrode_max_train_steps:
        args.max_train_steps = (
            args.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch,
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info('***** Running training *****', extra={'ranks': [0]})
    logger.info(f'Num examples: {len(train_dataset)}', extra={'ranks': [0]})
    logger.info(f'Num Epochs: {args.num_train_epochs}', extra={'ranks': [0]})
    logger.info(
        f'Batch size per device: {args.per_device_train_batch_size}',
        extra={'ranks': [0]},
    )
    logger.info(
        f'Global train batch size: {total_batch_size}',
        extra={'ranks': [0]},
    )
    logger.info(
        f'Gradient accumulation steps: {args.gradient_accumulation_steps}',
        extra={'ranks': [0]},
    )
    logger.info(
        f'Total optimization steps: {args.max_train_steps}',
        extra={'ranks': [0]},
    )

    completed_steps = 0
    starting_epoch = 0
    resume_step = 0

    # Potentially load in the weights and states from a previous save
    args.checkpoint_dir = (
        'checkpoints'
        if args.output_dir is None
        else os.path.join(args.output_dir, 'checkpoints')
    )
    if args.resume_from_checkpoint and os.path.isdir(args.checkpoint_dir):
        # Get the most recent checkpoint
        dir_path = pathlib.Path(args.checkpoint_dir)
        checkpoint_pattern = re.compile(r'^step_(\d+)$')
        checkpoints = {
            match.group(1): str(p)
            for p in dir_path.iterdir()
            if (match := checkpoint_pattern.fullmatch(p.name)) is not None
        }

        if len(checkpoints) > 0:
            checkpoint_step = max(int(key) for key in checkpoints)
            resume_path = dir_path / f'step_{checkpoint_step}'

            logger.info(
                f'Resumed from checkpoint: {resume_path}',
                extra={'ranks': [0]},
            )
            accelerator.load_state(resume_path)

            kfac_path = resume_path / 'kfac.pt'
            if kfac_path.is_file() and preconditioner is not None:
                kfac_state_dict = torch.load(kfac_path)
                preconditioner.load_state_dict(kfac_state_dict)
                logger.info(
                    'Loaded KFAC state from checkpoint',
                    extra={'ranks': [0]},
                )

            # Multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = checkpoint_step * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    writer: SummaryWriter | None = None
    completed_steps = starting_epoch * num_update_steps_per_epoch
    global_step_timer = Timer()
    step_loss = 0.0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        completed_steps += 1
                    continue

            if writer is None:
                writer = create_summary_writer(
                    os.path.join(args.output_dir, 'tensorboard'),
                    vars(args),
                    [
                        'train/loss',
                        'train/lr',
                        'train/epoch',
                        'train/samples_per_second',
                    ],
                    purge_step=completed_steps,
                )

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                step_loss += loss.float().item()
                accelerator.backward(loss)
                if (
                    preconditioner is not None
                    and step % args.gradient_accumulation_steps == 0
                ):
                    preconditioner.step()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step
            # behind the scenes
            if accelerator.sync_gradients:
                completed_steps += 1
                step_time = global_step_timer.lap()
                log_step(
                    logger,
                    step=completed_steps,
                    epoch=epoch,
                    loss=step_loss / args.gradient_accumulation_steps,
                    samples_per_second=total_batch_size / step_time,
                    time=step_time,
                    lr=lr_scheduler.get_last_lr()[0],
                    fmt_str=LOG_FMT,
                    writer=writer,
                    tensorboard_prefix='train',
                    skip_tensorboard=['phase', 'time'],
                )
                step_loss = 0.0

            if (
                isinstance(checkpointing_steps, int)
                and accelerator.sync_gradients
                and completed_steps % checkpointing_steps == 0
            ):
                output_dir = f'step_{completed_steps }'
                output_dir = os.path.join(args.checkpoint_dir, output_dir)
                accelerator.save_state(output_dir)
                if preconditioner is not None and accelerator.is_main_process:
                    torch.save(
                        preconditioner.state_dict(),
                        os.path.join(output_dir, 'kfac.pt'),
                    )

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for _step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size),
                ),
            )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')

        logger.info(
            f'Validation | epoch: {epoch} | eval_loss: {eval_loss:.3f} | '
            f'perplexity: {perplexity:.3f}',
            extra={'ranks': [0]},
        )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            with open(
                os.path.join(args.output_dir, 'all_results.json'),
                'w',
            ) as f:
                json.dump({'perplexity': perplexity}, f)

    if writer is not None:
        writer.close()

    step_time = global_step_timer.get_history_mean()
    # Stop timer after getting avg step time because it will add a history
    # entry from between end of last step and now
    global_step_timer.stop()
    total_time = global_step_timer.get_history_sum()

    logger.info(
        'Training complete ('
        f'total training time (s): {total_time:.3f}, '
        f'avg step time (s): {step_time:.3f})',
        extra={'ranks': [0]},
    )

    return 0
