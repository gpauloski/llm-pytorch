from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from typing import Any
from typing import Literal

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForPreTraining

from llm.checkpoint import load_checkpoint
from llm.checkpoint import save_checkpoint
from llm.datasets.nvidia import Batch
from llm.datasets.nvidia import sharded_dataset
from llm.loss import BertPretrainingCriterion
from llm.models import bert
from llm.utils import create_summary_writer
from llm.utils import flattened_config


def checkpoint(
    global_step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: colossalai.logging.DistributedLogger,
) -> None:  # pragma: no cover
    if dist.get_rank() == 0:
        save_checkpoint(
            checkpoint_dir=gpc.config.CHECKPOINT_DIR,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            phase=gpc.config.PHASE,
        )
    logger.info(
        f'Saved checkpoint at global step {global_step}',
        ranks=[0],
    )


def load_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: colossalai.logging.DistributedLogger,
) -> tuple[int, int]:  # pragma: no cover
    global_step = 0
    epoch = 0

    os.makedirs(gpc.config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint = load_checkpoint(gpc.config.CHECKPOINT_DIR, map_location='cpu')
    if checkpoint is not None:
        logger.info(f'Loaded checkpoint from {checkpoint.filepath}', ranks=[0])
        model.load_state_dict(checkpoint.model_state_dict)

        if checkpoint.kwargs['phase'] == gpc.config.PHASE:
            logger.info(
                'Checkpoint from current phase. Loading optimizer state',
                ranks=[0],
            )
            if checkpoint.optimizer_state_dict is not None:
                optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                device = (
                    f'cuda:{torch.cuda.current_device()}'
                    if torch.cuda.is_available()
                    else 'cpu'
                )
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            if checkpoint.scheduler_state_dict is not None:
                scheduler.load_state_dict(checkpoint.scheduler_state_dict)
            global_step = checkpoint.global_step
            epoch = checkpoint.kwargs['epoch']
        else:
            logger.info(
                'Checkpoint from new phase. Resetting optimizer state',
                ranks=[0],
            )
    else:
        logger.info('No checkpoint found to resume from.', ranks=[0])

    return global_step, epoch


def log_step(
    logger: colossalai.logger.DistributedLogger,
    *,
    epoch: int,
    global_step: int,
    step_loss: float,
    step_time: float,
    lr: float,
    writer: SummaryWriter | None,
) -> None:  # pragma: no cover
    logger.info(
        f'phase: {gpc.config.PHASE} | epoch: {epoch} | step: {global_step} | '
        f'loss: {step_loss:.3f} | lr: {lr:.2e} | time (s): {step_time:.3f}',
        ranks=[0],
    )
    if writer is not None:
        writer.add_scalar('train/loss', step_loss, global_step)
        writer.add_scalar('train/lr', lr, global_step)
        writer.add_scalar('train/epoch', epoch, global_step)


def get_optimizer_grouped_parameters(
    model: BertForPreTraining,
) -> list[dict[str, Any]]:  # pragma: no cover
    # configure the weight decay for bert models
    params = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    params_decay = [p for n, p in params if not any(v in n for v in no_decay)]
    params_no_decay = [p for n, p in params if any(v in n for v in no_decay)]

    return [
        {'params': params_decay, 'weight_decay': 0.01},
        {'params': params_no_decay, 'weight_decay': 0.0},
    ]


def get_optimizer(
    model: BertForPreTraining,
    name: Literal['lamb', 'adam'],
    lr: float,
) -> torch.optim.Optimizer:  # pragma: no cover
    grouped_params = get_optimizer_grouped_parameters(model)

    if name == 'adam':
        # betas from BERT paper: https://arxiv.org/pdf/1810.04805.pdf
        optimizer = FusedAdam(grouped_params, lr=lr, betas=[0.9, 0.999])
    elif name == 'lamb':
        optimizer = FusedLAMB(grouped_params, lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {name}')

    return optimizer


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    argv = argv if argv is not None else sys.argv[1:]
    parser = colossalai.get_default_parser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)

    if args.debug:
        colossalai.launch(
            config=args.config,
            rank=0,
            world_size=1,
            host='localhost',
            port=29400,
            backend='gloo',
        )
        args.local_rank = 0
    else:
        colossalai.launch_from_torch(config=args.config)

    gradient_accumulation = gpc.config.GLOBAL_BATCH_SIZE / (
        dist.get_world_size() * gpc.config.BATCH_SIZE
    )
    if gradient_accumulation != int(gradient_accumulation):
        raise ValueError(
            'Global batch size must be divisible by the product of the '
            'world size and batch size.',
        )
    gpc.config.update({'gradient_accumulation': int(gradient_accumulation)})

    colossalai.logging.disable_existing_loggers()
    logger = colossalai.logging.get_dist_logger()
    # TODO: https://github.com/hpcaitech/ColossalAI/issues/2109
    # logger.log_to_file(os.path.join(gpc.config.OUTPUT_DIR, 'logs'))

    logger.info(
        f'Launching training from config at {args.config} '
        f'(debug: {args.debug})',
        ranks=[0],
    )

    model = bert.from_config(
        gpc.config.BERT_CONFIG,
        gpc.config.get('GRADIENT_CHECKPOINTING', False),
    )
    optimizer = get_optimizer(model, gpc.config.OPTIMIZER, gpc.config.LR)
    criterion = BertPretrainingCriterion(gpc.config.BERT_CONFIG['vocab_size'])
    scheduler = LinearWarmupLR(
        optimizer,
        total_steps=gpc.config.STEPS,
        warmup_steps=gpc.config.WARMUP_STEPS,
    )

    global_step, epoch = load_state(model, optimizer, scheduler, logger)

    writer = create_summary_writer(
        gpc.config.TENSORBOARD_DIR,
        flattened_config(),
        ['train/loss', 'train/lr', 'train/epoch'],
        purge_step=global_step,
    )

    engine, _, _, scheduler = colossalai.initialize(
        model,
        optimizer,
        criterion=criterion,
        lr_scheduler=scheduler,
        # Pass fake iterable for train_dataloader because setting
        # gradient_accumulation requires initialize to have a dataloader
        train_dataloader=list(range(0, 1024)),
    )

    micro_step = 0
    global_step_timer = colossalai.utils.Timer()
    step_loss = 0.0

    model.train()
    start_time = time.perf_counter()

    while global_step < gpc.config.STEPS:
        dataset = sharded_dataset(
            gpc.config.DATA_DIR,
            gpc.config.BATCH_SIZE,
            seed=gpc.config.seed,
        )
        for batch in dataset:
            micro_step += 1

            if micro_step % gpc.config.gradient_accumulation == 1:
                global_step_timer.start()

            batch = Batch(*[t.cuda() for t in batch])
            engine.zero_grad()
            output = engine(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                token_type_ids=batch.token_type_ids,
            )
            loss = engine.criterion(
                prediction_scores=output.prediction_logits,
                masked_lm_labels=batch.masked_labels,
                seq_relationship_score=output.seq_relationship_logits,
                next_sentence_labels=batch.next_sentence_labels,
            )
            engine.backward(loss)
            engine.step()

            step_loss += loss.float().item()

            if micro_step % gpc.config.gradient_accumulation == 0:
                global_step += 1

                global_step_timer.stop(keep_in_history=True)
                log_step(
                    logger,
                    epoch=epoch,
                    global_step=global_step,
                    step_loss=step_loss / gpc.config.gradient_accumulation,
                    step_time=global_step_timer.get_elapsed_time(),
                    lr=scheduler.get_last_lr()[0],
                    writer=writer,
                )

                if global_step % gpc.config.CHECKPOINT_STEPS == 0:
                    checkpoint(
                        global_step=global_step,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        logger=logger,
                    )

                step_loss = 0.0

            # Wait to step scheduler until after we have logged current LR
            scheduler.step()

            if global_step >= gpc.config.STEPS:
                # Break out of dataset loop, then while loop will also break
                break

        epoch += 1

    writer.close()

    step_time = (
        f'{global_step_timer.get_history_mean():.3f}'
        if global_step_timer.has_history
        else 'N/A'
    )
    logger.info(
        'Training complete ('
        f'total training time (s): {time.perf_counter() - start_time:.3f}, '
        f'avg step time (s): {step_time})',
        ranks=[0],
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
