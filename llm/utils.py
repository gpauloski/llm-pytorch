from __future__ import annotations


def gradient_accumulation_steps(
    global_batch_size: int,
    local_batch_size: int,
    world_size: int,
) -> int:
    effective_batch = local_batch_size * world_size

    if global_batch_size % effective_batch != 0:
        raise ValueError(
            f'The global batch size ({global_batch_size}) must be evenly '
            'divisible by the product of the local batch size '
            f'({local_batch_size}) and the world size ({world_size}).',
        )

    return global_batch_size // effective_batch
