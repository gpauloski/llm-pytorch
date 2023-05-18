from __future__ import annotations

import logging

import transformers
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import CONFIG_MAPPING
from transformers import MODEL_MAPPING

logger = logging.getLogger('llm.trainers.gpt')

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def load_model(
    *,
    config_name: str | None = None,
    model_name_or_path: str | None = None,
    model_type: str | None = None,
    tokenizer_name: str | None = None,
    use_slow_tokenizer: bool = False,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Load pretrained model and tokenizer.

    In distributed training, the `.from_pretrained` methods guarantee that only
    one local process can concurrently download model & vocab.
    """
    if config_name is not None:
        config = AutoConfig.from_pretrained(config_name)
    elif model_name_or_path is not None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_type]()
        logger.warning(
            'You are instantiating a new config instance from scratch.',
            extra={'ranks': [0]},
        )

    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=not use_slow_tokenizer,
        )
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=not use_slow_tokenizer,
        )
    else:
        raise ValueError(
            'You are instantiating a new tokenizer from scratch. This is not '
            'supported by this script. You can do it from another script, '
            'save it, and load it from here, using --tokenizer_name.',
        )

    if model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool('.ckpt' in model_name_or_path),
            config=config,
        )
    else:
        logger.info('Training new model from scratch', extra={'ranks': [0]})
        model = AutoModelForCausalLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If
    # you are creating a model from scratch on a small vocab and want a smaller
    # embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
