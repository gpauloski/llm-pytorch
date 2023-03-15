<style>
.md-typeset h2, h3, h4 {
  font-weight: 400;
  font-family: var(--md-code-font-family);
}

.md-typeset h2 {
  border-bottom-style: solid;
  border-color: var(--md-default-fg-color--lighter);
  border-width: 2px;
}

.md-typeset h3, h4 {
  border-bottom-style: dashed;
  border-color: var(--md-default-fg-color--lighter);
  border-width: 1px;
}
</style>

# CLI Reference

This page provides documentation for our command line tools.

!!! Warning

    The usage examples show the executable module the CLI belongs to.
    To run the CLI, you must execute the module using the Python interpreter.
    E.g.,
    ```bash
    $ python -m llm.preprocess.download --help
    ```

!!! Note

    This list is not exhaustive. In particular, the training scripts
    provided in the `llm.trainers` module are not listed here.

::: mkdocs-click
    :module: llm.preprocess.download
    :command: cli
    :prog_name: llm.preprocess.download
    :depth: 1
    :list_subcommands: True
    :style: table

::: mkdocs-click
    :module: llm.preprocess.roberta
    :command: cli
    :prog_name: llm.preprocess.roberta
    :depth: 1
    :list_subcommands: True
    :style: table

::: mkdocs-click
    :module: llm.preprocess.shard
    :command: cli
    :prog_name: llm.preprocess.shard
    :depth: 1
    :list_subcommands: True
    :style: table

::: mkdocs-click
    :module: llm.preprocess.vocab
    :command: cli
    :prog_name: llm.preprocess.vocab
    :depth: 1
    :list_subcommands: True
    :style: table
