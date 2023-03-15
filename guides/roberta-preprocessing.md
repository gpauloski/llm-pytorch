# RoBERTa Pre-training Pre-processing Guide

This guide walks through downloading, formatting, and encoding the Wiki and
Books corpora for pretraining RoBERTa.

This guide assumes you have installed the `llm` packages and its dependencies
as described in the [README](../README.md).

Note: using both the Wiki and Books corpora is not necessary, and you can
skip one or the other. This instructions will also work for your own text
corpora---just skip the download step.

## 0. Install Extras

The `llm.preprocess` module has a few extra requirements so reinstall the
`llm` package with the `preprocess` extra.
```
$ pip install -e -U .[preprocess]
```

## 1. Download

Create a `datasets/` directory. This directory will contain all of the
files produced.
```
$ mkdir datasets/
```

Download the datasets.
```
$ python -m llm.preprocess.download --dataset wikipedia --output datasets/downloaded/
$ python -m llm.preprocess.download --dataset bookcorpus --output datasets/downloaded/
```
This will result in two files: `datasets/downloaded/wikipedia-{date}.en.txt`
and `datasets/downloaded/bookcorpus.txt`.
Each of these files has the format one one sentence per line with documents
separated by blank lines.

## 2. Shard the Files

Next we shard the files to make them easier to work with. For small dataset this
may not be needed.

```
$ python -m llm.preprocess.shard --input datasets/downloaded/*.txt --output datasets/sharded/wikibooks/ --size 250MB
```

Now we have a set of sharded files in `datasets/sharded/wikibooks/` that are
each approximately 250 MB. The format of these files is still one sentence
per line with documents separated by blank lines.

## 3. Build the Vocab

Now we build the vocab. BPE and wordpiece tokenizers are supported as well
as cased/uncased. This example creates an uncased wordpiece vocab will 50,000
tokens.

```
$ python -m llm.preprocess.vocab \
      --input datasets/sharded/wikibooks \
      --output datasets/vocabs/wikibooks-50k-vocab.txt \
      --size 50000 \
      --tokenizers wordpiece \
```

## 4. Encode the Shards

Now we can encode the shards using our vocabulary.

```
$ python -m llm.preprocess.roberta \
      --input datasets/sharded/wikibooks/* \
      --output datasets/encoded/wikibooks/ \
      --vocab datasets/vocabs/wikibooks-50k-vocab.txt \
      --tokenizer wordpiece \
      --max-seq-len 512 \
      --processes 4
```

Encoding a 250MB shard can take around 16GB of RAM so adjust the number of
processes (parallel workers encoding a shard) as needed to fit your shard size
and system memory.

This produces a set of encoded HDF5 files in `datasets/encoded/wikibooks`
with each encoded file corresponding to a shard. Each encoded file contains
the `input_ids`, `attention_masks`, and `special_tokens_masks` attributes
which are each numpy arrays of shape `(samples, max_seq_len)`.

In contrast to BERT encoding, the samples are not pre masked and next sentence
prediction is not done. In other words, masking must be done at runtime and
each sample represents contiguous sentences draw from the same document until
the max sequence length is reached.
