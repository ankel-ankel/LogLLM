# LogLLM (DeBERTa-v3-Large + Llama-3.1-8B)

Fork of the original project: `https://github.com/guanwei49/LogLLM`

## 1. Changes

- Changed the models BERT + Llama setup to `DeBERTa-v3-Large` + `Llama-3.1-8B`.
- Replaced the BERT-specific encoder loading with `AutoTokenizer` + `AutoModel`.
- Adjusted encoder input/output handling so the same LogLLM pipeline still works with the new encoder.
- Updated the encoder fine-tuning (LoRA) hookup for the new encoder structure.

## 2. Environment Setup

- Python version: `Python 3.14`

- Install packages:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install transformers datasets peft accelerate bitsandbytes safetensors
pip install scikit-learn
pip install tqdm
pip install sentencepiece protobuf
```

## 3. Open-source Models [DeBERTa-v3-Large](https://huggingface.co/microsoft/deberta-v3-large) and [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)

Place model folders under `models/`:

```text
models/
|-- deberta-v3-large/
|   |-- config.json
|   |-- tokenizer_config.json
|   |-- tokenizer.json / spm.model / vocab files
|   |-- model.safetensors (or model shards)
|   `-- ...
`-- Meta-Llama-3.1-8B/
    |-- config.json
    |-- generation_config.json
    |-- model-00001-of-00004.safetensors
    |-- model-00002-of-00004.safetensors
    |-- model-00003-of-00004.safetensors
    |-- model-00004-of-00004.safetensors
    |-- model.safetensors.index.json
    |-- tokenizer.json
    |-- tokenizer_config.json
    `-- ...
```

## 4. Data

For dataset preparation, refer to the upstream repo:

- `https://github.com/guanwei49/LogLLM`

Put the Dataset in`data/`:

```text
data/
|-- BGL/
|-- HDFS_v1/
|-- Liberty/
`-- Thunderbird/
```
