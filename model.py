import inspect
import os.path

import numpy as np
import peft
import torch
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import logging as hf_logging


def merge_data(data):
    merged_data = []
    start_positions = []
    current_position = 0

    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)
    return merged_data, start_positions


def stack_and_pad_right(tensors):
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))
        padded_tensors.append(padded_tensor)

        padding_mask = torch.cat(
            [
                torch.ones(tensor.shape[0], dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )
        padding_masks.append(padding_mask)

    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks


def stack_and_pad_left(tensors):
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0))
        padded_tensors.append(padded_tensor)

        padding_mask = torch.cat(
            [
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(tensor.shape[0], dtype=torch.long),
            ]
        )
        padding_masks.append(padding_mask)

    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks


LLM_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


class LogLLM(nn.Module):
    def __init__(
        self,
        Bert_path,
        Llama_path,
        ft_path=None,
        is_train_mode=True,
        device=torch.device("cuda:0"),
        max_content_len=128,
        max_seq_len=128,
    ):
        super().__init__()
        self.max_content_len = max_content_len
        self.max_seq_len = max_seq_len
        self.device = device

        device_index = 0 if device.index is None else device.index
        self._single_device_map = {"": device_index}

        self.Llama_tokenizer = AutoTokenizer.from_pretrained(Llama_path, padding_side="right")
        if self.Llama_tokenizer.pad_token is None:
            self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token

        self.Llama_model = AutoModelForCausalLM.from_pretrained(
            Llama_path,
            quantization_config=LLM_BNB_CONFIG,
            low_cpu_mem_usage=True,
            device_map=self._single_device_map,
        )

        # Use AutoTokenizer/AutoModel so the encoder can be swapped (BERT/DeBERTa/...).
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(Bert_path)
        encoder_load_kwargs = {
            "low_cpu_mem_usage": True,
            "device_map": self._single_device_map,
        }
        if self.device.type == "cuda":
            encoder_load_kwargs["torch_dtype"] = torch.float16
        # microsoft/deberta-v3-large checkpoint includes MLM head weights; loading AutoModel is correct,
        # but transformers prints an "UNEXPECTED" load report for the unused head. Suppress only this load.
        prev_hf_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.encoder_model = AutoModel.from_pretrained(Bert_path, **encoder_load_kwargs)
        finally:
            hf_logging.set_verbosity(prev_hf_verbosity)

        self.encoder_forward_keys = set(inspect.signature(self.encoder_model.forward).parameters.keys())

        self.projector = nn.Linear(
            self.encoder_model.config.hidden_size,
            self.Llama_model.config.hidden_size,
            device=device,
        )

        self.instruc_tokens = self.Llama_tokenizer(
            [
                "Below is a sequence of system log messages:",
                ". Is this sequence normal or anomalous? \n",
            ],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        if ft_path is not None:
            print(f"Loading peft model from {ft_path}.")
            Llama_ft_path = os.path.join(ft_path, "Llama_ft")
            Bert_ft_path = os.path.join(ft_path, "Bert_ft")  # legacy name kept to avoid wider changes
            projector_path = os.path.join(ft_path, "projector.pt")
            self.Llama_model = PeftModel.from_pretrained(
                self.Llama_model,
                Llama_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            self.encoder_model = PeftModel.from_pretrained(
                self.encoder_model,
                Bert_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            self.projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=True))
        else:
            print("Creating peft model.")
            encoder_peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=4,
                lora_alpha=32,
                lora_dropout=0.01,
                target_modules=self._infer_encoder_lora_targets(),
            )
            self.encoder_model = get_peft_model(self.encoder_model, encoder_peft_config)

            Llama_peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.Llama_model = get_peft_model(self.Llama_model, Llama_peft_config)

        self._refresh_legacy_aliases()

    def _refresh_legacy_aliases(self):
        # Keep legacy attribute names so train.py/eval.py/custom scripts continue to work.
        self.Bert_tokenizer = self.encoder_tokenizer
        self.Bert_model = self.encoder_model

    def _infer_encoder_lora_targets(self):
        model_type = getattr(self.encoder_model.config, "model_type", "")
        if model_type == "deberta-v2":
            return ["query_proj", "key_proj", "value_proj"]
        if model_type in {"bert", "roberta"}:
            return ["query", "key", "value"]
        # Fallback for unknown encoders; PEFT may infer targets for some supported models.
        return None

    def _prepare_encoder_inputs(self, inputs):
        filtered = {k: v for k, v in inputs.items() if k in self.encoder_forward_keys}
        return {k: v.to(self.device) for k, v in filtered.items()}

    def _mean_pool(self, last_hidden_state, attention_mask):
        attn = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * attn).sum(dim=1)
        denom = attn.sum(dim=1).clamp(min=1)
        return summed / denom

    def _encode_messages(self, inputs):
        encoder_inputs = self._prepare_encoder_inputs(inputs)
        encoder_outputs = self.encoder_model(**encoder_inputs)

        pooled = getattr(encoder_outputs, "pooler_output", None)
        if pooled is None:
            pooled = self._mean_pool(encoder_outputs.last_hidden_state, encoder_inputs["attention_mask"])
        return pooled.float()

    def _get_llm_input_embeddings(self):
        return self.Llama_model.get_input_embeddings()

    def _llm_embed(self, input_ids):
        embed_layer = self._get_llm_input_embeddings()
        return embed_layer(input_ids.to(embed_layer.weight.device))

    def _project_to_llm_dim(self, encoder_embeddings):
        projected = self.projector(encoder_embeddings.to(self.projector.weight.dtype))
        llm_dtype = self._get_llm_input_embeddings().weight.dtype
        return projected.to(dtype=llm_dtype)

    def _split_seq_embeddings(self, message_embeddings, seq_positions):
        if torch.is_tensor(seq_positions):
            split_positions = seq_positions.detach().cpu().tolist()
        else:
            split_positions = list(seq_positions)
        return torch.tensor_split(message_embeddings, split_positions)

    def save_ft_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        Llama_ft_path = os.path.join(path, "Llama_ft")
        Bert_ft_path = os.path.join(path, "Bert_ft")  # legacy folder name kept intentionally
        projector_path = os.path.join(path, "projector.pt")
        self.Llama_model.save_pretrained(Llama_ft_path, safe_serialization=True)
        self.encoder_model.save_pretrained(Bert_ft_path, safe_serialization=True)
        torch.save(self.projector.state_dict(), projector_path)

    def set_train_only_projector(self):
        for _, param in self.projector.named_parameters():
            param.requires_grad = True
        for _, param in self.encoder_model.named_parameters():
            param.requires_grad = False
        for _, param in self.Llama_model.named_parameters():
            param.requires_grad = False

    def set_train_only_Llama(self):
        for _, param in self.projector.named_parameters():
            param.requires_grad = False
        for _, param in self.encoder_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = "lora" in name

    def set_train_projectorAndBert(self):
        for _, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.encoder_model.named_parameters():
            param.requires_grad = "lora" in name
        for _, param in self.Llama_model.named_parameters():
            param.requires_grad = False

    def set_finetuning_all(self):
        for _, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.encoder_model.named_parameters():
            param.requires_grad = "lora" in name
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = "lora" in name

    def train_helper(self, inputs, seq_positions, labels):
        """
        :param inputs: tokenized log messages for encoder. Sequences are concatenated.
        :param seq_positions:
        :param labels: np.array with values in ['anomalous', 'normal']
        :return: logits on label tokens, target token ids
        """
        batch_size = len(labels)

        message_embeddings = self._encode_messages(inputs)
        message_embeddings = self._project_to_llm_dim(message_embeddings)
        seq_embeddings = self._split_seq_embeddings(message_embeddings, seq_positions)

        prefix = "The sequence is "
        max_len = max(len(s) for s in labels) + len(prefix)
        labels = np.char.add(np.char.add(prefix, labels.astype(f"U{max_len}")), ".")
        answer_tokens = self.Llama_tokenizer(list(labels), padding=True, return_tensors="pt").to(self.device)

        target_tokens_ids = torch.cat(
            [
                answer_tokens["input_ids"][:, 1:],
                torch.full((batch_size, 1), self.Llama_tokenizer.eos_token_id, device=self.device),
            ],
            dim=-1,
        )
        target_tokens_atts = answer_tokens["attention_mask"].bool()

        answer_tokens_ids = answer_tokens["input_ids"][:, 1:]
        answer_tokens_atts = answer_tokens["attention_mask"].bool()[:, 1:]

        instruc_embeddings = self._llm_embed(self.instruc_tokens["input_ids"])
        answer_embeddings = self._llm_embed(answer_tokens_ids)

        ins1 = instruc_embeddings[0][self.instruc_tokens["attention_mask"][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens["attention_mask"][1].bool()][1:]

        embeddings = []
        target_lens = []
        for seq_embedding, answer_embedding, answer_tokens_att in zip(
            seq_embeddings, answer_embeddings, answer_tokens_atts
        ):
            full_prompt_embedding = torch.cat([ins1, seq_embedding, ins2, answer_embedding[answer_tokens_att]])
            target_lens.append(int(answer_tokens_att.sum().item()))
            embeddings.append(full_prompt_embedding)

        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        attention_mask = attention_mask.to(self.device)
        label_mask = attention_mask.clone()
        for i in range(label_mask.shape[0]):
            label_mask[i, : -target_lens[i] - 1] = 0
        label_mask = label_mask.bool()

        llm_output = self.Llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        return llm_output[label_mask], target_tokens_ids[target_tokens_atts]

    def forward(self, inputs, seq_positions):
        """
        :param inputs: tokenized log messages for encoder. Sequences are concatenated.
        :param seq_positions:
        :return: generated answer token ids
        """
        batch_size = len(seq_positions) + 1

        message_embeddings = self._encode_messages(inputs)
        message_embeddings = self._project_to_llm_dim(message_embeddings)
        seq_embeddings = self._split_seq_embeddings(message_embeddings, seq_positions)

        prefix = "The sequence is"
        answer_prefix_tokens = self.Llama_tokenizer(prefix, padding=True, return_tensors="pt")["input_ids"][
            0, 1:
        ].to(self.device)

        instruc_embeddings = self._llm_embed(self.instruc_tokens["input_ids"])
        answer_prefix_tokens_embeddings = self._llm_embed(answer_prefix_tokens)

        ins1 = instruc_embeddings[0][self.instruc_tokens["attention_mask"][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens["attention_mask"][1].bool()][1:]

        prompt_embeddings = []
        for seq_embedding in seq_embeddings:
            prompt_embedding = torch.cat([ins1, seq_embedding, ins2, answer_prefix_tokens_embeddings])
            prompt_embeddings.append(prompt_embedding)

        inputs_embeds, attention_mask = stack_and_pad_left(prompt_embeddings)
        attention_mask = attention_mask.to(self.device)

        pad_token_id = self.Llama_tokenizer.pad_token_id
        eos_token_id = self.Llama_tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id[0]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(self.device) if eos_token_id is not None else None

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        this_peer_finished = False
        answer = []
        past_key_values = None

        while not this_peer_finished:
            if past_key_values is None:
                outputs = self.Llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            else:
                outputs = self.Llama_model(
                    inputs_embeds=next_tokens_embeddings[:, None, :],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            answer.append(next_tokens)

            next_tokens_embeddings = self._llm_embed(next_tokens)
            attention_mask = torch.cat([attention_mask, unfinished_sequences[:, None]], dim=1)

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            if len(answer) > 5:
                this_peer_finished = True

        return torch.stack(answer, dim=1)
