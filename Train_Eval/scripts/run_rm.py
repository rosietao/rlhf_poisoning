#!/usr/bin/env python
# coding=utf-8

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
import random
import warnings

import torch
import transformers
from transformers import AutoModelForSequenceClassification, set_seed

# Silence noisy transformers warnings similar to run_dpo
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.logging")

try:
    from alignment import (
        DataArguments,
        H4ArgumentParser,
        ModelArguments,
        apply_chat_template,
        get_checkpoint,
        get_datasets,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
        get_tokenizer,
    )
except Exception as e:
    print(f"Error importing alignment package: {e}")
    raise

from trl import RewardTrainer, RewardConfig
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def main():
    # Parse arguments using existing config classes; RewardConfig from TRL
    # Use extended RMConfig to accept sas_k and sas_threshold
    from alignment import RMConfig
    parser = H4ArgumentParser((ModelArguments, DataArguments, RMConfig))
    model_args, data_args, training_args = parser.parse()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Auto-append sas_k, sas_threshold, and dataset suffix to output_dir for traceability
    try:
        sas_k_str = str(getattr(training_args, "sas_k", ""))
        sas_thr_str = str(getattr(training_args, "sas_threshold", ""))
        
        # Extract dataset suffix from dataset_mixer
        dataset_suffix = ""
        if hasattr(data_args, 'dataset_mixer') and data_args.dataset_mixer:
            # Get the first dataset path from dataset_mixer
            dataset_path = list(data_args.dataset_mixer.keys())[0]
            # Extract suffix after "rlhf_sampled_hf_"
            if "rlhf_sampled_hf_" in dataset_path:
                dataset_suffix = "_" + dataset_path.split("rlhf_sampled_hf_")[-1]
        
        # Build complete suffix
        suffix_parts = []
        if sas_k_str != "" and sas_thr_str != "":
            suffix_parts.append(f"_k{sas_k_str}-thr{sas_thr_str}")
        if dataset_suffix:
            suffix_parts.append(dataset_suffix)
        
        if suffix_parts:
            suffix = "".join(suffix_parts)
            if not str(training_args.output_dir).endswith(suffix):
                training_args.output_dir = f"{training_args.output_dir}{suffix}"
    except Exception:
        pass

    # Ensure max_length defaults to a valid int for TRL filtering/tokenization
    if getattr(training_args, "max_length", None) is None:
        training_args.max_length = 2048
    if getattr(training_args, "max_prompt_length", None) is None:
        training_args.max_prompt_length = training_args.max_length

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Resume from checkpoint if exists
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Seed
    set_seed(training_args.seed)

    # Load datasets as in run_dpo; keep same columns
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=[
            "prompt",
            "chosen",
            "rejected",
            "chosen_sas_score",
            "rejected_sas_score",
        ],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    column_names = list(raw_datasets["train"].features)

    # Tokenizer
    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    # Apply chat template only if inputs are chat message lists; skip for plain strings
    try:
        sample_example = raw_datasets["train"][0] if len(raw_datasets["train"]) > 0 else None
    except Exception:
        sample_example = None

    should_apply_template = False
    if sample_example is not None:
        prompt_val = sample_example.get("prompt", None)
        # Expect a list of messages (dicts) for chat; strings mean already formatted
        if isinstance(prompt_val, list):
            should_apply_template = True

    if should_apply_template:
        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "rm",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Formatting comparisons with prompt template for RM",
        )

    # Rename if the dataset still has text_* columns
    for split in ["train", "test"]:
        if "text_prompt" in raw_datasets[split].column_names:
            raw_datasets[split] = raw_datasets[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )

    # Log a few samples
    for index in random.sample(range(len(raw_datasets["train"])), min(3, len(raw_datasets["train"]))):
        logger.info(f"Prompt sample {index}:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index}:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index}:\n\n{raw_datasets['train'][index]['rejected']}")

    # Model dtype and quantization
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    # Reward model: sequence classification with 1 label (scalar reward)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        problem_type="regression",
        num_labels=1,
    )

    # Instantiate model explicitly (older TRL RewardTrainer may not accept model_init_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Custom SASTrainer with SAS-aware margin loss
    class SASTrainer(RewardTrainer):
        def __init__(self, *args, sas_k: float = 1.0, sas_threshold: float = 1e9, **kwargs):
            super().__init__(*args, **kwargs)
            self.sas_k = sas_k
            self.sas_threshold = sas_threshold

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Expect chosen/rejected batch like RewardTrainer; plus sas scores
            outputs = model(**inputs["chosen_inputs"])
            r_chosen = outputs.logits.squeeze(-1)
            outputs = model(**inputs["rejected_inputs"])
            r_rejected = outputs.logits.squeeze(-1)

            # Default BT log loss term on reward differences
            reward_diff = r_chosen - r_rejected

            # SAS margin
            chosen_sas = inputs.get("chosen_sas", None)
            rejected_sas = inputs.get("rejected_sas", None)
            if chosen_sas is not None and rejected_sas is not None:
                sas_delta = chosen_sas - rejected_sas
                # Debug prints: run only a few times to avoid spam
                if not hasattr(self, "_dbg_count"):
                    self._dbg_count = 0
                dbg_do_print = self._dbg_count < 5
                # only rank0 prints to avoid clutter
                try:
                    is_main_rank = int(os.environ.get("RANK", "0")) == 0
                except Exception:
                    is_main_rank = True

                # Thresholding: if delta > threshold, set delta = 0 (one-sided)
                zero_ratio = None
                if self.sas_threshold is not None and self.sas_threshold > 0:
                    mask = sas_delta > self.sas_threshold
                    if dbg_do_print:
                        try:
                            zero_ratio = mask.float().mean().item()
                        except Exception:
                            zero_ratio = None
                    sas_delta = torch.where(mask, torch.zeros_like(sas_delta), sas_delta)
                # Curriculum: epoch 1 use k=0, from epoch>=1.0 use configured k
                current_epoch = getattr(self.state, "epoch", 0.0)
                effective_k = 0.0 if (current_epoch is None or current_epoch < 1) else self.sas_k
                margin = effective_k * sas_delta
                reward_diff = reward_diff - margin

                if dbg_do_print and is_main_rank:
                    try:
                        print(
                            {
                                "dbg_epoch": (float(current_epoch) if current_epoch is not None else None),
                                "dbg_global_step": int(getattr(self.state, "global_step", -1)),
                                "dbg_sas_k": float(self.sas_k),
                                "dbg_threshold": (float(self.sas_threshold) if self.sas_threshold is not None else None),
                                "dbg_effective_k": float(effective_k),
                                "dbg_delta_mean": float(sas_delta.mean().item()),
                                "dbg_delta_std": float(sas_delta.std().item()),
                                "dbg_delta_min": float(sas_delta.min().item()),
                                "dbg_delta_max": float(sas_delta.max().item()),
                                "dbg_zero_ratio": (float(zero_ratio) if zero_ratio is not None else None),
                            }
                        , flush=True)
                    except Exception:
                        pass
                    self._dbg_count += 1
            else:
                # If SAS not present, print once to make it obvious
                if not hasattr(self, "_dbg_no_sas"):
                    try:
                        is_main_rank = int(os.environ.get("RANK", "0")) == 0
                    except Exception:
                        is_main_rank = True
                    if is_main_rank:
                        try:
                            print({"dbg_no_sas": True, "keys": list(inputs.keys())}, flush=True)
                        except Exception:
                            pass
                    self._dbg_no_sas = True

            loss = F.binary_cross_entropy_with_logits(reward_diff, torch.ones_like(reward_diff))

            if return_outputs:
                return loss, {"reward_diff": reward_diff}
            return loss

    # Data collator to include SAS tensors
    def sas_data_collator(features):
        # Build concatenated texts: prompt + response for chosen/rejected
        prompt_texts = [f.get("prompt", "") for f in features]
        chosen_texts = [f.get("chosen", "") for f in features]
        rejected_texts = [f.get("rejected", "") for f in features]

        sep = "\n\n"
        concat_chosen = [p + sep + c for p, c in zip(prompt_texts, chosen_texts)]
        concat_rejected = [p + sep + r for p, r in zip(prompt_texts, rejected_texts)]

        max_len = effective_max_length
        chosen_tok = tokenizer(
            concat_chosen,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        rejected_tok = tokenizer(
            concat_rejected,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        batch = {
            "chosen_inputs": {"input_ids": chosen_tok.input_ids, "attention_mask": chosen_tok.attention_mask},
            "rejected_inputs": {"input_ids": rejected_tok.input_ids, "attention_mask": rejected_tok.attention_mask},
        }

        # Collect SAS; fail fast if missing when training expects SAS
        has_chosen = "chosen_sas_score" in features[0]
        has_rejected = "rejected_sas_score" in features[0]
        if not (has_chosen and has_rejected):
            raise ValueError(
                f"SAS scores missing in dataset batch. Found keys: {list(features[0].keys())}. "
                "Expected 'chosen_sas_score' and 'rejected_sas_score'. Please point RM config to a dataset that contains SAS scores or disable SAS usage."
            )
        batch["chosen_sas"] = torch.tensor([f.get("chosen_sas_score", 0.0) for f in features], dtype=torch.float32)
        batch["rejected_sas"] = torch.tensor([f.get("rejected_sas_score", 0.0) for f in features], dtype=torch.float32)
        return batch

    # Read extra hyperparameters from training_args if present; fallback defaults
    sas_k = getattr(training_args, "sas_k", 1.0)
    sas_threshold = getattr(training_args, "sas_threshold", 0.0)

    # Ensure TRL sees a concrete max_length; avoid double-spec conflict by clearing args.max_length
    effective_max_length = getattr(training_args, "max_length", None) or 2048
    try:
        setattr(training_args, "max_length", None)
    except Exception:
        pass

    trainer = SASTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets.get("test", None),
        tokenizer=tokenizer,
        data_collator=sas_data_collator,
        peft_config=get_peft_config(model_args),
        sas_k=sas_k,
        sas_threshold=sas_threshold,
        max_length=effective_max_length,
    )

    # Train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"]) if "train" in raw_datasets else 0
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # Save model
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Evaluate
    if training_args.do_eval and raw_datasets.get("test") is not None:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"]) if "test" in raw_datasets else 0
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
