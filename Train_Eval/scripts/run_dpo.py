#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

# 添加本地src路径到sys.path，优先使用本地修改的alignment包
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 过滤transformers的警告以避免日志格式化错误
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.logging")

import logging
import random

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

# 尝试导入alignment包，如果decontaminate_humaneval不可用则设为None
try:
    from alignment import (
        DataArguments,
        DPOConfig,
        H4ArgumentParser,
        ModelArguments,
        apply_chat_template,
        get_checkpoint,
        get_datasets,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
        get_tokenizer,
        is_adapter_model,
    )
    # 尝试导入decontaminate_humaneval，如果失败则设为None
    try:
        from alignment import decontaminate_humaneval
        DECONTAM_AVAILABLE = True
    except Exception:
        decontaminate_humaneval = None
        DECONTAM_AVAILABLE = False
        print("decontaminate_humaneval is not available. Skipping decontamination.")
except Exception as e:
    print(f"Error importing alignment package: {e}")
    raise
from peft import PeftConfig, PeftModel
from trl import DPOTrainer


logger = logging.getLogger(__name__)


class AnticausalDPOTrainer(DPOTrainer):
    """扩展的DPO训练器，使用预计算的SAS scores进行anticausal regularization"""
    
    def __init__(self, *args, **kwargs):
        # 提取anticausal相关参数
        self.if_anticausal = kwargs.pop('if_anticausal', False)
        self.anticausal_weight = kwargs.pop('anticausal_weight', 0.1)
        
        super().__init__(*args, **kwargs)
        
        if self.if_anticausal:
            logger.info(f"🔬 Anticausal regularization enabled with weight: {self.anticausal_weight}")
            logger.info("📊 Using pre-computed SAS scores for anticausal regularization")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """重写compute_loss方法，在DPO loss基础上加入anticausal regularization"""
        # 调用原始的DPO loss计算
        dpo_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        if self.if_anticausal:
            # 从inputs中获取预计算的SAS scores
            chosen_sas_score = inputs.get("chosen_sas_score", None)
            rejected_sas_score = inputs.get("rejected_sas_score", None)
            
            if chosen_sas_score is not None and rejected_sas_score is not None:
                # 计算anticausal regularization loss
                # 调整后的reward差距：
                # chosen: 原本reward=1，现在变成 1 - chosen_sas_score
                # rejected: 原本reward=0，现在变成 0 - rejected_sas_score
                # 差距 = (1 - chosen_sas_score) - (0 - rejected_sas_score) = (1 - chosen_sas_score) + rejected_sas_score
                anticausal_loss = (1 - chosen_sas_score) + rejected_sas_score
                
                # 组合loss：DPO loss + anticausal regularization
                total_loss = dpo_loss + self.anticausal_weight * anticausal_loss
                
                # 记录loss到日志
                if self.accelerator.is_main_process:
                    self.log({
                        'dpo_loss': dpo_loss.item(),
                        'anticausal_loss': anticausal_loss.item(),
                        'total_loss': total_loss.item(),
                        'chosen_sas_score': chosen_sas_score.item(),
                        'rejected_sas_score': rejected_sas_score.item(),
                        'adjusted_chosen_reward': (1 - chosen_sas_score).item(),
                        'adjusted_rejected_reward': (0 - rejected_sas_score).item()
                    })
                
                if return_outputs:
                    return total_loss, outputs
                else:
                    return total_loss
            else:
                logger.warning("SAS scores not found in inputs, skipping anticausal regularization")
        
        if return_outputs:
            return dpo_loss, outputs
        else:
            return dpo_loss


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
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

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["prompt", "chosen", "rejected"],  # 只保留我们数据集实际有的列
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    ##########################
    # Decontaminate benchmarks
    ##########################
    num_raw_train_samples = len(raw_datasets["train"])
    
    # Only run decontamination if decontaminate_humaneval is available
    if decontaminate_humaneval is not None:
        # Create a wrapper function that uses the correct text_column
        def decontaminate_wrapper(samples):
            return decontaminate_humaneval(samples, text_column="text_chosen")
        
        raw_datasets = raw_datasets.filter(
            decontaminate_wrapper,
            batched=True,
            batch_size=10_000,
            num_proc=1,
            desc="Decontaminating HumanEval samples",
        )
        num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
        logger.info(
            f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
        )
    else:
        logger.info("Skipping decontamination as decontaminate_humaneval is not available.")

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        # 检查数据集是否已经有正确的列名
        if "text_prompt" in raw_datasets[split].column_names:
            raw_datasets[split] = raw_datasets[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )
        # 如果已经是正确的列名，则不需要重命名

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    # 跳过adapter模型检查以避免网络问题
    # if is_adapter_model(model, model_args.model_revision) is True:
    #     logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
    #     peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    #     model_kwargs = dict(
    #         revision=model_args.base_model_revision,
    #         trust_remote_code=model_args.trust_remote_code,
    #         attn_implementation=model_args.attn_implementation,
    #         torch_dtype=torch_dtype,
    #         use_cache=False if training_args.gradient_checkpointing else True,
    #         device_map=get_kbit_device_map() if quantization_config is not None else None,
    #         quantization_config=quantization_config,
    #     )
    #     base_model = AutoModelForCausalLM.from_pretrained(
    #         peft_config.base_model_name_or_path,
    #         **model_kwargs,
    #     )
    #     model = PeftModel.from_pretrained(
    #         base_model,
    #         model_args.model_name_or_path,
    #         revision=model_args.model_revision,
    #     )
    #     model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer
    #########################
    
    # 检查是否启用anticausal regularization
    if_anticausal = getattr(training_args, 'if_anticausal', False)
    anticausal_weight = getattr(training_args, 'anticausal_weight', 0.1)
    
    if if_anticausal:
        logger.info(f"🔬 Anticausal regularization enabled with weight: {anticausal_weight}")
        
        # 使用扩展的DPO训练器
        trainer = AnticausalDPOTrainer(
            model,
            ref_model,
            model_init_kwargs=model_kwargs,
            ref_model_init_kwargs=ref_model_kwargs,
            args=training_args,
            beta=training_args.beta,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            tokenizer=tokenizer,
            max_length=training_args.max_length,
            max_prompt_length=training_args.max_prompt_length,
            peft_config=get_peft_config(model_args),
            loss_type=training_args.loss_type,
            if_anticausal=if_anticausal,
            anticausal_weight=anticausal_weight,
        )
    else:
        # 使用原始的DPO训练器
        trainer = DPOTrainer(
            model,
            ref_model,
            model_init_kwargs=model_kwargs,
            ref_model_init_kwargs=ref_model_kwargs,
            args=training_args,
            beta=training_args.beta,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            tokenizer=tokenizer,
            max_length=training_args.max_length,
            max_prompt_length=training_args.max_prompt_length,
            peft_config=get_peft_config(model_args),
            loss_type=training_args.loss_type,
        )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        # Create model card with correct parameters for trl 0.12.2
        trainer.create_model_card(
            model_name=model_args.model_name_or_path.split("/")[-1] if "/" in model_args.model_name_or_path else model_args.model_name_or_path,
            dataset_name=", ".join(list(data_args.dataset_mixer.keys())),
            tags=["alignment-handbook", "dpo"],
        )
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        # push_to_hub uses the same parameters as create_model_card via **kwargs
        trainer.push_to_hub(
            model_name=model_args.model_name_or_path.split("/")[-1] if "/" in model_args.model_name_or_path else model_args.model_name_or_path,
            dataset_name=", ".join(list(data_args.dataset_mixer.keys())),
            tags=["alignment-handbook", "dpo"],
        )

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
