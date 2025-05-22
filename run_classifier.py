#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse
import glob
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# TensorBoard writer fallback
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

from transformers import (
    WEIGHTS_NAME,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from sklearn.metrics import precision_score, recall_score  # <-- added
from utils import compute_metrics, convert_examples_to_features, output_modes, processors

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "roberta": (
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizer,
    )
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, optimizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    scaler = GradScaler() if args.fp16 else None

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")
    scheduler_last = os.path.join(checkpoint_last, "scheduler.pt")
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss = 0.0
    logging_loss = 0.0

    model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)
    model.train()

    for epoch_idx, _ in enumerate(train_iterator):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                "labels": batch[3],
            }

            if args.fp16:
                with autocast(device_type="cuda", enabled=True):
                    outputs = model(**inputs)
                    loss = outputs[0]
                loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(**inputs)
                loss = outputs[0]
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, checkpoint=str(global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar(f"eval_{key}", value, global_step)
                            logger.info("    %s = %s", key, value)
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        # Save checkpoint at end of epoch
        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            results = evaluate(args, model, tokenizer, checkpoint=str(args.start_epoch + epoch_idx))
            os.makedirs(checkpoint_last, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(checkpoint_last)
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_last, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(checkpoint_last, "scheduler.pt"))
            with open(os.path.join(checkpoint_last, "idx_file.txt"), "w", encoding="utf-8") as idxf:
                idxf.write(str(args.start_epoch + epoch_idx) + "\n")

            if results.get("acc", 0) > getattr(train, "best_acc", 0):
                train.best_acc = results["acc"]
                best_dir = os.path.join(args.output_dir, "checkpoint-best")
                os.makedirs(best_dir, exist_ok=True)
                model_to_save.save_pretrained(best_dir)
                torch.save(optimizer.state_dict(), os.path.join(best_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(best_dir, "scheduler.pt"))

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode="dev"):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if mode == "dev":
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, ttype="dev")
        else:
            eval_dataset, instances = load_and_cache_examples(args, eval_task, tokenizer, ttype="test")

        os.makedirs(eval_output_dir, exist_ok=True)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        logger.info(f"***** Running evaluation {prefix} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_loss = 0.0
        preds = None
        out_label_ids = None

        for batch in tqdm(
            eval_dataloader,
            desc="Evaluating",
            disable=False,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                    "labels": batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / len(eval_dataloader)
        if args.output_mode == "classification":
            preds_label = np.argmax(preds, axis=1)

        # compute core metrics
        result = compute_metrics(eval_task, preds_label, out_label_ids)

        # --- added: compute precision & recall ----------------------------------
        result['precision'] = precision_score(out_label_ids, preds_label, zero_division=0)
        result['recall']    = recall_score(out_label_ids, preds_label, zero_division=0)
        # -----------------------------------------------------------------------

        results.update(result)

        if mode == "dev":
            with open(os.path.join(eval_output_dir, "eval_results.txt"), "a+") as writer:
                writer.write(f"evaluate {checkpoint}\n")
                for key, value in sorted(result.items()):
                    writer.write(f"{key} = {value:.6f}\n")
        else:
            os.makedirs(os.path.dirname(args.test_result_dir), exist_ok=True)
            with open(args.test_result_dir, "w") as writer:
                for i, logit in enumerate(preds.tolist()):
                    inst = "<CODESPLIT>".join(instances[i])
                    writer.write(f"{inst}<CODESPLIT>{'<CODESPLIT>'.join(map(str, logit))}\n")

    return results


def load_and_cache_examples(args, task, tokenizer, ttype="train"):
    processor = processors[task]()
    output_mode = output_modes[task]
    base = {
        "train": args.train_file,
        "dev": args.dev_file,
        "test": args.test_file,
    }[ttype].rsplit(".", 1)[0]
    cache_file = os.path.join(
        args.data_dir,
        f"cached_{ttype}_{base}_{args.model_name_or_path.split('/')[-1]}_{args.max_seq_length}_{task}",
    )

    try:
        logger.info("Loading features from %s", cache_file)
        features = torch.load(cache_file)
        if ttype == "test":
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)
    except:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if ttype == "train":
            examples = processor.get_train_examples(args.data_dir, args.train_file)
        elif ttype == "dev":
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)
        else:
            examples, instances = processor.get_test_examples(args.data_dir, args.test_file)

        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 1,
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            torch.save(features, cache_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return (dataset, instances) if ttype == "test" else dataset


def main():
    parser = argparse.ArgumentParser()

    ### Required parameters ###
    parser.add_argument("--data_dir", type=str, required=True, help="Input data dir")
    parser.add_argument(
        "--model_type", type=str, required=True, choices=list(MODEL_CLASSES.keys())
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Local path to pretrained CodeBERT"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="codesearch",
        choices=list(processors.keys()),
        help="Task name",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to save checkpoints"
    )

    ### Other parameters ###
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="Max input sequence length after tokenization",
    )
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU for training"
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU for evaluation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Steps to accumulate before backward/update",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Initial learning rate")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Training epochs")
    parser.add_argument(
        "--max_steps", default=-1, type=int, help="If > 0: total number of training steps"
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup steps")
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--save_steps", default=50, type=int)
    parser.add_argument("--eval_all_checkpoints", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", default="O1", type=str)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--server_ip", default="", type=str)
    parser.add_argument("--server_port", default="", type=str)
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="Dev file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--pred_model_dir", default=None, type=str, help="Model dir for prediction")
    parser.add_argument("--test_result_dir", default="test_results.tsv", type=str, help="Where to write test results")
    parser.add_argument(
        "--dataloader_num_workers",
        default=4,
        type=int,
        help="Number of subprocesses for DataLoader",
    )

    args = parser.parse_args()

    # start from scratch (no checkpoint resume)
    args.start_epoch = 0
    args.start_step = 0

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError(f"Task not found: {args.task_name}")
    processors_cls = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=len(processors_cls.get_labels()),
        finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        local_files_only=True,
    )

    # Freeze everything except the classification head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype="train")
        train(args, train_dataset, model, tokenizer, optimizer)

    if args.do_eval:
        evaluate(args, model, tokenizer)

    if args.do_predict:
        model = model_class.from_pretrained(args.pred_model_dir)
        model.to(args.device)
        evaluate(args, model, tokenizer, mode="test")


if __name__ == "__main__":
    main()
