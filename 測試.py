#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# ===== 使用者設定 =====
# 1. 你的 fine‑tuned 模型資料夾（含 config.json, pytorch_model.bin）
MODEL_DIR       = "/home/mahua0301/FSR25/model/all4/checkpoint-15932"
# 2. 處理後資料集路徑（由 common_voice.save_to_disk() 存出）
PROCESSED_DIR   = "/home/mahua0301/FSR25/訓練資料/ori"
# 3. 輸出結果資料夾與檔名
OUTPUT_DIR      = "./result"
OUTPUT_FILE     = "whisper_all4-2_evalori.txt"
# 4. Batch size／要顯示的示例筆數
BATCH_SIZE      = 32
NUM_EXAMPLES    = 5
# =====================

# 自訂 collator，專門對 speech seq2seq 做 padding
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # pad input_features
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        # pad labels
        label_list = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_list, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # remove potential leading BOS-only row
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.cpu().numpy()
    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.cpu().numpy()

    # restore padding token for -100
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.batch_decode(label_ids, skip_special_tokens=True)

    # 顯示前幾筆
    for i in range(min(NUM_EXAMPLES, len(pred_strs))):
        print(f"[{i+1:2d}] PRED ➜ {pred_strs[i]}")
        print(f"      REF ➜ {label_strs[i]}")
        print("-"*40)

    wer = 100 * wer_metric.compute(predictions=pred_strs, references=label_strs)
    cer = 100 * cer_metric.compute(predictions=pred_strs, references=label_strs)
    return {"wer": wer, "cer": cer}

if __name__ == "__main__":
    # 設備與快取清理
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # 1. 載入 processor（feature_extractor + tokenizer）
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="zh", task="transcribe")

    # 2. 載入 fine‑tuned 模型
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    # 若有自訂 generation config，一併載入
    try:
        gen_cfg = GenerationConfig.from_pretrained(MODEL_DIR)
        model.generation_config = gen_cfg
    except:
        pass
    model.to(device).eval()

    # 3. 載入評分工具
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # 4. 載入已處理好的資料集
    ds = load_from_disk(PROCESSED_DIR)
    # 假設存成了 train/test/val split，這裡我們評估 val
    val_ds = ds["val"]
    print(f"Loaded processed validation set, total samples: {len(val_ds)}\n")

    # 5. 準備 collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 6. 建立 Seq2SeqTrainer（僅用於 predict + compute_metrics）
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        do_train=False,
        dataloader_num_workers=8,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )

    # 7. 預測並計算指標
    print("Running prediction...\n")
    preds, labels, metrics = trainer.predict(val_ds)
    print(f"\n=== Overall Metrics ===")
    print(f"WER: {metrics['test_wer']:.2f}%")
    print(f"CER: {metrics['test_cer']:.2f}%")

    # 8. 將所有結果寫入檔案
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"WER: {metrics['test_wer']:.2f}%  CER: {metrics['test_cer']:.2f}%\n")
        f.write("="*40 + "\n")
        # 把所有 pred/ref 寫入
        pred_strs  = processor.batch_decode(preds, skip_special_tokens=True)
        label_ids  = labels.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_strs = processor.batch_decode(label_ids, skip_special_tokens=True)
        for p, r in zip(pred_strs, label_strs):
            f.write(f"PRED: {p}\nREF : {r}\n" + ("-"*40) + "\n")
    print(f"\nFull detailed results saved to: {out_path}")
