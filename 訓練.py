from datasets import load_from_disk
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor, WhisperTokenizer

# 初始化 processor 和 tokenizer
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="zh", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="zh", task="transcribe")

# 載入預處理後的數據集
processed_cache_dir = "/home/mahua0301/FSR25/訓練資料/all-large"
common_voice = load_from_disk(processed_cache_dir)
# 初始化 Whisper 模型
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model.generation_config.language = "zh"
model.generation_config.task = "transcribe"
model.gradient_checkpointing_disable()
model.config.use_cache = False
model.generation_config.forced_decoder_ids = None

#設定
from math import ceil
train_batch_size = 1
grad_acc_steps = 1
train_steps_per_epoch = ceil(len(common_voice["train"]) / (train_batch_size * grad_acc_steps))
EVAL_EVERY_EPOCHS = 2
train_epoch = 2
eval_steps = train_steps_per_epoch * EVAL_EVERY_EPOCHS

# 定義數據整理器
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# 定義評估指標
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 將 -100 padding 恢復成 pad_token_id 以便正確解碼
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # 解碼預測與標註
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 🔍 顯示前幾筆預測 vs 標註
    for i in range(min(5, len(pred_str))):
        print(f"預測: {pred_str[i]}")
        print(f"標註: {label_str[i]}")

    # 計算 WER / CER
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

# 設置訓練參數
training_args = Seq2SeqTrainingArguments(
    output_dir="./model/large",
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=train_epoch,
    max_steps=-1,
    gradient_checkpointing=False,
    fp16=True,
    # 改成每 N 步驟評估與儲存一次
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=eval_steps,
    save_steps=eval_steps,

    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,

    logging_steps=500,
    report_to=["tensorboard"],

    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
)

# 初始化訓練器
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,
)

# 開始訓練
trainer.train()