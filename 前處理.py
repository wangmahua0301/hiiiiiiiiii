from datasets import load_dataset, Audio
import torch
torch.backends.cudnn.benchmark = True
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

# 定義快取目錄
processed_cache_dir = "/home/mahua0301/FSR25/訓練資料/all-large"

# 載入數據集
data_files = {
    "train": "/home/mahua0301/FSR25/data/全部合併/訓練_all.csv",
    "test": "/home/mahua0301/FSR25/data/全部合併/測試_all.csv",
    "val": "/home/mahua0301/FSR25/data/全部合併/驗證_all.csv"
}
common_voice = load_dataset("csv", data_files=data_files)
print(common_voice)

# 初始化 Whisper 模型的特征提取器和分詞器
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="zh", task="transcribe")

# 測試分詞器
input_str = common_voice["train"][0]["客語漢字"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

# 初始化 Whisper 處理器
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="zh", task="transcribe")

# 將音頻數據轉換為 16kHz 採樣率
common_voice = common_voice.cast_column("檔名", Audio(sampling_rate=16000))
print(common_voice["train"][0])

# 定義數據預處理函數
def prepare_dataset(batch):
    audio = batch["檔名"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["客語漢字"]).input_ids
    return batch

# 執行數據預處理
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=3)

# 將處理後的數據集保存到指定目錄
common_voice.save_to_disk(processed_cache_dir)
print(f"Processed dataset saved to {processed_cache_dir}")