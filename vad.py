import os
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import torchaudio

# 載入 VAD 模型
model = load_silero_vad()

# 資料夾設定
oris = ["詔安腔", "大埔腔"]
base_input_dir = '/mnt/d/data/FSR-2025-Hakka-train(原資料)'
base_output_dir = '/mnt/d/data/FSR-2025-Hakka-train'

for ori in oris:
    input_dir = os.path.join(base_input_dir, ori)
    output_dir = os.path.join(base_output_dir, f"{ori}vad5")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n開始處理：{ori}")
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith('.wav'):
                continue

            fpath = os.path.join(root, fname)
            wav = read_audio(fpath, sampling_rate=16000)

            # 優化參數設置以去除結尾1秒空白
            speech_ts = get_speech_timestamps(
                wav,
                model,
                sampling_rate=16000,
                threshold=0.25,               # 提高閾值以排除結尾低能量空白
                min_silence_duration_ms=100, # 確保0.2秒間隔不被切斷
                min_speech_duration_ms=150,  # 保留短語音片段
                speech_pad_ms=10,            # 極小化前後空白保留
                window_size_samples=512      # 提高檢測精細度
            )

            print(f"處理檔案：{fname}，偵測語音段數：{len(speech_ts)}")

            if not speech_ts:
                print("沒有偵測到語音，跳過。")
                continue

            # 合併所有語音段
            speech_segments = [wav[ts['start']:ts['end']] for ts in speech_ts]
            combined = torch.cat(speech_segments, dim=0)

            # 輸出儲存
            sr = 16000
            rel_dir = os.path.relpath(root, input_dir)
            out_dir_full = os.path.join(output_dir, rel_dir)
            os.makedirs(out_dir_full, exist_ok=True)
            outname = f"{os.path.splitext(fname)[0]}.wav"
            outpath = os.path.join(out_dir_full, outname)
            torchaudio.save(outpath, combined.unsqueeze(0), sr)

            print(f"完成並儲存：{outpath}")