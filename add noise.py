import os
import random
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

# 口音清單
oris = ["詔安腔", "大埔腔"]
# 噪聲類別清單
noise_labels = ["noise", "speech", "music"]
# SNR 可選擇值
snr_choices = [10, 15, 20]

# 混音處理函數：捕捉當前的 noise_files, out_root, noise_label
def mix_audio(src_path, noise_files, out_root, noise_label):
    audio = AudioSegment.from_wav(src_path)
    combined_noise = AudioSegment.silent(duration=len(audio))
    num_noises = random.randint(2, 3)
    selected = random.sample(noise_files, k=min(num_noises, len(noise_files)))
    snr_db = random.choice(snr_choices)

    for noise_path in selected:
        noise = AudioSegment.from_wav(noise_path)
        # 重複填滿長度
        if len(noise) < len(audio):
            noise = noise * (len(audio) // len(noise) + 1)
        noise = noise[:len(audio)]
        attenuate = audio.dBFS - noise.dBFS - snr_db
        noise = noise + attenuate
        combined_noise = combined_noise + 5
        combined_noise = combined_noise.overlay(noise)


    mixed = audio.overlay(combined_noise)

    rel = os.path.relpath(src_path, start=src_root)
    base = os.path.basename(rel)
    name, ext = os.path.splitext(base)
    new_base = f"{name}_{noise_label}{ext}"
    out_path = os.path.join(out_root, os.path.dirname(rel), new_base)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mixed.export(out_path, format='wav')
    print(f"{rel} 混音完成！SNR={snr_db}dB → 輸出：{new_base}")

# 主流程：對每種口音、每種噪聲類別分別跑一次
for accent in oris:
    src_root = rf"/mnt/d/data/FSR-2025-Hakka-train/{accent}"
    # 收集所有原始檔
    src_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(src_root)
        for f in fs if f.endswith(".wav")
    ]

    for noise_label in noise_labels:
        noise_root = rf"/mnt/d/data/musan/{noise_label}"
        out_root = rf"/mnt/d/data/FSR-2025-Hakka-train/{accent}{noise_label}"
        os.makedirs(out_root, exist_ok=True)

        # 收集當前噪聲類別的所有檔
        noise_files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(noise_root)
            for f in fs if f.endswith(".wav")
        ]

        # 多執行緒混音
        max_workers = max(4, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(mix_audio, src, noise_files, out_root, noise_label)
                for src in src_files
            ]
            # 等待所有完成
            for f in as_completed(futures):
                _ = f.result()

print("✅ 所有口音、所有噪聲類別的混音都完成了！")
