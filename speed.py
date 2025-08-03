import os
import random
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed


root_data_dir = "/mnt/d/data/FSR-2025-Hakka-train"
oris = ["詔安腔", "大埔腔"]
slow_rates = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
fast_rates = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
speed_types = [("slow", slow_rates), ("fast", fast_rates)]


def change_speed(sound, speed=1.0):
    new_frame_rate = int(sound.frame_rate * speed)
    altered = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
    return altered.set_frame_rate(sound.frame_rate)

def perturb_and_save(src_path, src_root, out_root, label, rate_choices):
    audio = AudioSegment.from_wav(src_path)
    speed = random.choice(rate_choices)
    perturbed = change_speed(audio, speed)
    
    # 維持資料夾樹狀結構，僅檔名變動
    rel = os.path.relpath(src_path, start=src_root)
    base = os.path.basename(rel)
    name, ext = os.path.splitext(base)
    new_base = f"{name}_{label}{ext}"
    out_path = os.path.join(out_root, os.path.dirname(rel), new_base)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    perturbed.export(out_path, format="wav")
    print(f"{rel} → {label} 完成，輸出至：{new_base}")

for accent in oris:
    src_root = os.path.join(root_data_dir, accent)
    src_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(src_root)
        for f in fs if f.endswith(".wav")
    ]

    for label, rate_list in speed_types:
        out_root = os.path.join(root_data_dir, f"{accent}_{label}")
        os.makedirs(out_root, exist_ok=True)
        max_workers = max(4, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    perturb_and_save,
                    src, src_root, out_root, label, rate_list
                )
                for src in src_files
            ]
            for f in as_completed(futures):
                _ = f.result()

print("處理完成")
