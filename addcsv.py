import pandas as pd
from pathlib import Path

# ====== 所有路徑、檔名相關設定集中於最上方 ======

# 口音資料根目錄
root_dir = Path('/mnt/d/data/FSR-2025-Hakka-train')

# 清單：要處理的 CSV 檔名（不含副檔名）
csv_names = ['訓練', '測試', '驗證']

# 多種擴增後綴
noise_types = ['noise', 'speech', 'music', 'fast', 'slow']

# 輸入 CSV 路徑樣板
input_csv_template = '/home/mahua0301/FSR-25/data/全部合併/{name}.csv'

# 輸出 CSV 路徑樣板（單一擴增）
output_csv_template = '/mnt/d/FSR25/data/全部合併/{name}_{noise}.csv'
# 輸出 CSV 路徑樣板（合併全部）
output_all_csv_template = '/mnt/d/FSR25/data/全部合併/{name}_all.csv'

# ====== 路徑擴充函數 ======

def transform_path(path_str, noise_type):
    p = Path(path_str)
    try:
        rel = p.relative_to(root_dir)
    except Exception:
        return path_str
    accent = rel.parts[0]
    new_accent = f"{accent}_{noise_type}"
    remainder = rel.parts[1:]
    new_path = root_dir / new_accent / Path(*remainder)
    stem = new_path.stem
    suffix = new_path.suffix
    return str(new_path.with_name(f"{stem}_{noise_type}{suffix}"))

# ====== 主流程 ======

for name in csv_names:
    input_csv = input_csv_template.format(name=name)
    df = pd.read_csv(input_csv)
    original_count = len(df)
    print(f"📄 {name} 原始資料筆數: {original_count}")

    # 儲存全部合併的清單
    dfs_all = [df]

    for noise in noise_types:
        df_aug = df.copy()
        df_aug['檔名'] = df_aug['檔名'].apply(lambda p: transform_path(p, noise))
        dfs_all.append(df_aug)

        output_csv = output_csv_template.format(name=name, noise=noise)
        df_combined = pd.concat([df, df_aug], ignore_index=True)
        print(f"➡️ {name}_{noise} 筆數（原始 + 擴增）: {len(df_combined)}")
        df_combined.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"已輸出：{output_csv}")

    # 儲存全部合併檔案
    df_out = pd.concat(dfs_all, ignore_index=True)
    output_all_csv = output_all_csv_template.format(name=name)
    print(f"🌀 {name}_all 總筆數（含所有擴增類型）: {len(df_out)}")
    df_out.to_csv(output_all_csv, index=False, encoding='utf-8-sig')
    print(f"已輸出：{output_all_csv}\n")
