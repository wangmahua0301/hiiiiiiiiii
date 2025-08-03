#%%
import os
import pandas as pd
import random

input_dir = "/mnt/d/data/FSR-2025-Hakka-train"
output_dir = "/home/mahua0301/FSR25/data"
OUTPUT_DIR_1 = "/home/mahua0301/FSR25/data/男女合併"
OUTPUT_DIR_2 = "/home/mahua0301/FSR25/data/全部合併"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(OUTPUT_DIR_1, exist_ok=True)
os.makedirs(OUTPUT_DIR_2, exist_ok=True)

DIALECT_MAP = {
    "DF": "大埔腔", "DM": "大埔腔",
    "ZF": "詔安腔", "ZM": "詔安腔",
}

def extract_speaker_id(filename):
    return filename.split("_")[0]

def build_abs_path(wav_name):
    dialect_dir = DIALECT_MAP[wav_name[:2]]
    speaker = extract_speaker_id(wav_name)
    return os.path.join(input_dir, dialect_dir, speaker, wav_name)

def filter_df(df):
    mask = df["備註"].fillna("").str.contains("正確讀音")
    return df[~mask]

def count_speakers(csv_file):
    df = pd.read_csv(csv_file)
    df = filter_df(df)
    df["檔名"] = df["檔名"].apply(build_abs_path)
    speakers = set(df["檔名"].apply(lambda p: extract_speaker_id(os.path.basename(p))))
    return {"speakers": speakers, "rows": len(df)}

def split_csv_by_speaker(csv_path, output_dir, n_val=3, n_test=3, seed=42):
    df = pd.read_csv(csv_path)
    df = filter_df(df)
    df["檔名"] = df["檔名"].apply(build_abs_path)
    df["speaker_id"] = df["檔名"].apply(lambda p: extract_speaker_id(os.path.basename(p)))

    speakers = df["speaker_id"].unique().tolist()
    random.seed(seed)
    selected = random.sample(speakers, n_val + n_test)
    val_set = set(selected[:n_val])
    test_set = set(selected[n_val:])

    df_val = df[df["speaker_id"].isin(val_set)]
    df_test = df[df["speaker_id"].isin(test_set)]
    df_train = df[~df["speaker_id"].isin(val_set | test_set)]

    base = os.path.splitext(os.path.basename(csv_path))[0]

    train_path = os.path.join(output_dir, f"{base}.csv")
    val_path = os.path.join(output_dir, f"{base.replace('訓練', '驗證', 1)}.csv")
    test_path = os.path.join(output_dir, f"{base.replace('訓練', '測試', 1)}.csv")

    df_train.drop(columns=["speaker_id"]).to_csv(train_path, index=False)
    df_val.drop(columns=["speaker_id"]).to_csv(val_path, index=False)
    df_test.drop(columns=["speaker_id"]).to_csv(test_path, index=False)

    print(f"【{base}】訓練筆數: {len(df_train)}")
    print(f"【{base}】驗證筆數: {len(df_val)}")
    print(f"【{base}】測試筆數: {len(df_test)}")

    print(f"【{base}】驗證: {sorted(val_set)}")
    print(f"【{base}】測試: {sorted(test_set)}")

    return len(df_train), len(df_val), len(df_test)


def main():
    csv_paths = [
        f"{input_dir}/訓練_DF_大埔腔_女_edit.csv",
        f"{input_dir}/訓練_DM_大埔腔_男_edit.csv",
        f"{input_dir}/訓練_ZF_詔安腔_女_edit.csv",
        f"{input_dir}/訓練_ZM_詔安腔_男_edit.csv",
    ]

    total_speakers = set()
    total_rows = 0

    for path in csv_paths:
        result = count_speakers(path)
        total_speakers |= result["speakers"]
        total_rows += result["rows"]

    for csv_path in csv_paths:
        split_csv_by_speaker(csv_path, output_dir)

if __name__ == "__main__":
    main()
#%%

merge_pairs_1 = [
    ([
        "data/訓練_DF_大埔腔_女_edit.csv",
        "data/訓練_DM_大埔腔_男_edit.csv"
    ], "訓練_大埔腔.csv"),
    ([
        "data/測試_DF_大埔腔_女_edit.csv",
        "data/測試_DM_大埔腔_男_edit.csv"
    ], "測試_大埔腔.csv"),
    ([
        "data/驗證_DF_大埔腔_女_edit.csv",
        "data/驗證_DM_大埔腔_男_edit.csv"
    ], "驗證_大埔腔.csv"),
    ([
        "data/訓練_ZF_詔安腔_女_edit.csv",
        "data/訓練_ZM_詔安腔_男_edit.csv"
    ], "訓練_詔安腔.csv"),
    ([
        "data/測試_ZF_詔安腔_女_edit.csv",
        "data/測試_ZM_詔安腔_男_edit.csv"
    ], "測試_詔安腔.csv"),
    ([
        "data/驗證_ZF_詔安腔_女_edit.csv",
        "data/驗證_ZM_詔安腔_男_edit.csv"
    ], "驗證_詔安腔.csv"),
]

print("\n男女合併")
for inputs, output_name in merge_pairs_1:
    dfs = [pd.read_csv(f) for f in inputs]
    merged = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR_1, output_name)
    merged.to_csv(output_path, index=False)
    print(f"{output_path}（共 {len(merged)} 筆）")

merge_pairs_2 = [
    ([
        os.path.join(OUTPUT_DIR_1, "訓練_大埔腔.csv"),
        os.path.join(OUTPUT_DIR_1, "訓練_詔安腔.csv")
    ], "訓練.csv"),
    ([
        os.path.join(OUTPUT_DIR_1, "測試_大埔腔.csv"),
        os.path.join(OUTPUT_DIR_1, "測試_詔安腔.csv")
    ], "測試.csv"),
    ([
        os.path.join(OUTPUT_DIR_1, "驗證_大埔腔.csv"),
        os.path.join(OUTPUT_DIR_1, "驗證_詔安腔.csv")
    ], "驗證.csv"),
]

print("\n全部合併")
for inputs, output_name in merge_pairs_2:
    dfs = [pd.read_csv(f) for f in inputs]
    merged = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(OUTPUT_DIR_2, output_name)
    merged.to_csv(output_path, index=False)
    print(f"{output_path}（共 {len(merged)} 筆）")
