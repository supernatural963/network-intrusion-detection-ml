import gc
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def extract_labels_from_filename(filename: str):
    """استخراج برچسب ترافیک بر اساس نام فایل استاندارد"""
    name = filename.lower()
    if "benign" in name:
        return "benign", "benign"
    elif "botnet" in name:
        return "attack", "botnet"
    elif "bruteforce" in name:
        return "attack", "bruteforce"
    elif "ddos" in name:
        return "attack", "ddos"
    elif "dos" in name:
        return "attack", "dos"
    elif "infiltration" in name:
        return "attack", "infiltration"
    elif "portscan" in name:
        return "attack", "portscan"
    elif "webattacks" in name:
        return "attack", "web"
    else:
        return "unknown", "unknown"

def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """کاهش مصرف رم با تبدیل داده‌های ۶۴ بیتی به ۳۲ بیتی"""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)
    return df

def clean_inf_and_nan(df: pd.DataFrame) -> pd.DataFrame:
    """پاک‌سازی مقادیر بی‌نهایت و نامعتبر پکت‌های شبکه"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    return df

def load_and_merge_dataset(data_dir: Path):
    """پایپ‌لاین پردازش جریانی و ترکیب فایل‌های پارکت"""
    parquet_files = sorted(list(data_dir.glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"هیچ فایلی در مسیر {data_dir} یافت نشد!")

    print(f"\n[+] تعداد {len(parquet_files)} فایل Parquet شناسایی شد. آغاز بارگذاری...")
    
    dfs = []
    total_raw_memory = 0.0

    for file_path in parquet_files:
        print(f"  |-- در حال پردازش: {file_path.name}")
        df_chunk = pd.read_parquet(file_path, engine="pyarrow")
        total_raw_memory += df_chunk.memory_usage(deep=True).sum() / (1024 ** 2)
        
        label, attack_type = extract_labels_from_filename(file_path.name)
        df_chunk["label"] = label
        df_chunk["attack_type"] = attack_type
        
        df_chunk = downcast_dataframe(df_chunk)
        df_chunk = clean_inf_and_nan(df_chunk)
        
        dfs.append(df_chunk)
        del df_chunk
        gc.collect()

    print("\n[+] در حال تجمیع نهایی دیتاست در حافظه...")
    merged_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    optimized_memory = merged_df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    print("\n" + "=" * 60)
    print("           گزارش ارزیابی سلامت و بهینه‌سازی داده")
    print("=" * 60)
    print(f"ابعاد نهایی ماتریس داده (Shape) : {merged_df.shape}")
    print(f"تعداد کل رکوردهای شبکه          : {len(merged_df):,}")
    print(f"مصرف تخمینی حافظه اولیه (خام)   : {total_raw_memory:.2f} MB")
    print(f"مصرف بهینه‌شده پس از Downcast    : {optimized_memory:.2f} MB")
    print(f"میزان صرفه‌جویی در رم سیستم     : {((total_raw_memory - optimized_memory) / total_raw_memory) * 100:.1f}%")
    print("-" * 60)
    print("توزیع کلاس‌های اصلی (Binary Labels):")
    print(merged_df["label"].value_counts().to_string())
    print("-" * 60)
    
    num_cols = merged_df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(merged_df[num_cols].values).sum()
    nan_count = merged_df[num_cols].isna().sum().sum()
    print(f"تعداد مقادیر Inf باقیمانده      : {inf_count} [OK]")
    print(f"تعداد مقادیر NaN باقیمانده      : {nan_count} [OK]")
    print("=" * 60)
    print("[SUCCESS] ادغام و پیش‌پردازش با موفقیت کامل شد.\n")
    
    return merged_df

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_ROOT / "data"
    load_and_merge_dataset(DATA_PATH)