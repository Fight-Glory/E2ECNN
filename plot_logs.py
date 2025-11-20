"""基于 CSV 训练日志的可视化脚本。

用法示例：
    python plot_logs.py                           # 自动扫描项目 recon/ 目录下的日志
    python plot_logs.py logs/run1.csv logs/run2.csv --output-dir figures --smooth-window 3
    python plot_logs.py logs_dir/ --train-val-overlay

脚本将自动识别实验名称并输出 loss / PSNR / SSIM 等对比曲线。
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 常见字段名称映射到统一列名，便于兼容不同日志格式
COLUMN_ALIASES: Dict[str, str] = {
    "exp_name": "exp_name",
    "experiment": "exp_name",
    "experiment_name": "exp_name",
    "run": "exp_name",
    "epoch": "epoch",
    "phase": "phase",
    "split": "phase",
    "loss": "loss",
    "train_loss": "loss",
    "val_loss": "loss",
    "psnr": "psnr",
    "ssim": "ssim",
    "learning_rate": "learning_rate",
    "lr": "learning_rate",
    "time_sec": "time_sec",
    "duration": "time_sec",
    "time": "time_sec",
}

# 默认使用的图像 DPI
DEFAULT_DPI = 200


def gather_csv_files(inputs: Iterable[str]) -> List[Path]:
    """收集用户传入的文件或目录中的所有 CSV 路径。"""
    csv_files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            # 修改：改为递归扫描，便于直接输入 recon/ 目录（其中包含时间戳子目录）
            csv_files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            csv_files.append(path)
    return csv_files


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 的列名统一到约定字段，便于后续处理。"""
    rename_map = {}
    for col in df.columns:
        clean_col = col.strip().lower()
        if clean_col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[clean_col]
    df = df.rename(columns=rename_map)
    return df


def ensure_exp_name(df: pd.DataFrame, fallback_name: str) -> pd.DataFrame:
    """确保存在 exp_name 列，如果缺失则使用文件名填充。"""
    if "exp_name" not in df.columns:
        df["exp_name"] = fallback_name
    else:
        df["exp_name"] = df["exp_name"].fillna(fallback_name)
    return df


def load_logs(file_paths: Iterable[Path]) -> pd.DataFrame:
    """读取多个 CSV 日志并合并。"""
    frames: List[pd.DataFrame] = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 无法读取 {path}: {exc}")
            continue
        df = normalize_columns(df)
        df = ensure_exp_name(df, path.stem)
        # 标准化 phase 文本，缺失则记为 unknown
        if "phase" in df.columns:
            df["phase"] = df["phase"].astype(str).str.lower().str.strip()
        else:
            df["phase"] = "unknown"
        # epoch 列尽量转为整数
        if "epoch" in df.columns:
            df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
        frames.append(df)
    if not frames:
        raise ValueError("未能读取到任何有效的 CSV 日志")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    """对序列做简单滑动平均平滑。"""
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def plot_metric_over_epochs(
    df: pd.DataFrame,
    metric: str,
    phase: str,
    output_path: Path,
    smooth_window: int = 1,
    dpi: int = DEFAULT_DPI,
) -> None:
    """绘制指定 phase 下的指标随 epoch 变化对比图。"""
    if metric not in df.columns:
        print(f"[INFO] 数据缺少 {metric} 列，跳过 {metric} 曲线绘制。")
        return

    phase_df = df[df["phase"] == phase].copy()
    if phase_df.empty:
        print(f"[INFO] 没有 phase='{phase}' 的记录，跳过 {metric} 曲线绘制。")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))

    for exp_name, sub_df in phase_df.groupby("exp_name"):
        sub_df = sub_df.sort_values("epoch")
        epochs = sub_df["epoch"]
        values = smooth_series(sub_df[metric], smooth_window)
        plt.plot(epochs, values, label=str(exp_name))

    plt.title(f"{metric.upper()} ({phase}) 对比")
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"[OK] 保存图像到 {output_path}")


def plot_train_val_overlay(
    df: pd.DataFrame,
    metric: str,
    exp_name: str,
    output_path: Path,
    smooth_window: int = 1,
    dpi: int = DEFAULT_DPI,
) -> None:
    """为单个实验绘制 train/val 指标对比图。"""
    if metric not in df.columns:
        return
    exp_df = df[df["exp_name"] == exp_name]
    train_df = exp_df[exp_df["phase"] == "train"].sort_values("epoch")
    val_df = exp_df[exp_df["phase"] == "val"].sort_values("epoch")
    if train_df.empty or val_df.empty:
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))

    plt.plot(train_df["epoch"], smooth_series(train_df[metric], smooth_window), label="train")
    plt.plot(val_df["epoch"], smooth_series(val_df[metric], smooth_window), label="val")

    plt.title(f"{exp_name}: {metric.upper()} train vs val")
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"[OK] 保存图像到 {output_path}")


def summarize_best_scores(df: pd.DataFrame) -> None:
    """打印每个实验在验证集的最佳 PSNR/SSIM（若存在）。"""
    val_df = df[df["phase"] == "val"]
    if val_df.empty:
        print("[INFO] 没有验证集记录，跳过最佳指标统计。")
        return

    for exp_name, sub_df in val_df.groupby("exp_name"):
        msg_parts = [f"实验 {exp_name}:"]
        if "psnr" in sub_df.columns:
            idx = sub_df["psnr"].idxmax()
            max_psnr = sub_df.loc[idx, "psnr"]
            ep = sub_df.loc[idx, "epoch"]
            msg_parts.append(f"最佳 PSNR={max_psnr:.3f} (epoch={ep})")
        if "ssim" in sub_df.columns:
            idx = sub_df["ssim"].idxmax()
            max_ssim = sub_df.loc[idx, "ssim"]
            ep = sub_df.loc[idx, "epoch"]
            msg_parts.append(f"最佳 SSIM={max_ssim:.4f} (epoch={ep})")
        print(" ".join(msg_parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="训练日志可视化工具")
    parser.add_argument(
        "inputs",
        nargs="*",  # 修改：允许为空，默认扫描 recon/ 目录
        help="一个或多个 CSV 文件或目录路径；为空时自动扫描项目根目录下的 recon/",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="输出图像目录，默认 figures/",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="滑动平均窗口大小，用于平滑曲线（>=1）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="输出图像 DPI，默认 200",
    )
    parser.add_argument(
        "--train-val-overlay",
        action="store_true",
        help="为每个实验额外绘制 train/val 对比曲线（若同时存在两类记录）",
    )
    args = parser.parse_args()

    # 修改：若未传入路径，自动扫描项目根目录的 recon/ 目录（包含子目录）
    input_list = args.inputs if args.inputs else ["recon"]

    csv_files = gather_csv_files(input_list)
    if not csv_files:
        raise SystemExit("未找到任何 CSV 文件，请检查输入路径或 recon/ 目录是否存在。")

    print(f"[INFO] 读取 {len(csv_files)} 个日志文件：")
    for path in csv_files:
        print(f"  - {path}")

    df = load_logs(csv_files)
    output_dir = Path(args.output_dir)

    # 绘制验证集指标对比
    plot_metric_over_epochs(
        df,
        metric="loss",
        phase="val",
        output_path=output_dir / "loss_val_compare.png",
        smooth_window=args.smooth_window,
        dpi=args.dpi,
    )
    plot_metric_over_epochs(
        df,
        metric="psnr",
        phase="val",
        output_path=output_dir / "psnr_val_compare.png",
        smooth_window=args.smooth_window,
        dpi=args.dpi,
    )
    plot_metric_over_epochs(
        df,
        metric="ssim",
        phase="val",
        output_path=output_dir / "ssim_val_compare.png",
        smooth_window=args.smooth_window,
        dpi=args.dpi,
    )

    # 若需要，绘制单个实验的 train/val 对比
    if args.train_val_overlay:
        for exp_name in df["exp_name"].unique():
            plot_train_val_overlay(
                df,
                metric="loss",
                exp_name=exp_name,
                output_path=output_dir / f"loss_train_val_{exp_name}.png",
                smooth_window=args.smooth_window,
                dpi=args.dpi,
            )

    summarize_best_scores(df)


if __name__ == "__main__":
    main()
