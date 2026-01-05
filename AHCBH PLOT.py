# -*- coding: utf-8 -*-
"""
Read & merge AH/RH wide tables from two Excel files, then plot:
- Heatmap: Absolute Humidity (g/m^3) vs time-height (pixel blocks)
- Right axis: CBH (m a.g.l.) from parsed MWR CBH .ASC files (1-min processed)
"""
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
def safe_ts(s):
    """把 '2024-08-10 1' 这类脏字符串安全解析成 Timestamp"""
    s = str(s).strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        s = m.group(1)
    return pd.to_datetime(s, errors="raise")

# =========================
# 0) 参数区（按你图的范围）
# =========================
# ---- AH/RH Excel sources ----
EXCEL_FILES = [
    r"E:\Beijing2024\MWR\ahrh\04-06\ah&rh_merged.xlsx",
    r"E:\Beijing2024\MWR\ahrh\04-07\ah&rh_merged.xlsx",
]
SHEET_AH = "ah_data"
SHEET_RH = "rh_data"

# ---- Plot window ----
T0 = "2024-08-05 00:00"
T1 = "2024-08-11 00:00"
TIME_FREQ = "10min"   # 10min / 1min 都可以

# ---- Height grid ----
Z_TOP = 3000          # 左轴高度上限 (m)
DZ = 25               # 垂直分辨率 (m)

# ---- Colormap range ----
Q_VMIN, Q_VMAX = 0, 22
CMAP = "rainbow"

# ---- CBH axis range ----
CBH_MIN, CBH_MAX = 0, 10000

# ---- OPTIONAL: Save merged AH/RH to a new Excel ----
SAVE_MERGED = True
MERGED_OUT_XLSX = r"E:\Beijing2024\MWR\ahrh\ah_rh_merged_from_2files.xlsx"

# =========================
# 0b) CBH 数据：优先读已输出 CSV；没有就解析 .ASC
# =========================
CBH_DATA_DIR = Path(r"E:\Beijing2024\MWR\cbh")

# 你原 CBH 代码里的两个 period（这里保留一致）
period1_start = "2024-06-06"
period1_end   = "2024-06-09"
period2_start = "2024-08-05"
period2_end   = "2024-08-10 1"
period1_start = safe_ts(period1_start).strftime("%Y-%m-%d")
period1_end   = safe_ts(period1_end).strftime("%Y-%m-%d")
period2_start = safe_ts(period2_start).strftime("%Y-%m-%d")
period2_end   = safe_ts(period2_end).strftime("%Y-%m-%d")

USE_BJT_FOR_PERIODS = True  # 你的 CBH 代码默认 True

DROP_CBH_EQ_10000_AS_NAN = False
DROP_CBH_EQ_0_AS_NAN     = False
DROP_WHEN_RAINFLAG_1     = False

CBH_OUT_CSV = CBH_DATA_DIR / "CBH_1min_period1_period2_merged_noTZ_interp.csv"

# =========================================================
# 1) 读取 AH/RH 宽表（通用）
# =========================================================
def _guess_time_column(df: pd.DataFrame) -> str:
    """
    猜测时间列：优先找包含 time/timestamp/datetime 的列；否则用第一列。
    """
    cols = list(df.columns)
    lowered = [str(c).strip().lower() for c in cols]
    for key in ["timestamp", "datetime", "date_time", "time", "date"]:
        for c, lc in zip(cols, lowered):
            if key in lc:
                return c
    return cols[0]

def read_wide_sheet_one_file(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    if df.empty:
        return df

    tcol = _guess_time_column(df)
    df = df.rename(columns={tcol: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # 强制数值化（高度列）
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def merge_same_sheet_across_files(xlsx_files, sheet_name: str) -> pd.DataFrame:
    frames = []
    for fp in xlsx_files:
        d = read_wide_sheet_one_file(fp, sheet_name)
        if not d.empty:
            frames.append(d)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0, ignore_index=False)

    # 同一 timestamp 重复：取均值
    out = out.groupby(out.index).mean(numeric_only=True)
    out = out.sort_index()
    return out

# =========================================================
# 2) AH 宽表 -> 规则高度网格（0..Z_TOP, step DZ）
# =========================================================
def _parse_height_m(colname) -> float | None:
    """
    从列名解析高度（m）。
    兼容示例：'0', '25', '100m', 'H_150', 'z=200', '200.0'
    解析不到返回 None。
    """
    s = str(colname).strip()
    # 直接是数字
    try:
        return float(s)
    except Exception:
        pass

    # 正则抓取第一个数（允许小数）
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def wide_to_height_grid(df_wide: pd.DataFrame, z_top: int, dz: int) -> pd.DataFrame:
    """
    输入：index=time，columns=高度（字符串/数字），values=AH
    输出：columns=规则高度（数值列），并按高度插值到 0..z_top 每 dz m
    """
    if df_wide.empty:
        return df_wide

    # 解析列名 -> 高度
    hmap = {}
    for c in df_wide.columns:
        hm = _parse_height_m(c)
        if hm is not None:
            hmap[c] = hm

    if not hmap:
        raise ValueError("无法从 ah_data 的列名解析出高度（m）。请检查列名是否包含高度数字。")

    df = df_wide[list(hmap.keys())].copy()
    df = df.rename(columns=hmap)

    # 同一高度重复列：取均值
    # 同一高度重复列：取均值（避免 groupby(axis=1) 的弃用警告）
    df = df.T.groupby(level=0).mean(numeric_only=True).T

    # 按高度排序
    df = df.reindex(sorted(df.columns), axis=1)

    # 目标高度网格
    z_target = np.arange(0, z_top + dz, dz, dtype=float)

    # 对每个时刻沿高度插值到目标网格
    cols_src = df.columns.to_numpy(dtype=float)

    arr_src = df.to_numpy(dtype=float)  # (nt, nz_src)
    nt = arr_src.shape[0]
    arr_tgt = np.full((nt, z_target.size), np.nan, dtype=float)

    for i in range(nt):
        y = arr_src[i, :]
        mask = np.isfinite(y) & np.isfinite(cols_src)
        if mask.sum() < 2:
            continue
        xs = cols_src[mask]
        ys = y[mask]
        # 要求 xs 单调（已排序），np.interp 不支持 nan 外推 -> 用边界值填充后再手动置 nan
        # 这里选择：超出源高度范围的目标高度置 nan（不外推）
        zmin, zmax = xs.min(), xs.max()
        in_range = (z_target >= zmin) & (z_target <= zmax)
        arr_tgt[i, in_range] = np.interp(z_target[in_range], xs, ys)

    df_tgt = pd.DataFrame(arr_tgt, index=df.index, columns=z_target)
    return df_tgt

# =========================================================
# 3) CBH 生成：复用你给的解析逻辑（并做“有则读，无则算”）
# =========================================================
def parse_one_cbh_file(fp: Path) -> pd.DataFrame:
    yy_list, mo_list, da_list, ho_list, mi_list, se_list, rf_list, cbh_list = [], [], [], [], [], [], [], []

    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "," not in line:
                continue
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 8:
                continue

            try:
                yy = int(parts[0]); mo = int(parts[1]); da = int(parts[2])
                ho = int(parts[3]); mi = int(parts[4]); se = int(parts[5])
                rf = int(float(parts[6])); cbh = float(parts[7])
            except Exception:
                continue

            yyyy = 2000 + yy if yy <= 69 else 1900 + yy
            yy_list.append(yyyy); mo_list.append(mo); da_list.append(da)
            ho_list.append(ho);   mi_list.append(mi); se_list.append(se)
            rf_list.append(rf);   cbh_list.append(cbh)

    if not yy_list:
        return pd.DataFrame(columns=["timestamp_utc", "timestamp_bjt", "rain_flag", "cbh_m", "source_file"])

    dt_utc = pd.to_datetime(
        {"year": yy_list, "month": mo_list, "day": da_list,
         "hour": ho_list, "minute": mi_list, "second": se_list},
        errors="coerce",
        utc=True
    )

    df = pd.DataFrame({
        "timestamp_utc": dt_utc,
        "rain_flag": rf_list,
        "cbh_m": cbh_list,
        "source_file": fp.name
    }).dropna(subset=["timestamp_utc"])

    df["rain_flag"] = pd.to_numeric(df["rain_flag"], errors="coerce").fillna(0).astype(int)
    df["cbh_m"]     = pd.to_numeric(df["cbh_m"], errors="coerce")
    df["timestamp_bjt"] = df["timestamp_utc"].dt.tz_convert("Asia/Shanghai")
    return df

def make_inclusive_day_range(start_date: str, end_date: str, tz: str):
    """[start 00:00, end+1day 00:00)"""
    start = pd.Timestamp(start_date).tz_localize(tz)
    end_excl = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize(tz)
    return start, end_excl

def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if DROP_WHEN_RAINFLAG_1:
        out.loc[out["rain_flag"] == 1, "cbh_m"] = np.nan
    if DROP_CBH_EQ_10000_AS_NAN:
        out.loc[out["cbh_m"] == 10000.0, "cbh_m"] = np.nan
    if DROP_CBH_EQ_0_AS_NAN:
        out.loc[out["cbh_m"] == 0.0, "cbh_m"] = np.nan
    return out

def one_period_to_1min(df_all: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if USE_BJT_FOR_PERIODS:
        tcol = "timestamp_bjt"
        tz = "Asia/Shanghai"
    else:
        tcol = "timestamp_utc"
        tz = "UTC"

    p_start, p_end_excl = make_inclusive_day_range(start_date, end_date, tz)

    df = df_all[(df_all[tcol] >= p_start) & (df_all[tcol] < p_end_excl)].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "cbh_m_1min_mean", "rain_flag_1min_max"])

    df = apply_qc(df)
    df = df[[tcol, "cbh_m", "rain_flag"]].dropna(subset=[tcol]).sort_values(tcol)

    # 同一秒重复 -> 聚合
    g = df.groupby(tcol, as_index=True)
    sec = pd.DataFrame({
        "cbh_m": g["cbh_m"].mean(),
        "rain_flag_max": g["rain_flag"].max()
    }).sort_index()

    # 1分钟聚合
    cbh_1min  = sec["cbh_m"].resample("1min").mean()
    rain_1min = sec["rain_flag_max"].resample("1min").max()

    out = pd.DataFrame({
        "cbh_m_1min_mean": cbh_1min,
        "rain_flag_1min_max": rain_1min
    })

    full_index = pd.date_range(p_start, p_end_excl - pd.Timedelta(minutes=1), freq="1min", tz=tz)
    out = out.reindex(full_index)

    out["cbh_m_1min_mean"] = out["cbh_m_1min_mean"].interpolate(method="time", limit_area="inside")
    out["rain_flag_1min_max"] = out["rain_flag_1min_max"].fillna(0).astype(int)

    out = out.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].dt.tz_localize(None)  # tz-aware -> naive
    return out

def load_or_build_cbh_1min() -> pd.DataFrame:
    if CBH_OUT_CSV.exists():
        cbh = pd.read_csv(CBH_OUT_CSV)
        cbh["timestamp"] = pd.to_datetime(cbh["timestamp"], errors="coerce")
        cbh = cbh.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        # 兼容列名（你的代码最后 rename 为 CBH）
        if "CBH" not in cbh.columns:
            # 尝试从原列名推断
            for cand in ["cbh_m_1min_mean", "cbh_m", "cbh"]:
                if cand in cbh.columns:
                    cbh = cbh.rename(columns={cand: "CBH"})
                    break
        cbh["CBH"] = pd.to_numeric(cbh["CBH"], errors="coerce")
        return cbh[["CBH"]]

    # 否则现场解析 .ASC
    files = sorted(CBH_DATA_DIR.glob("*.ASC")) + sorted(CBH_DATA_DIR.glob("*.asc"))
    if not files:
        raise FileNotFoundError(f"No .ASC files found in {CBH_DATA_DIR}")

    frames = []
    for fp in files:
        d = parse_one_cbh_file(fp)
        if not d.empty:
            frames.append(d)
    if not frames:
        raise RuntimeError("Parsed 0 valid rows from all CBH files. Check file format.")

    cbh_all = pd.concat(frames, ignore_index=True)

    p1 = one_period_to_1min(cbh_all, period1_start, period1_end)
    p2 = one_period_to_1min(cbh_all, period2_start, period2_end)

    out = pd.concat([p1, p2], ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True).dropna(subset=["timestamp"])
    out["cbh_m_1min_mean"] = pd.to_numeric(out["cbh_m_1min_mean"], errors="coerce")

    out.to_csv(CBH_OUT_CSV, index=False, encoding="utf-8-sig")

    cbh = out.rename(columns={"cbh_m_1min_mean": "CBH"})
    cbh["timestamp"] = pd.to_datetime(cbh["timestamp"], errors="coerce")
    cbh = cbh.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # 你的原代码最后是 cbh_1min = cbh.resample('1min').mean()
    cbh_1min = cbh.resample("1min").mean(numeric_only=True)
    return cbh_1min[["CBH"]]

# =========================================================
# 4) MAIN：读入合并 -> 对齐时间/高度 -> 画图
# =========================================================
# 4.1 合并 AH/RH
ah_wide = merge_same_sheet_across_files(EXCEL_FILES, SHEET_AH)
rh_wide = merge_same_sheet_across_files(EXCEL_FILES, SHEET_RH)
ah_wide.index = ah_wide.index + pd.Timedelta(hours=8)
rh_wide.index = rh_wide.index + pd.Timedelta(hours=8)

if ah_wide.empty:
    raise RuntimeError("合并后 ah_data 为空：请检查文件路径/工作表名/数据是否存在。")
if rh_wide.empty:
    print("[WARN] 合并后 rh_data 为空（不影响本图，因为本图只画 AH + CBH）。")

# 4.2 AH -> 规则高度网格
ah_grid = wide_to_height_grid(ah_wide, z_top=Z_TOP, dz=DZ)
# =========================================================
# 4.3 用 period1+period2 构造时间序列（中间断轴）
# =========================================================

def _clean_date_str(s: str) -> str:
    s = str(s).strip()
    # 抓取开头的 YYYY-MM-DD（如果存在）
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    return s  # 其他情况原样返回，让 pd.to_datetime 决定

def period_times(start_date: str, end_date: str, freq: str):
    """[start 00:00, end+1day 00:00) 以 freq 生成时间序列（naive datetime）"""
    start_date = _clean_date_str(start_date)
    end_date   = _clean_date_str(end_date)

    p_start = pd.to_datetime(start_date, errors="raise")  # 00:00
    p_end_excl = pd.to_datetime(end_date, errors="raise") + pd.Timedelta(days=1)
    return pd.date_range(p_start, p_end_excl, freq=freq, inclusive="left")

times1 = period_times(period1_start, period1_end, TIME_FREQ)
times2 = period_times(period2_start, period2_end, TIME_FREQ)

# 规则高度
z = np.arange(0, Z_TOP + DZ, DZ, dtype=float)

# AH：重采样到 TIME_FREQ，然后分别对齐到 times1/times2
ah_res = ah_grid.resample(TIME_FREQ).mean(numeric_only=True)

ah_p1 = ah_res.reindex(times1).reindex(columns=z)
ah_p2 = ah_res.reindex(times2).reindex(columns=z)

q1 = np.clip(ah_p1.to_numpy(dtype=float), Q_VMIN, Q_VMAX)  # (nt1, nz)
q2 = np.clip(ah_p2.to_numpy(dtype=float), Q_VMIN, Q_VMAX)  # (nt2, nz)

# CBH：从 1min -> TIME_FREQ，再分别截取 period1/period2
cbh_1min = load_or_build_cbh_1min()

cbh_p1 = (cbh_1min.loc[(cbh_1min.index >= times1[0]) & (cbh_1min.index < times1[-1] + pd.Timedelta(TIME_FREQ))]
          .resample(TIME_FREQ).mean(numeric_only=True).reindex(times1))
cbh_p2 = (cbh_1min.loc[(cbh_1min.index >= times2[0]) & (cbh_1min.index < times2[-1] + pd.Timedelta(TIME_FREQ))]
          .resample(TIME_FREQ).mean(numeric_only=True).reindex(times2))

cbh1 = cbh_p1["CBH"].to_numpy(dtype=float)
cbh2 = cbh_p2["CBH"].to_numpy(dtype=float)
def make_month_first_formatter(window_start):
    """
    第一个主刻度显示: '06\\nJun'（day + newline + month）
    后续主刻度只显示: '07' '08' ...
    """
    w0 = pd.Timestamp(window_start).normalize()

    def _fmt(x, pos=None):
        dt = mdates.num2date(x).replace(tzinfo=None)
        d0 = pd.Timestamp(dt).normalize()
        if d0 == w0:
            return dt.strftime("%d\n%b")   # 06\nJun
        return dt.strftime("%d")           # 07
    return FuncFormatter(_fmt)

# =========================================================
# 5) 画图：broken x-axis（仅底部断轴标记）+ 共享色标 + CBH 右轴
# =========================================================
plt.rcParams.update({
    "font.family": "Arial",
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
})

w1, w2 = len(times1), len(times2)

fig = plt.figure(figsize=(14, 4.3), dpi=200)
gs = fig.add_gridspec(1, 2, width_ratios=[w1, w2], wspace=0.05)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

dt_freq = pd.Timedelta(TIME_FREQ)

# --- period1 heatmap ---
x0_1 = mdates.date2num(times1[0])
x1_1 = mdates.date2num(times1[-1] + dt_freq)
im1 = ax1.imshow(
    q1.T, origin="lower", aspect="auto", interpolation="nearest",
    extent=[x0_1, x1_1, z[0], z[-1]],
    cmap=CMAP, vmin=Q_VMIN, vmax=Q_VMAX
)

# --- period2 heatmap ---
x0_2 = mdates.date2num(times2[0])
x1_2 = mdates.date2num(times2[-1] + dt_freq)
im2 = ax2.imshow(
    q2.T, origin="lower", aspect="auto", interpolation="nearest",
    extent=[x0_2, x1_2, z[0], z[-1]],
    cmap=CMAP, vmin=Q_VMIN, vmax=Q_VMAX
)

# 统一 y 范围：两段都用 0–3500
ax1.set_ylim(0, 3000)
ax1.set_ylabel("Height(m a.g.l.)", fontsize=18)
ax1.set_xlabel("Datetime(BJT)", fontsize=18,x=1.25)

for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(mdates.DayLocator())                 # 每天 00:00
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))     # 每天 12:00（小刻度）
    ax.tick_params(axis="x", which="major", labelsize=18, width=1.5, length=6)
    ax.tick_params(axis="x", which="minor", width=1.2, length=3)

# 两个窗口分别设置“首日带月份”的 formatter
ax1.xaxis.set_major_formatter(make_month_first_formatter(period1_start))
ax2.xaxis.set_major_formatter(make_month_first_formatter(period2_start))

# 确保 xlim 精准（避免 locator/formatter跑偏）
ax2.set_xlim(safe_ts(period2_start), safe_ts(period2_end) + pd.Timedelta(days=1))
ax1.set_xlim(safe_ts(period1_start), safe_ts(period1_end) + pd.Timedelta(days=1))
ax1.tick_params(axis="y", which="both", left=True, labelleft=True)

# 2) 彻底关闭 ax2 的 y ticks/labels（中间那条），但不要用 yaxis.set_visible(False)
ax2.tick_params(axis="y", which="both",
                left=False, right=False,
                labelleft=False, labelright=False,
                length=0)
ax2.set_yticks([])                 # 双保险：清空 ax2 y ticks
ax2.set_ylabel("")
# ---------- 1) 强制显示最左侧(ax1)的 y 刻度和标签 ----------
yticks = np.arange(0, 3000 + 1, 500)
ax1.set_yticks(yticks)
ax1.set_yticklabels([f"{int(v)}" for v in yticks], fontsize=18)

# ===== y 轴：major + 1 个 minor（最左侧 ax1）=====
ax1.yaxis.set_major_locator(MultipleLocator(500))   # 0, 500, 1000, ...
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))    # 每个 major 间分成 2 段 -> 1 个 minor
ax1.tick_params(axis="y", which="minor", left=True, length=3, width=1.5)
ax1.tick_params(axis="y", which="major", left=True, length=6, width=1.5)
ax1.spines["left"].set_visible(True)

# 给左侧留空间，避免 y ticklabel 被裁掉（很关键）
fig.subplots_adjust(left=0.14)

# ---------- 2) 彻底关掉断轴中间(ax2左边)的 y 轴痕迹 ----------
ax2.tick_params(axis="y", which="both",
                left=False, right=False,
                labelleft=False, labelright=False,
                length=0)
ax2.set_yticks([])
ax2.spines["left"].set_visible(False)

# --- CBH: 两段各自 twinx（仅最右侧显示刻度/标签）---
ax1r = ax1.twinx()
ax2r = ax2.twinx()

# 3) 清掉 ax1r（左半段 CBH 右轴）在中间断轴处的紫色刻度/标签（只保留曲线）
ax1r.set_yticks([])
ax1r.tick_params(axis="y", which="both",
                 right=False, left=False,
                 labelright=False, labelleft=False,
                 length=0)
ax1r.set_ylabel("")
ax1r.spines["right"].set_visible(False)   # 双保险：不要让它在中间出现紫色轴线

ax2.set_yticks([])          # 双保险：清空 ax2 自己的 y ticks
ax2.set_ylabel("")

ax1r.plot(times1, cbh1, color="purple", linewidth=1.5)
ax2r.plot(times2, cbh2, color="purple", linewidth=1.5)
ax2r.tick_params(axis="y", colors="purple", labelsize=18, width=1.5)
ax2r.set_ylabel("Cloud Base Height(m a.g.l.)", fontsize=18, color="purple", rotation=270, labelpad=18)
ax2r.yaxis.label.set_color("purple")

# --- 隐藏中间轴线（spines）---
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax1r.spines["right"].set_visible(False)
ax2r.spines["left"].set_visible(False)

# --- 断轴斜杠：仅底部保留（不画顶部）---
d = 0.015
kwargs1 = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=1.5)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs1)  # 仅底部

kwargs2 = dict(transform=ax2.transAxes, color="k", clip_on=False, linewidth=1.5)
ax2.plot((-d, +d), (-d, +d), **kwargs2)        # 仅底部
# ===== 强制最左侧 y 轴刻度/标签显示（放在最后，避免被后续操作覆盖）=====
ax1.yaxis.set_ticks_position("left")
ax1.yaxis.set_label_position("left")
ax1.spines["left"].set_visible(True)
yticks = np.arange(0, 3000 + 1, 500)
ax1.set_yticks(yticks)

# 防止 ticklabel 被裁剪（双保险）
for lab in ax1.get_yticklabels():
    lab.set_clip_on(False)

# --- 共享 colorbar ---
fig.subplots_adjust(left=0.14, right=0.82)  # left 你可以再增大到 0.14

cax = fig.add_axes([0.90, 0.15, 0.018, 0.70])
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label("Absolute Humidity(g/m$^3$)", fontsize=18, rotation=270, labelpad=22)
cbar.ax.tick_params(labelsize=18, width=1.5)

ax2r.spines["right"].set_color("purple")
ax1r.set_ylim(-500, 10500)
ax2r.set_ylim(-500, 10500) # 使用默认的最大值，或者也可以是其他值
# --- 保存为 TIFF（300 dpi，透明背景，紧边界，deflate 压缩）---
out_tif = r"E:\Beijing2024\出图TIF\AH_CBH_brokenaxis.tif"
fig.savefig(
    out_tif,
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_adobe_deflate"}
)
print("[OK] Saved:", out_tif)

plt.show()

# ======================================================================
# APPEND ONLY: plot RH (replace AH field with RH) with the SAME settings
# Paste this AFTER your existing plt.show()
# ======================================================================

# 1) RH -> 规则高度网格（复用你上面已经读好的 rh_wide）
if rh_wide.empty:
    raise RuntimeError("rh_data 为空：请检查 rh_data 工作表是否有数据。")

rh_grid = wide_to_height_grid(rh_wide, z_top=Z_TOP, dz=DZ)

# 2) RH：重采样到 TIME_FREQ，然后分别对齐到 times1/times2（完全照抄 AH 的逻辑）
rh_res = rh_grid.resample(TIME_FREQ).mean(numeric_only=True)
rh_p1 = rh_res.reindex(times1).reindex(columns=z)
rh_p2 = rh_res.reindex(times2).reindex(columns=z)

RH_VMIN, RH_VMAX = 0, 100
r1 = np.clip(rh_p1.to_numpy(dtype=float), RH_VMIN, RH_VMAX)
r2 = np.clip(rh_p2.to_numpy(dtype=float), RH_VMIN, RH_VMAX)

# 3) 画图（完全复用你当前的样式/断轴/CBH右轴，仅把 q1/q2 换成 r1/r2）
w1, w2 = len(times1), len(times2)

fig = plt.figure(figsize=(14, 4.3), dpi=200)
gs = fig.add_gridspec(1, 2, width_ratios=[w1, w2], wspace=0.05)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

dt_freq = pd.Timedelta(TIME_FREQ)

# --- period1 heatmap (RH) ---
x0_1 = mdates.date2num(times1[0])
x1_1 = mdates.date2num(times1[-1] + dt_freq)
im1 = ax1.imshow(
    r1.T, origin="lower", aspect="auto", interpolation="nearest",
    extent=[x0_1, x1_1, z[0], z[-1]],
    cmap=CMAP, vmin=RH_VMIN, vmax=RH_VMAX
)

# --- period2 heatmap (RH) ---
x0_2 = mdates.date2num(times2[0])
x1_2 = mdates.date2num(times2[-1] + dt_freq)
im2 = ax2.imshow(
    r2.T, origin="lower", aspect="auto", interpolation="nearest",
    extent=[x0_2, x1_2, z[0], z[-1]],
    cmap=CMAP, vmin=RH_VMIN, vmax=RH_VMAX
)

# y axis
ax1.set_ylim(0, 3000)
ax1.set_ylabel("Height(m a.g.l.)", fontsize=18)
ax1.set_xlabel("Datetime(BJT)", fontsize=18, x=1.25)

for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
    ax.tick_params(axis="x", which="major", labelsize=18, width=1.5, length=6)
    ax.tick_params(axis="x", which="minor", width=1.2, length=3)

ax1.xaxis.set_major_formatter(make_month_first_formatter(period1_start))
ax2.xaxis.set_major_formatter(make_month_first_formatter(period2_start))

ax2.set_xlim(safe_ts(period2_start), safe_ts(period2_end) + pd.Timedelta(days=1))
ax1.set_xlim(safe_ts(period1_start), safe_ts(period1_end) + pd.Timedelta(days=1))

ax1.tick_params(axis="y", which="both", left=True, labelleft=True)
ax2.tick_params(axis="y", which="both",
                left=False, right=False,
                labelleft=False, labelright=False,
                length=0)
ax2.set_yticks([])
ax2.set_ylabel("")

# 强制最左侧 y ticks（照抄）
yticks = np.arange(0, 3000 + 1, 500)
ax1.set_yticks(yticks)
ax1.set_yticklabels([f"{int(v)}" for v in yticks], fontsize=18)

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
ax1.yaxis.set_major_locator(MultipleLocator(500))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.tick_params(axis="y", which="minor", left=True, length=3, width=1.5)
ax1.tick_params(axis="y", which="major", left=True, length=6, width=1.5)

ax1.spines["left"].set_visible(True)
fig.subplots_adjust(left=0.14)

ax2.set_yticks([])
ax2.spines["left"].set_visible(False)

# --- CBH (复用你已算好的 cbh1/cbh2) ---
ax1r = ax1.twinx()
ax2r = ax2.twinx()

ax1r.set_yticks([])
ax1r.tick_params(axis="y", which="both",
                 right=False, left=False,
                 labelright=False, labelleft=False,
                 length=0)
ax1r.set_ylabel("")
ax1r.spines["right"].set_visible(False)

ax1r.plot(times1, cbh1, color="purple", linewidth=1.5)
ax2r.plot(times2, cbh2, color="purple", linewidth=1.5)

ax2r.tick_params(axis="y", colors="purple", labelsize=18, width=1.5)
ax2r.set_ylabel("Cloud Base Height(m a.g.l.)", fontsize=18, color="purple",
                rotation=270, labelpad=18)
ax2r.yaxis.label.set_color("purple")

ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax1r.spines["right"].set_visible(False)
ax2r.spines["left"].set_visible(False)

# --- 断轴斜杠（底部）---
d = 0.015
kwargs1 = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=1.5)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs1)

kwargs2 = dict(transform=ax2.transAxes, color="k", clip_on=False, linewidth=1.5)
ax2.plot((-d, +d), (-d, +d), **kwargs2)

# 最左侧 y 轴双保险
ax1.yaxis.set_ticks_position("left")
ax1.yaxis.set_label_position("left")
ax1.spines["left"].set_visible(True)
ax1.set_yticks(yticks)
for lab in ax1.get_yticklabels():
    lab.set_clip_on(False)

# --- 共享 colorbar ---
fig.subplots_adjust(left=0.14, right=0.82)
cax = fig.add_axes([0.90, 0.15, 0.018, 0.70])
cbar = fig.colorbar(im2, cax=cax)

# 这里仅改了色标文字（不影响任何参数/范围/样式）
cbar.set_label("Relative Humidity(%)", fontsize=18, rotation=270, labelpad=22)
cbar.ax.tick_params(labelsize=18, width=1.5)

ax2r.spines["right"].set_color("purple")
ax1r.set_ylim(-500, 10500)
ax2r.set_ylim(-500, 10500)

# --- 保存为 TIFF（同样的保存参数）---
out_tif_rh = r"E:\Beijing2024\出图TIF\RH_CBH_brokenaxis.tif"
fig.savefig(
    out_tif_rh,
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_adobe_deflate"}
)
print("[OK] Saved:", out_tif_rh)

plt.show()
