
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from glob import glob
from datetime import datetime
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from metpy.units import units
import metpy.calc as mpcalc
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MultipleLocator


#####处理微波辐射计数据#####
folder_path = 'E:\\Beijing2024\\MWR\\ahrh\\04-07\\'  # 指定文件夹路径
ah = pd.read_excel(folder_path+'ah&rh_merged.xlsx',sheet_name="ah_data")
ah = ah.rename(columns={'timestamp':  'timestamps'})       
ah['timestamps'] = pd.to_datetime(ah['timestamps'])
ah.set_index('timestamps', inplace=True)
rh = pd.read_excel(folder_path+'ah&rh_merged.xlsx',sheet_name="rh_data")
rh = rh.rename(columns={'timestamp':  'timestamps'})
rh['timestamps'] = pd.to_datetime(rh['timestamps'])
rh.set_index('timestamps', inplace=True)

folder_path = 'E:\\Beijing2024\\MWR\\temp\\0702-0810\\'
temp = pd.read_excel(folder_path+'temp merged.xlsx',sheet_name="0702-0810")
temp = temp.rename(columns={'timestamp':  'timestamps'}) 
temp['timestamps'] = pd.to_datetime(temp['timestamps'], format='%Y/%m/%d %H:%M:%S')
temp.set_index('timestamps', inplace=True)

#####处理微波辐射计数据#####
folder_path2 = 'E:\\Beijing2024\\MWR\\ahrh\\04-06\\'  # 指定文件夹路径
ah2 = pd.read_excel(folder_path2+'ah&rh_merged.xlsx',sheet_name="ah_data")
ah2 = ah2.rename(columns={'timestamp':  'timestamps'})       
ah2['timestamps'] = pd.to_datetime(ah2['timestamps'])
ah2.set_index('timestamps', inplace=True)
rh2 = pd.read_excel(folder_path2+'ah&rh_merged.xlsx',sheet_name="rh_data")
rh2 = rh2.rename(columns={'timestamp':  'timestamps'})
rh2['timestamps'] = pd.to_datetime(rh2['timestamps'])
rh2.set_index('timestamps', inplace=True)

folder_path2 = 'E:\\Beijing2024\\MWR\\temp\\0605-0624北京\\'
temp2 = pd.read_excel(folder_path2+'temp merged.xlsx',sheet_name="0605-0624")
temp2 = temp2.rename(columns={'timestamp':  'timestamps'}) 
temp2['timestamps'] = pd.to_datetime(temp2['timestamps'], format='%Y/%m/%d %H:%M:%S')
temp2.set_index('timestamps', inplace=True)

temp = pd.concat([temp,temp2], axis=0).sort_index()
rh = pd.concat([rh,rh2], axis=0).sort_index()

start_time = '2024-06-05 10:35:00'#temp.index.min()
end_time = '2024-08-10 11:58:59'#temp.index.max()

temp_1min = temp[(temp.index >= start_time)&(temp.index <= end_time)]
temp_1min_index = pd.date_range(start=start_time,end=end_time,freq='1min')  # '1T' 表示1分钟
combined_index = temp_1min.index.union(temp_1min_index).drop_duplicates().sort_values()
temp_1min_reindexed = temp_1min.reindex(combined_index) 
temp_1min_reindexed_interp = temp_1min_reindexed.interpolate(method='time') 
temp_1min_reindexed_interp.update(temp_1min) 
temp_1min_reindexed_interp_final = temp_1min_reindexed_interp.reset_index().rename(columns={'index':  'timestamps'})
temp_1min_reindexed_interp_final = temp_1min_reindexed_interp_final[temp_1min_reindexed_interp_final['timestamps'].dt.second  == 0].reset_index(drop=True)

rh_1min = rh[(rh.index >= start_time)&(rh.index <= end_time)]
rh_1min_index = pd.date_range(start=start_time,end=end_time,freq='1min')  # '1T' 表示1分钟
combined_index = rh_1min.index.union(rh_1min_index).drop_duplicates().sort_values()
rh_1min_reindexed = rh_1min.reindex(combined_index) 
rh_1min_reindexed_interpolated = rh_1min_reindexed.interpolate(method='time') 
rh_1min_reindexed_interpolated.update(rh_1min) 
rh_1min_reindexed_interpolated_final = rh_1min_reindexed_interpolated.reset_index().rename(columns={'index':  'timestamps'})
rh_1min_reindexed_interpolated_final = rh_1min_reindexed_interpolated_final[rh_1min_reindexed_interpolated_final['timestamps'].dt.second  == 0].reset_index(drop=True)

temp = temp_1min_reindexed_interp_final
rh = rh_1min_reindexed_interpolated_final
temp = temp.set_index("timestamps")
rh = rh.set_index("timestamps")

temp.index = temp.index+pd.Timedelta(hours=8)
rh.index = rh.index+pd.Timedelta(hours=8)

# =========================
# 1. 常数 & 压力计算函数
# =========================

L = 0.0065     # 温度梯度 (K/m)
g = 9.8017104  # 重力加速度 (m/s^2)
M = 0.0289644  # 空气平均摩尔质量 (kg/mol)
R = 8.3144598  # 气体常数 (J/(mol·K))

# =========================
# 2. 选择时间窗口 & 对齐数据
# =========================

# 两个时间窗口
win1_start = "2024-06-06 00:00"
win1_end   = "2024-06-11 00:00"   # 结束时间用 < 06-11
win2_start = "2024-08-05 00:00"
win2_end   = "2024-08-11 00:00"

# 保证列名是数值高度
temp.columns = temp.columns.astype(float)
rh.columns   = rh.columns.astype(float)
rh = rh.reindex(columns=temp.columns)  # 列顺序对齐
# 确保索引是时间类型
temp.index = pd.to_datetime(temp.index)
rh.index   = pd.to_datetime(rh.index)

# 先排序（不是必须，但推荐）
temp = temp.sort_index()
rh   = rh.sort_index()

# 取两者公共部分的时间索引
common_index = temp.index.intersection(rh.index)

# 用公共索引重新对齐两个表
temp = temp.loc[common_index]
rh   = rh.loc[common_index]

# 时间掩码
mask1 = (temp.index >= win1_start) & (temp.index < win1_end)
mask2 = (temp.index >= win2_start) & (temp.index < win2_end)
mask  = mask1 | mask2

temp_sel = temp.loc[mask].copy()
rh_sel   = rh.loc[mask].copy()

if temp_sel.empty:
    raise ValueError("在两个时间窗内 temp 没有数据，请检查时间索引。")

###############################################################
folder_path = 'E:\\Beijing2024\\MWR\\met\\'  # 指定路径
all_data = []  # 创建一个空列表以存储数据帧

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)  # 获取文件完整路径
    # 使用pandas读取文件，跳过前8行
    data = pd.read_table(file_path, skiprows=19,delimiter=',', header=None, encoding='ANSI')
    all_data.append(data)  # 将数据帧添加到列表中
merged_data = pd.concat(all_data, ignore_index=True)

met=merged_data.iloc[:,:13]

met.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second','rain_flag','pressure','temperature','RH', 'ws', 'wd','rainfall_rate_mmh']
met['Year'] = met['Year'].apply(lambda x: 2000 + x)

# 将两位数年份转换为四位数年份
# 创建一个新的列来存储时间戳
met['timestamp'] = pd.to_datetime(met[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
met['timestamp'] = met['timestamp'] + pd.Timedelta(hours=8)

df1_met = pd.DataFrame(met)
df1_met['timestamp'] = pd.to_datetime(df1_met['timestamp'])
df1_met.set_index('timestamp', inplace=True)
met_1min=df1_met.resample('1min').mean().interpolate()
met_1min = met_1min[['pressure', 'temperature', 'RH','rainfall_rate_mmh']]

# met_1min 对齐到 MWR 时间轴（索引必须是 DatetimeIndex）
met_1min_sel = met_1min.reindex(temp_sel.index).interpolate('time')

# 地面气压和温度（已知是 hPa 和 K）
P0_series = met_1min_sel['pressure'].to_numpy()      # hPa
T0_series = met_1min_sel['temperature'].to_numpy()   # K

# 高度数组（相对地面/仪器）
heights_m = temp_sel.columns.to_numpy()  # m

# =========================
# 3. 计算整层气压 P(t,z)
# =========================
def calculate_pressure_series(P0_series, T0_series, heights):
    """
    P(z) = P0 * (1 - L*z/T0)^(g*M/(R*L))

    P0_series: 1D array, 地面气压 [hPa]
    T0_series: 1D array, 地面温度 [K]
    heights  : 1D array, 高度（相对地面/仪器）[m]

    返回: pressures (n_time, n_height) [hPa]
    """
    P0_series = np.asarray(P0_series, dtype=float)
    T0_series = np.asarray(T0_series, dtype=float)
    heights   = np.asarray(heights,   dtype=float)

    pressures = np.zeros((len(P0_series), len(heights)), dtype=float)
    exponent  = g * M / (R * L)

    for i in range(len(P0_series)):
        P0 = P0_series[i]
        T0 = T0_series[i]  # K
        pressures[i, :] = P0 * (1 - (L * heights) / T0) ** exponent

    return pressures


pressure_2d = calculate_pressure_series(P0_series, T0_series, heights_m)

pressure_df = pd.DataFrame(
    pressure_2d,
    index=temp_sel.index,
    columns=temp_sel.columns
)


# =========================
# 4. 计算露点温度 Td
# =========================

# 温度：K
T = temp_sel.to_numpy() * units.kelvin
# 把 rh_sel 中所有等于 0 的元素改成 0.1
rh_sel = rh_sel.mask(rh_sel == 0, 0.01)#RH=0计算露点会报错

# 相对湿度：0–1
RH = (rh_sel.to_numpy() / 100.0) * units.dimensionless

# 露点（MetPy 会返回 Quantity）
Td = mpcalc.dewpoint_from_relative_humidity(T, RH)

# 转成摄氏度，方便查看
#Td_c = Td.to('degC').m

dewpoint_df = pd.DataFrame(
    Td,
    index=temp_sel.index,
    columns=temp_sel.columns
)

# =========================
# 5. 计算比湿 q
# =========================

# 气压加单位
P = pressure_2d * units.hectopascal   # hPa

# 比湿（kg/kg）
q = mpcalc.specific_humidity_from_dewpoint(P, Td, phase='auto')


q_gkg  = q.to('g/kg').m


q_gkg_df = pd.DataFrame(
    q_gkg,
    index=temp_sel.index,
    columns=temp_sel.columns
)


# ---------------------------
# 0) 目标气压层 (hPa)
# ---------------------------
p_levels_desc = np.arange(850, 300 - 1, -25, dtype=float)  # 1000, 975, ..., 200
p_levels_asc  = p_levels_desc[::-1]                         # 200 -> 1000 (给 np.interp 用)

# ---------------------------
# 1) 取数组（确保列顺序一致）
# ---------------------------
# temp_sel: K
T = temp_sel.to_numpy(dtype=float)

# q_gkg_df: g/kg -> kg/kg（插值更规范；最后再转回 g/kg）
q = (q_gkg_df.to_numpy(dtype=float)) / 1000.0

# pressure_df: 可能是 Pa 或 hPa
p = pressure_df.to_numpy(dtype=float)


# ---------------------------
# 2) 单行插值函数：p(z) -> y(p)
#    - 自动对 p 排序成升序
#    - 去重
#    - 不外推（范围外 NaN）
# ---------------------------
def interp_to_pressure_row(p_row, y_row, p_target_asc):
    m = np.isfinite(p_row) & np.isfinite(y_row)
    if m.sum() < 2:
        return np.full_like(p_target_asc, np.nan, dtype=float)

    x = p_row[m]
    y = y_row[m]

    # 升序排序（np.interp 要求 xp 升序）
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # 去重
    x_u, idx_u = np.unique(x, return_index=True)
    y_u = y[idx_u]
    if x_u.size < 2:
        return np.full_like(p_target_asc, np.nan, dtype=float)

    yi = np.interp(p_target_asc, x_u, y_u)

    # 禁止外推：目标超出有效范围 -> NaN
    out = (p_target_asc < x_u.min()) | (p_target_asc > x_u.max())
    yi[out] = np.nan
    return yi

# ---------------------------
# 3) 循环每个时刻插值
# ---------------------------
nt = p.shape[0]
T_p = np.empty((nt, len(p_levels_asc)), dtype=float)
q_p = np.empty((nt, len(p_levels_asc)), dtype=float)

for i in range(nt):
    T_p[i, :] = interp_to_pressure_row(p[i, :], T[i, :], p_levels_asc)
    q_p[i, :] = interp_to_pressure_row(p[i, :], q[i, :], p_levels_asc)

# ---------------------------
# 4) 组装输出：列为 1000->200 hPa
# ---------------------------
temp_p_df = pd.DataFrame(T_p[:, ::-1], index=temp_sel.index, columns=p_levels_desc)      # K
q_p_gkg_df = pd.DataFrame(q_p[:, ::-1] * 1000.0, index=q_gkg_df.index, columns=p_levels_desc)  # g/kg

print("temp_p_df:", temp_p_df.shape, "K")
print("q_p_gkg_df:", q_p_gkg_df.shape, "g/kg")

with pd.ExcelWriter('E:\\Beijing2024\\数据处理结果2nd\\MWR 850-300 t q .xlsx', engine='openpyxl', date_format='yyyy/mm/dd hh:mm:ss') as writer:

    temp_p_df.reset_index().to_excel(writer, sheet_name='T_1min', index=False)
    
    q_p_gkg_df.reset_index().to_excel(writer, sheet_name='rh_1min', index=False)
    
    #ah_1min_reindexed_interpolated_final.to_excel(writer, sheet_name='ah_1min', index=False)
    
    #q_1min.to_excel(writer, sheet_name='q_1min', index=False)
    
    #θv_1min.to_excel(writer, sheet_name='θv_1min', index=False)

    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
            # 设置A到F列
        for col in ['A']:
            worksheet.column_dimensions[col].width = 20



#######读取卫星数据########

fy4B_T = pd.read_excel("E:\\Beijing2024\\fy4A\\t&q_fy4b-06050610_08040811.xlsx",sheet_name='t_filter')

fy4B_q = pd.read_excel("E:\\Beijing2024\\fy4A\\t&q_fy4b-06050610_08040811.xlsx",sheet_name='q_filter')

#######读取探空数据########

radiosonde = pd.read_excel("E:\\Beijing2024\\探空数据-54511\\radiosonde_all_profiles.xlsx",sheet_name='interp_300_850_25hPa')
radiosonde['time'] = pd.to_datetime(radiosonde['time'])

# 加上8小时
radiosonde['time'] = radiosonde['time'] + pd.Timedelta(hours=8)
#######读取铁塔数据########
# Set the path to your target directory
path = r'E:\Beijing2024\铁塔\2024铁塔平均场6月6至6月98月5日至8月10日10天数据(1)\0805-0810\\'

# Get a list of all files in the folder
files = [f for f in os.listdir(path) if f.endswith('.txt')]  # Assuming all files are .txt files

# Initialize an empty list to collect data frames
all_data = []

# Loop through each file in the folder
for file in files:
    file_path = os.path.join(path, file)
    
    # Read data from the file
    df = pd.read_table(file_path, skiprows=5, header=None, sep=r'\s+')
    print(file)
    # Iterate over the data and create timestamps
    for i in range(1, 4321):  # i 从 1 到 4320
        # 获取当前时间点的各个部分
        if (18*i - 2) >= len(df):
            break
        year = "20" + str(df.iloc[18*i-2, 1])  # 获取年份（假设年份在第 1 列）
        day = df.iloc[18*i-2, 3]  # 获取日期（假设日期在第 3 列）
        month = df.iloc[18*i-2, 5]  # 获取月份（假设月份在第 5 列）
        hour = df.iloc[18*i-1, 1]  # 获取小时（假设小时在第 1 列）
        minute = df.iloc[18*i-1, 3]  # 获取分钟（假设分钟在第 3 列）
        second = df.iloc[18*i-1, 5]  # 获取秒（假设秒在第 5 列）

        # 创建时间戳
        timestamps = pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:{second}")

        # 将时间戳添加到 DataFrame 中
        df.loc[18*i-17:18*i-3, 'timestamps'] = timestamps
    
    # Append the processed DataFrame to the list
    all_data.append(df)

# Concatenate all DataFrames from all files into one large DataFrame
final_df = pd.concat(all_data, ignore_index=True)
tieta_filtered = final_df[~final_df.iloc[:, 0].str.contains('=|-', na=False)]
tieta_filtered.columns = tieta_filtered.iloc[0]
tieta_filtered.columns.values[6] = 'timestamps'
tieta_filtered = tieta_filtered[1:].reset_index(drop=True)  # 删除第一行并重置索引
tieta_filtered.set_index('timestamps', inplace=True)
tieta_filtered.replace('999.0', np.nan, inplace=True)
#Vnw和Vse是由同一条线上两个传感器测到的，可以任取其中一个或者作平均
tieta_filtered['ws_mean(m/s)'] = tieta_filtered.iloc[:, 1:3].apply(pd.to_numeric, errors='coerce').mean(axis=1)
tieta_filtered['T(c)'] = tieta_filtered['T(c)'].apply(lambda x: pd.to_numeric(x,  errors='coerce'))
tieta_filtered['T(c)'] = tieta_filtered['T(c)'].astype(float).round(3)
tieta_filtered['T(K)'] = tieta_filtered['T(c)']+273.15
tieta_filtered['RH(%)'] = tieta_filtered['RH(%)'].apply(lambda x: pd.to_numeric(x,  errors='coerce'))
tieta_filtered['RH(%)'] = tieta_filtered['RH(%)'].astype(float).round(1)
tieta_filtered['D(deg)'] = tieta_filtered['D(deg)'].apply(lambda x: pd.to_numeric(x,  errors='coerce'))
tieta_filtered['D(deg)'] = tieta_filtered['D(deg)'].astype(float).round(1)
tieta_filtered['H(m)'] = tieta_filtered['H(m)'].apply(lambda x: pd.to_numeric(x,  errors='coerce'))
tieta_filtered['H(m)'] = tieta_filtered['H(m)'].astype(float).round(0)
tieta_filtered['u'] = tieta_filtered['ws_mean(m/s)'] *np.sin(np.deg2rad(tieta_filtered['D(deg)']))
tieta_filtered['v'] = tieta_filtered['ws_mean(m/s)'] *np.cos(np.deg2rad(tieta_filtered['D(deg)']))
tieta_filtered = tieta_filtered.drop(columns=['Vnw(m/s)', 'Vse(m/s)', 'T(c)'])
tieta_filtered.columns = ['H(m)', 'wd_tieta', 'RH_tieta', 'ws_tieta', 'T(K)_tieta','u','v']
tieta_1min = (tieta_filtered.reset_index().groupby(['H(m)', pd.Grouper(key='timestamps', freq='1min')]).mean()).reset_index()
df = tieta_1min.copy()
rh_mean_8_320 = (
    df[df["H(m)"].isin([8.0, 320.0])]
      .groupby("timestamps", as_index=False)["RH_tieta"]
      .mean()
      .rename(columns={"RH_tieta": "RH_mean_8_320"})
)

# =========================
# 0) USER CONFIG
# =========================
DATA_DIR = Path(r"E:\Beijing2024\MWR\cbh")
OUT_DIR  = DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

period1_start = "2024-06-06"
period1_end   = "2024-06-09"
period2_start = "2024-08-05"
period2_end   = "2024-08-10"

# True: 用 BJT(Asia/Shanghai) 截取 period + 做 1min
# False: 用 UTC 截取 period + 做 1min
USE_BJT_FOR_PERIODS = True

# 可选 QC（按你的科研口径自行开关）
DROP_CBH_EQ_10000_AS_NAN = False
DROP_CBH_EQ_0_AS_NAN     = False
DROP_WHEN_RAINFLAG_1     = False

OUT_CSV = OUT_DIR / "CBH_1min_period1_period2_merged_noTZ_interp.csv"

# =========================
# 1) PARSER
# =========================
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


# =========================
# 2) HELPERS
# =========================
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
    """
    对单个 period：
    - 截取（按 USE_BJT_FOR_PERIODS 选择 UTC/BJT 时间轴）
    - 同秒重复：cbh 取均值，rain_flag 取 max
    - 1min mean
    - 强制补齐 period 内完整分钟序列
    - cbh NaN 做线性插值（按时间，只填内部缺测，不跨边界）
    - 去掉时区（输出 naive datetime）
    """
    if USE_BJT_FOR_PERIODS:
        tcol = "timestamp_bjt"
        tz = "Asia/Shanghai"
    else:
        tcol = "timestamp_utc"
        tz = "UTC"

    p_start, p_end_excl = make_inclusive_day_range(start_date, end_date, tz)

    df = df_all[(df_all[tcol] >= p_start) & (df_all[tcol] < p_end_excl)].copy()
    if df.empty:
        # 返回一个空的、但结构一致的结果
        return pd.DataFrame(columns=["timestamp", "cbh_m_1min_mean", "rain_flag_1min_max"])

    df = apply_qc(df)

    df = df[[tcol, "cbh_m", "rain_flag"]].dropna(subset=[tcol]).sort_values(tcol)

    # 1) 同一秒重复 -> 聚合
    g = df.groupby(tcol, as_index=True)
    sec = pd.DataFrame({
        "cbh_m": g["cbh_m"].mean(),
        "rain_flag_max": g["rain_flag"].max()
    }).sort_index()

    # 2) 1 分钟聚合
    cbh_1min  = sec["cbh_m"].resample("1min").mean()
    rain_1min = sec["rain_flag_max"].resample("1min").max()

    out = pd.DataFrame({
        "cbh_m_1min_mean": cbh_1min,
        "rain_flag_1min_max": rain_1min
    })

    # 3) 只保留本 period 的完整分钟时间戳（避免生成两个 period 之间的大空档）
    full_index = pd.date_range(p_start, p_end_excl - pd.Timedelta(minutes=1), freq="1min", tz=tz)
    out = out.reindex(full_index)

    # 4) cbh 线性插值补 NaN（按时间，仅填内部缺测；不会跨 period 连接）
    out["cbh_m_1min_mean"] = out["cbh_m_1min_mean"].interpolate(
        method="time",
        limit_area="inside"
    )

    # 5) rain_flag 一般不做插值，缺测就当 0
    out["rain_flag_1min_max"] = out["rain_flag_1min_max"].fillna(0).astype(int)

    # 6) 去掉时区：tz-aware -> naive
    out = out.reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out["timestamp"].dt.tz_localize(None)

    return out

# =========================
# 3) MAIN
# =========================

files = sorted(DATA_DIR.glob("*.ASC")) + sorted(DATA_DIR.glob("*.asc"))
if not files:
    raise FileNotFoundError(f"No .ASC files found in {DATA_DIR}")

frames = []
for fp in files:
    d = parse_one_cbh_file(fp)
    if not d.empty:
        frames.append(d)

if not frames:
    raise RuntimeError("Parsed 0 valid rows from all files. Check file format.")

cbh_all = pd.concat(frames, ignore_index=True)

# 分别处理两个 period（关键：避免跨 period 插值/生成空档分钟）
p1 = one_period_to_1min(cbh_all, period1_start, period1_end)
p2 = one_period_to_1min(cbh_all, period2_start, period2_end)

out = pd.concat([p1, p2], ignore_index=True)
out = out.sort_values("timestamp").reset_index(drop=True).dropna()
out['cbh_m_1min_mean'] = out['cbh_m_1min_mean'].astype(float)

cbh = out
cbh = cbh.rename(columns={'cbh_m_1min_mean':  'CBH'})   
 
cbh['timestamp'] = pd.to_datetime(cbh['timestamp'])
cbh = cbh.set_index("timestamp")
cbh_1min = cbh.resample('1min').mean()

"""
这是做再分析做双线性插值的代码
"""

# ================== 基本设置 ==================
data_dir = r'E:\Beijing2024\ERA5'

# 只遍历当前目录下的 .nc 文件，不进入子文件夹
files = [
    f for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.nc')
]

target_lon = 116.3705
target_lat = 39.9745

target_vars = ['q', 'u', 'v', 'w', 't', 'r']  # 按你原来设定
Altitude = 49.0  # 举例：地面高度（m），你按真实高度改

all_dfs = []  # 存储每个文件的结果，最后再 concat

for file in files:
    file_path = os.path.join(data_dir, file)
    print(f"正在处理文件：{file_path}")

    # 1. 打开数据集
    ds = xr.open_dataset(file_path)
    variables = [var for var in ds.data_vars if var in target_vars]

    # 2. 原始经纬度
    raw_lats = ds.latitude.values  # 通常为降序
    raw_lons = ds.longitude.values

    # 3. 定位包围目标点的网格索引
    lat_idx = np.searchsorted(raw_lats[::-1], target_lat)
    lon_idx = np.searchsorted(raw_lons, target_lon)

    if lat_idx == 0 or lat_idx == len(raw_lats):
        raise ValueError("目标纬度超出范围")
    if lon_idx == 0 or lon_idx == len(raw_lons):
        raise ValueError("目标经度超出范围")

    surrounding_points = [
        (lat_idx-1, lon_idx-1),
        (lat_idx-1, lon_idx),
        (lat_idx,   lon_idx-1),
        (lat_idx,   lon_idx),
    ]
    adjusted_points = [
        (len(raw_lats)-1 - i, j) for i, j in surrounding_points
    ]

    # 4. 为插值准备“升序纬度”
    lats = raw_lats[::-1]
    lons = raw_lons

    # 5. 时间与层次
    times = ds.valid_time.values
    levels = ds.pressure_level.values

    index = pd.MultiIndex.from_product(
        [times, levels],
        names=['valid_time', 'pressure_level']
    )

    # ⚠️ 关键：每个文件先建一个骨架 DataFrame
    file_df = pd.DataFrame({
        'valid_time': index.get_level_values('valid_time'),
        'pressure_level': index.get_level_values('pressure_level')
    })

    # 6. 变量循环：在 file_df 中一列一列填数据
    for var in variables:
        print(f"  插值变量：{var}")

        # 维度调整：lat, lon, time, level 再反转纬度
        data = ds[var].transpose(
            'latitude', 'longitude', 'valid_time', 'pressure_level'
        ).values[::-1]

        # 设置插值器
        interpolator = RegularGridInterpolator(
            (lats, lons),
            data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # 对单点插值，结果形状：(time, level)
        result = interpolator([[target_lat, target_lon]]).squeeze(0)

        # 展平成一维，与 index 对齐
        file_df[var] = result.ravel()

    # 7. 这个文件的结果暂存
    all_dfs.append(file_df)

# 8. 所有文件拼接
final_df = pd.concat(all_dfs, ignore_index=True)

# 9. 计算派生量（要求已有 u, v, w, t, pressure_level）
if {'w', 't', 'pressure_level'}.issubset(final_df.columns):
    final_df['vws(10⁻² {m/s})'] = (
        mpcalc.vertical_velocity(
            final_df['w'].to_numpy() * units('Pa/s'),
            final_df['pressure_level'].to_numpy() * units.hPa,
            final_df['t'].to_numpy() * units.K,
            mixing_ratio=0 * units('kg/kg')
        ).m
    )

if {'u', 'v'}.issubset(final_df.columns):
    final_df['hws(m/s)'] = np.sqrt(
        np.square(final_df['u']) + np.square(final_df['v'])
    )
    final_df['hwd(deg)'] = np.mod(180+ np.degrees(np.arctan2(final_df['u'],final_df['v'])),360)

# 标准大气高度（相对海平面），再减去站点海拔得到 AGL
final_df['height'] = (
    1000.0
    * mpcalc.pressure_to_height_std(
        final_df['pressure_level'].to_numpy() * units.hPa
    ).m
    - Altitude
)

# 10. 时间处理与列排序
final_df['valid_time'] = pd.to_datetime(final_df['valid_time'])
final_df['BJT'] = (final_df['valid_time'] + pd.Timedelta(hours=8)).dt.strftime(
    '%Y/%m/%d %H:%M:%S'
)

cols = ['BJT', 'valid_time', 'pressure_level'] + sorted(
    [c for c in final_df.columns if c not in ['BJT', 'valid_time', 'pressure_level']],
    key=lambda x: x.lower()
)
final_df = final_df[cols]

print("\n插值结果示例：")
print(final_df.head())
print(f"\n总行数：{len(final_df)}")
print(f"列：{final_df.columns.tolist()}")

# 8. 所有文件拼接
final_df = pd.concat(all_dfs, ignore_index=True)

# 9. 计算派生量（要求已有 u, v, w, t, pressure_level）
if {'w', 't', 'pressure_level'}.issubset(final_df.columns):
    final_df['vws(10⁻² {m/s})'] = (
        mpcalc.vertical_velocity(
            final_df['w'].to_numpy() * units('Pa/s'),
            final_df['pressure_level'].to_numpy() * units.hPa,
            final_df['t'].to_numpy() * units.K,
            mixing_ratio=0 * units('kg/kg')
        ).m
    )

if {'u', 'v'}.issubset(final_df.columns):
    final_df['hws(m/s)'] = np.sqrt(
        np.square(final_df['u']) + np.square(final_df['v'])
    )
    final_df['hwd(deg)'] = np.mod(180+ np.degrees(np.arctan2(final_df['u'],final_df['v'])),360)

# 标准大气高度（相对海平面），再减去站点海拔得到 AGL
final_df['height'] = (
    1000.0
    * mpcalc.pressure_to_height_std(
        final_df['pressure_level'].to_numpy() * units.hPa
    ).m
    - Altitude
)

# 10. 时间处理与列排序
final_df['valid_time'] = pd.to_datetime(final_df['valid_time'])
final_df['BJT'] = (final_df['valid_time'] + pd.Timedelta(hours=8)).dt.strftime(
    '%Y/%m/%d %H:%M:%S'
)

cols = ['BJT', 'valid_time', 'pressure_level'] + sorted(
    [c for c in final_df.columns if c not in ['BJT', 'valid_time', 'pressure_level']],
    key=lambda x: x.lower()
)
final_df = final_df[cols]

print("\n插值结果示例：")
print(final_df.head())
print(f"\n总行数：{len(final_df)}")
print(f"列：{final_df.columns.tolist()}")
# ============================================================
# 11. ERA5：只保留温度和比湿，并将 q 从 kg/kg 转为 g/kg
# ============================================================
if 'q' not in final_df.columns or 't' not in final_df.columns:
    raise KeyError("ERA5 结果中没有找到 q 或 t，请检查 target_vars 和插值过程。")

# 比湿单位转换：kg/kg -> g/kg
final_df['q (g/kg)'] = final_df['q'] * 1000.0

# 只保留需要的列（可以按需加上 'height'）
era5_tq = final_df[['BJT', 'valid_time', 'pressure_level', 't', 'q (g/kg)']].copy()
era5_tq.rename(columns={'t': 'Temp (K)'}, inplace=True)

print("\nERA5 温度 & 比湿示例：")
print(era5_tq.head())
print(f"ERA5 T-q 数据总行数：{len(era5_tq)}")

# -*- coding: utf-8 -*-
"""
CMA-RA 再分析资料插值到单点（温度 T & 比湿 q）

功能：
1) 遍历 PARENT_DIR 下各个变量子文件夹（如 Temperature, Specific Humidity 等）中的 .grib2 文件
2) 对每个变量，在 isobaricInhPa 层上插值到指定经纬度 (TARGET_LON, TARGET_LAT)
3) 所有变量插值结果暂存到 data['常规变量'] / data['特殊变量'] 结构中
4) 最后从 “常规变量” 中抽取温度 T 和比湿 q：
   - q 单位从 kg/kg 转为 g/kg
   - 生成最终表 cra_tq：每个时间 × 每个气压层的 Temp (K) 与 q (g/kg)
"""


# ===================== 配置区 =====================
PARENT_DIR = r"E:\Beijing2024\CMA-RA\Reanalysis"  # 各变量子文件夹的上级目录
OUTPUT_DIR = r"E:\Beijing2024\CMA-RA"              # 输出目录
TARGET_LON = 116.3705
TARGET_LAT = 39.9745

# 输出文件名（可按需修改）
EXCEL_NAME = f"CRA_T_q_interp_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 关闭 cfgrib 索引文件
os.environ["CFGRIB_DISABLE_INDEX"] = "True"

# ==================== 核心处理逻辑 ====================
# data 用来存所有插值结果（跟你原来一样）
data = {
    '常规变量': {},
    '特殊变量': {}
}

# 遍历变量子文件夹
variable_dirs = [
    d for d in os.listdir(PARENT_DIR)
    if os.path.isdir(os.path.join(PARENT_DIR, d))
]

for variable_dir in variable_dirs:
    input_dir = os.path.join(PARENT_DIR, variable_dir)

    # 精确筛选 .grib2 文件（排除 .grib2.xxx）
    files = sorted([
        f for f in glob(os.path.join(input_dir, "*"))
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() == ".grib2"
    ])

    if not files:
        print(f"⚠️ 无有效 GRIB2 文件，跳过目录: {variable_dir}")
        continue

    try:
        # 只打开第一文件获取网格信息和变量名
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with xr.open_dataset(
                files[0],
                engine="cfgrib",
                decode_timedelta=True,
                backend_kwargs={
                    "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
                    "indexpath": False
                }
            ) as ds:
                # 识别变量名（通常 Temperature 文件就是 t，Specific Humidity 是 q 等）
                var_name = list(ds.data_vars)[0]
                print(f"✔️ 正在处理变量目录: {variable_dir} | 识别到变量名: {var_name}")

                unit = ds[var_name].attrs.get("units", "")

                # 处理坐标（保持你原来的写法：unique + sort）
                lons = np.unique(ds.longitude.values)
                lats = np.unique(ds.latitude.values)
                lons.sort()
                lats.sort()

                # 边界检查
                if not (lons[0] <= TARGET_LON <= lons[-1] and
                        lats[0] <= TARGET_LAT <= lats[-1]):
                    print(f"🚫 坐标越界，跳过变量目录: {variable_dir}")
                    continue

                # 计算网格索引（用于打印信息）
                i_lon = max(0, np.searchsorted(lons, TARGET_LON, side="right") - 1)
                i_lat = np.searchsorted(lats, TARGET_LAT, side="left")

    except Exception as e:
        print(f"❌ 初始化失败: {variable_dir} | {str(e)}")
        continue

    # ================== 取消缩放：SCALE_FACTOR 固定为 1 ==================
    SCALE_FACTOR = 1
    print(f"🔄 CRA 不做缩放, SCALE_FACTOR = {SCALE_FACTOR} (变量: {var_name})")

    # ================== 单位 & 列名处理 ==================
    if var_name == "r":        # 相对湿度
        scaled_unit = "%"
        col_name = f"rh ({scaled_unit})"
    elif var_name == "t":      # 温度
        scaled_unit = "K"
        col_name = f"Temp ({scaled_unit})"
    else:                       # 其他：保持原单位
        scaled_unit = unit
        col_name = f"{var_name} ({scaled_unit})" if unit else var_name

    # ================== 处理该变量所有文件 ==================
    for file in files:
        try:
            filename = os.path.basename(file)

            # 根据文件名解析时间（你原来的 27:37 保持不变）
            # 例如：xxxxxx_isobaric_2024080100.grib2 这样的结构
            time_str = filename[27:37]  # ← 若文件名结构有变，这里再调整
            dt = datetime.strptime(time_str, "%Y%m%d%H")

            with xr.open_dataset(
                file,
                engine="cfgrib",
                decode_timedelta=True,
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}
            ) as ds:
                # 对每个等压面做插值
                for level in ds.isobaricInhPa.values:
                    # 提取该层二维场
                    grid_data = ds[var_name].sel(isobaricInhPa=level).values

                    # 应用缩放因子（现在恒等于 1，因此不会改变数据）
                    if SCALE_FACTOR != 1:
                        grid_data = grid_data * SCALE_FACTOR

                    # 插值到单点
                    interpolator = RegularGridInterpolator(
                        (lats, lons), grid_data, bounds_error=False
                    )
                    value = interpolator([[TARGET_LAT, TARGET_LON]])[0]

                    # 确定分组：保留你原来的逻辑
                    sheet_group = "特殊变量" if variable_dir in [
                        # "Vertical velocity",   # 对应 w 变量（目前注释）
                        "Cloud Ice",            # 对应 cice 变量
                        "Cloud Mixing Ratio"    # 对应 clwmr 变量
                    ] else "常规变量"

                    timestamp = dt.strftime("%Y-%m-%d %H:00")
                    level_key = (timestamp, int(level))

                    # 初始化该时间层记录
                    if level_key not in data[sheet_group]:
                        data[sheet_group][level_key] = {
                            "timestamps": timestamp,
                            "Level (hPa)": int(level)
                        }

                    # 写入当前变量的插值值
                    data[sheet_group][level_key][col_name] = value

        except Exception as e:
            print(f"❌ 处理失败: {filename} | {str(e)}")
            continue

    # 打印网格信息（保持你原来的输出）
    print(f"\n🔍 完成变量目录: {variable_dir}")
    print(f"网格坐标索引: 经度[{i_lon}:{i_lon+1}] 纬度[{i_lat-1}:{i_lat}]")
    print("实际坐标点:")
    print(f"  NW: {lats[i_lat]:.4f}°N, {lons[i_lon]:.4f}°E")
    print(f"  NE: {lats[i_lat]:.4f}°N, {lons[i_lon+1]:.4f}°E")
    print(f"  SW: {lats[i_lat-1]:.4f}°N, {lons[i_lon]:.4f}°E")
    print(f"  SE: {lats[i_lat-1]:.4f}°N, {lons[i_lon+1]:.4f}°E\n")

# ==================== 构建最终的 T & q 表 ====================
if not data["常规变量"]:
    raise RuntimeError("常规变量数据为空，可能没找到任何变量或插值失败。")

# 将常规变量字典合并为 DataFrame
df_common = pd.DataFrame(list(data["常规变量"].values())).fillna(0)

# 排序：按时间升序，气压层按从高到低（可以按需改成 [True, True]）
df_common = df_common.sort_values(
    by=["timestamps", "Level (hPa)"],
    ascending=[True, False]
)

# --------- 寻找温度列（Temp (K)）---------
if "Temp (K)" in df_common.columns:
    t_col = "Temp (K)"
else:
    t_candidates = [
        c for c in df_common.columns
        if ("temp" in c.lower()) and ("(k" in c.lower())
    ]
    if not t_candidates:
        raise KeyError("CRA 数据中找不到温度列，请检查温度变量目录与列名。")
    t_col = t_candidates[0]

# --------- 寻找比湿列（q，单位 kg/kg）---------
q_candidates = [
    c for c in df_common.columns
    if c.strip().lower().startswith("q ")
       or c.strip().lower().startswith("q(")
       or c.strip().lower() == "q"
]
if not q_candidates:
    raise KeyError("CRA 数据中找不到比湿 q 列，请检查 'Specific Humidity' 变量目录。")

q_col = q_candidates[0]

# --------- 单位转换：kg/kg → g/kg ---------
df_common["q (g/kg)"] = df_common[q_col] * 1000.0

# --------- 构建最终结果表：cra_tq ---------
cra_tq = df_common[["timestamps", "Level (hPa)", t_col, "q (g/kg)"]].copy()
cra_tq.rename(columns={t_col: "Temp (K)"}, inplace=True)

# 可选：增加 datetime 与 BJT
cra_tq["time"] = pd.to_datetime(cra_tq["timestamps"])
cra_tq["BJT"] = (cra_tq["time"] + pd.Timedelta(hours=8)).dt.strftime(
    "%Y/%m/%d %H:%M:%S"
)

# 调整列顺序：BJT, time, Level, Temp, q
cra_tq = cra_tq[["BJT", "time", "Level (hPa)", "Temp (K)", "q (g/kg)"]]

print("\nCRA 温度 & 比湿示例：")
print(cra_tq.head())
print(f"CRA T-q 数据总行数：{len(cra_tq)}")

# ==================== 导出 Excel ====================
out_path = os.path.join(OUTPUT_DIR, EXCEL_NAME)
#cra_tq.to_excel(out_path, index=False)
print(f"\n✅ CRA 温度 & 比湿表已保存：{out_path}")
######################################################################################################
######################################################################################################
######################################################################################################
"""
ERA5 & CMA-RA 再分析资料
四点平均 + 最近格点 的 温度 (t) 和比湿 (q) 提取脚本

功能：
1. ERA5 部分
   - 遍历 ERA5_DIR 下所有 .nc 文件
   - 在目标点所在网格的 4 个格点上取平均（四点平均）
   - 同时提取距离目标经纬度最近的格点值（最近点）
   - 仅处理变量 t (温度) 和 q (比湿)
   - q 从 kg/kg 转为 g/kg
   - 输出：
        era5_tq_4pt   : 四点平均 T & q
        era5_tq_near  : 最近格点 T & q

2. CMA-RA (CRA) 部分
   - 遍历 CRA_PARENT_DIR 下各变量子文件夹（如 Temperature, Specific Humidity）
   - 仅处理 var_name 为 't' 和 'q' 的变量
   - 同时计算四点平均与最近格点的 T & q
   - q 从 kg/kg 转为 g/kg
   - 输出：
        cra_tq_4pt    : 四点平均 T & q
        cra_tq_near   : 最近格点 T & q
"""

# ================== 公共配置 ==================
TARGET_LON = 116.3705
TARGET_LAT = 39.9745

# ------------------ ERA5 配置 ------------------
ERA5_DIR = r"E:\Beijing2024\ERA5"
ERA5_OUT_EXCEL_4PT   = os.path.join(
    ERA5_DIR, f"ERA5_T_q_4point_mean_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
)
ERA5_OUT_EXCEL_NEAR  = os.path.join(
    ERA5_DIR, f"ERA5_T_q_nearest_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
)

# ------------------ CRA 配置 ------------------
CRA_PARENT_DIR = r"E:\Beijing2024\CMA-RA\Reanalysis"
CRA_OUT_DIR    = r"E:\Beijing2024\CMA-RA"

CRA_OUT_EXCEL_4PT  = os.path.join(
    CRA_OUT_DIR, f"CRA_T_q_4point_mean_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
)
CRA_OUT_EXCEL_NEAR = os.path.join(
    CRA_OUT_DIR, f"CRA_T_q_nearest_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
)

os.makedirs(CRA_OUT_DIR, exist_ok=True)

# ============================================================================
# 1. ERA5：四点平均 + 最近格点 温度 & 比湿
# ============================================================================
print("\n================ ERA5 四点平均 + 最近格点 提取 ==================")

era5_nc_files = [
    f for f in os.listdir(ERA5_DIR)
    if os.path.isfile(os.path.join(ERA5_DIR, f)) and f.endswith(".nc")
]
era5_nc_files.sort()

era5_dfs_4pt  = []   # 存放每个文件的四点平均结果
era5_dfs_near = []   # 存放每个文件的最近格点结果

for fname in era5_nc_files:
    file_path = os.path.join(ERA5_DIR, fname)
    print(f"=== ERA5: 处理文件 {file_path}")

    with xr.open_dataset(file_path) as ds:
        # 只关心 t 和 q
        vars_to_use = [v for v in ["t", "q"] if v in ds.data_vars]
        if not vars_to_use:
            print("  ⚠️ 该文件不含 t 或 q，跳过。")
            continue

        # 取经纬度坐标
        raw_lats = ds.latitude.values  # 可能是升序或降序
        raw_lons = ds.longitude.values

        # ---- 处理纬度（无论升/降序）----
        if raw_lats[0] <= raw_lats[-1]:
            lats_asc = raw_lats.copy()
        else:
            lats_asc = raw_lats[::-1]

        lat_idx = np.searchsorted(lats_asc, TARGET_LAT)
        if lat_idx == 0 or lat_idx == len(lats_asc):
            raise ValueError("ERA5：目标纬度超出范围")

        lat_low_val  = float(lats_asc[lat_idx - 1])
        lat_high_val = float(lats_asc[lat_idx])

        # 找到原始维度中的索引
        lat_low_idx  = int(np.where(np.isclose(raw_lats, lat_low_val))[0][0])
        lat_high_idx = int(np.where(np.isclose(raw_lats, lat_high_val))[0][0])

        # 最近纬度索引（在上/下两个格点里挑更近的）
        if abs(raw_lats[lat_low_idx] - TARGET_LAT) <= abs(raw_lats[lat_high_idx] - TARGET_LAT):
            lat_near_idx = lat_low_idx
        else:
            lat_near_idx = lat_high_idx

        # ---- 处理经度 ----
        if raw_lons[0] <= raw_lons[-1]:
            lons_asc = raw_lons.copy()
        else:
            lons_asc = raw_lons[::-1]

        lon_idx = np.searchsorted(lons_asc, TARGET_LON)
        if lon_idx == 0 or lon_idx == len(lons_asc):
            raise ValueError("ERA5：目标经度超出范围")

        lon_left_val  = float(lons_asc[lon_idx - 1])
        lon_right_val = float(lons_asc[lon_idx])

        lon_left_idx  = int(np.where(np.isclose(raw_lons, lon_left_val))[0][0])
        lon_right_idx = int(np.where(np.isclose(raw_lons, lon_right_val))[0][0])

        # 最近经度索引
        if abs(raw_lons[lon_left_idx] - TARGET_LON) <= abs(raw_lons[lon_right_idx] - TARGET_LON):
            lon_near_idx = lon_left_idx
        else:
            lon_near_idx = lon_right_idx

        # 时间 & 层次
        times  = ds.valid_time.values
        levels = ds.pressure_level.values

        index = pd.MultiIndex.from_product(
            [times, levels],
            names=["valid_time", "pressure_level"]
        )

        # 两个骨架 DataFrame：四点平均 & 最近格点
        df_4pt  = pd.DataFrame({
            "valid_time": index.get_level_values("valid_time"),
            "pressure_level": index.get_level_values("pressure_level"),
        })
        df_near = df_4pt.copy()

        # ---- 对 t, q 做四点平均 + 最近格点 ----
        for var in vars_to_use:
            print(f"  ERA5: 变量 {var} 四点平均 + 最近点")

            # 调整维度至 (latitude, longitude, valid_time, pressure_level)
            da = ds[var].transpose("latitude", "longitude", "valid_time", "pressure_level")

            # 四点平均
            sub_4pt = da.isel(
                latitude=[lat_low_idx, lat_high_idx],
                longitude=[lon_left_idx, lon_right_idx]
            )
            mean_da = sub_4pt.mean(dim=("latitude", "longitude"))  # (valid_time, pressure_level)
            values_4pt = mean_da.values.reshape(-1)

            # 最近格点
            sub_near   = da.isel(latitude=lat_near_idx, longitude=lon_near_idx)
            values_near = sub_near.values.reshape(-1)

            if var == "t":
                df_4pt["Temp (K)"]  = values_4pt
                df_near["Temp (K)"] = values_near
            elif var == "q":
                df_4pt["q (kg/kg)"]  = values_4pt
                df_near["q (kg/kg)"] = values_near

        # 单位换算：q -> g/kg
        if "q (kg/kg)" in df_4pt.columns:
            df_4pt["q (g/kg)"]  = df_4pt["q (kg/kg)"] * 1000.0
        if "q (kg/kg)" in df_near.columns:
            df_near["q (g/kg)"] = df_near["q (kg/kg)"] * 1000.0

        # 时间格式 & BJT
        for df in [df_4pt, df_near]:
            df["valid_time"] = pd.to_datetime(df["valid_time"])
            df["BJT"] = (df["valid_time"] + pd.Timedelta(hours=8)).dt.strftime(
                "%Y/%m/%d %H:%M:%S"
            )

        # 选择需要的列
        def _select_cols(df):
            cols = ["BJT", "valid_time", "pressure_level"]
            if "Temp (K)" in df.columns:
                cols.append("Temp (K)")
            if "q (g/kg)" in df.columns:
                cols.append("q (g/kg)")
            return df[cols]

        era5_dfs_4pt.append(_select_cols(df_4pt))
        era5_dfs_near.append(_select_cols(df_near))

# 合并 ERA5 所有文件
if era5_dfs_4pt:
    era5_tq_4pt = pd.concat(era5_dfs_4pt, ignore_index=True)
    era5_tq_4pt = era5_tq_4pt.sort_values(
        by=["valid_time", "pressure_level"]
    ).reset_index(drop=True)

    print("\n[ERA5] 四点平均 T & q 示例：")
    print(era5_tq_4pt.head())
    print(f"[ERA5] 四点平均总行数：{len(era5_tq_4pt)}")

    # 导出 Excel（如需要，取消注释）
    # era5_tq_4pt.to_excel(ERA5_OUT_EXCEL_4PT, index=False)
    print(f"[ERA5] 四点平均如需导出，可用：era5_tq_4pt.to_excel(r'{ERA5_OUT_EXCEL_4PT}', index=False)")

if era5_dfs_near:
    era5_tq_near = pd.concat(era5_dfs_near, ignore_index=True)
    era5_tq_near = era5_tq_near.sort_values(
        by=["valid_time", "pressure_level"]
    ).reset_index(drop=True)

    print("\n[ERA5] 最近格点 T & q 示例：")
    print(era5_tq_near.head())
    print(f"[ERA5] 最近格点总行数：{len(era5_tq_near)}")

    # 导出 Excel（如需要，取消注释）
    # era5_tq_near.to_excel(ERA5_OUT_EXCEL_NEAR, index=False)
    print(f"[ERA5] 最近格点如需导出，可用：era5_tq_near.to_excel(r'{ERA5_OUT_EXCEL_NEAR}', index=False)")

if not era5_dfs_4pt and not era5_dfs_near:
    print("\n[ERA5] 未生成任何数据，请检查 ERA5_DIR 和变量名。")

# ============================================================================
# 2. CMA-RA (CRA)：四点平均 + 最近格点 温度 & 比湿
# ============================================================================
print("\n================ CMA-RA (CRA) 四点平均 + 最近格点 提取 ==================")

os.environ["CFGRIB_DISABLE_INDEX"] = "True"

cra_records_4pt  = {}  # key: (time, level) -> dict (四点平均)
cra_records_near = {}  # key: (time, level) -> dict (最近点)

variable_dirs = [
    d for d in os.listdir(CRA_PARENT_DIR)
    if os.path.isdir(os.path.join(CRA_PARENT_DIR, d))
]

for variable_dir in variable_dirs:
    input_dir = os.path.join(CRA_PARENT_DIR, variable_dir)

    files = sorted([
        f for f in glob(os.path.join(input_dir, "*"))
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() == ".grib2"
    ])
    if not files:
        print(f"⚠️ CRA: 目录无 GRIB2 文件，跳过 {variable_dir}")
        continue

    # 先用首个文件探测变量名 & 网格信息
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with xr.open_dataset(
                files[0],
                engine="cfgrib",
                decode_timedelta=True,
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}
            ) as ds0:
                var_name = list(ds0.data_vars)[0]
                print(f"=== CRA: 变量目录 {variable_dir}, 识别到变量 {var_name}")

                # 只处理 t 和 q
                if var_name not in ("t", "q"):
                    print("    -> 不是 t 或 q，跳过该目录。")
                    continue

                lat_arr = ds0.latitude.values
                lon_arr = ds0.longitude.values

                # 纬度（升/降序）
                if lat_arr[0] <= lat_arr[-1]:
                    lats_asc = lat_arr.copy()
                else:
                    lats_asc = lat_arr[::-1]

                lat_idx = np.searchsorted(lats_asc, TARGET_LAT)
                if lat_idx == 0 or lat_idx == len(lats_asc):
                    print("    -> 目标纬度超出 CRA 范围，跳过该目录。")
                    continue

                lat_low_val  = float(lats_asc[lat_idx - 1])
                lat_high_val = float(lats_asc[lat_idx])

                lat_low_idx  = int(np.where(np.isclose(lat_arr, lat_low_val))[0][0])
                lat_high_idx = int(np.where(np.isclose(lat_arr, lat_high_val))[0][0])

                # 最近纬度索引
                if abs(lat_arr[lat_low_idx] - TARGET_LAT) <= abs(lat_arr[lat_high_idx] - TARGET_LAT):
                    lat_near_idx = lat_low_idx
                else:
                    lat_near_idx = lat_high_idx

                # 经度
                if lon_arr[0] <= lon_arr[-1]:
                    lons_asc = lon_arr.copy()
                else:
                    lons_asc = lon_arr[::-1]

                lon_idx = np.searchsorted(lons_asc, TARGET_LON)
                if lon_idx == 0 or lon_idx == len(lons_asc):
                    print("    -> 目标经度超出 CRA 范围，跳过该目录。")
                    continue

                lon_left_val  = float(lons_asc[lon_idx - 1])
                lon_right_val = float(lons_asc[lon_idx])

                lon_left_idx  = int(np.where(np.isclose(lon_arr, lon_left_val))[0][0])
                lon_right_idx = int(np.where(np.isclose(lon_arr, lon_right_val))[0][0])

                # 最近经度索引
                if abs(lon_arr[lon_left_idx] - TARGET_LON) <= abs(lon_arr[lon_right_idx] - TARGET_LON):
                    lon_near_idx = lon_left_idx
                else:
                    lon_near_idx = lon_right_idx

    except Exception as e:
        print(f"❌ CRA: 初始化失败 {variable_dir} | {e}")
        continue

    # ---- 处理该变量的所有时间文件 ----
    for file in files:
        filename = os.path.basename(file)

        # 根据你之前的命名规则：时间在 filename[27:37]
        try:
            time_str = filename[27:37]
            dt = datetime.strptime(time_str, "%Y%m%d%H")
        except Exception:
            print(f"    ⚠️ 无法从文件名解析时间，跳过：{filename}")
            continue

        try:
            with xr.open_dataset(
                file,
                engine="cfgrib",
                decode_timedelta=True,
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}
            ) as ds:
                for level in ds.isobaricInhPa.values:
                    da_level = ds[var_name].sel(isobaricInhPa=level)

                    # 四点平均
                    sub_4pt = da_level.isel(
                        latitude=[lat_low_idx, lat_high_idx],
                        longitude=[lon_left_idx, lon_right_idx]
                    )
                    val_4pt = float(sub_4pt.mean().values)

                    # 最近格点
                    val_near = float(
                        da_level.isel(latitude=lat_near_idx, longitude=lon_near_idx).values
                    )

                    key = (dt, int(level))

                    # 初始化记录
                    if key not in cra_records_4pt:
                        cra_records_4pt[key] = {
                            "time": dt,
                            "Level (hPa)": int(level)
                        }
                    if key not in cra_records_near:
                        cra_records_near[key] = {
                            "time": dt,
                            "Level (hPa)": int(level)
                        }

                    if var_name == "t":
                        cra_records_4pt[key]["Temp (K)"]  = val_4pt
                        cra_records_near[key]["Temp (K)"] = val_near
                    elif var_name == "q":
                        cra_records_4pt[key]["q (kg/kg)"]  = val_4pt
                        cra_records_near[key]["q (kg/kg)"] = val_near

        except Exception as e:
            print(f"    ❌ CRA: 处理失败 {filename} | {e}")
            continue

# ================== CRA 汇总 & 单位转换 ==================
if cra_records_4pt:
    cra_df_4pt = pd.DataFrame(list(cra_records_4pt.values()))
    cra_df_4pt = cra_df_4pt.sort_values(
        by=["time", "Level (hPa)"],
        ascending=[True, False]
    ).reset_index(drop=True)

    # q: kg/kg -> g/kg
    if "q (kg/kg)" in cra_df_4pt.columns:
        cra_df_4pt["q (g/kg)"] = cra_df_4pt["q (kg/kg)"] * 1000.0

    cra_df_4pt["BJT"] = (cra_df_4pt["time"] + pd.Timedelta(hours=8)).dt.strftime(
        "%Y/%m/%d %H:%M:%S"
    )

    cols_4pt = ["BJT", "time", "Level (hPa)"]
    if "Temp (K)" in cra_df_4pt.columns:
        cols_4pt.append("Temp (K)")
    if "q (g/kg)" in cra_df_4pt.columns:
        cols_4pt.append("q (g/kg)")

    cra_tq_4pt = cra_df_4pt[cols_4pt]

    print("\n[CRA] 四点平均 T & q 示例：")
    print(cra_tq_4pt.head())
    print(f"[CRA] 四点平均总行数：{len(cra_tq_4pt)}")

    # 导出 Excel（如需要，取消注释）
    # cra_tq_4pt.to_excel(CRA_OUT_EXCEL_4PT, index=False)
    print(f"[CRA] 四点平均如需导出，可用：cra_tq_4pt.to_excel(r'{CRA_OUT_EXCEL_4PT}', index=False)")

if cra_records_near:
    cra_df_near = pd.DataFrame(list(cra_records_near.values()))
    cra_df_near = cra_df_near.sort_values(
        by=["time", "Level (hPa)"],
        ascending=[True, False]
    ).reset_index(drop=True)

    if "q (kg/kg)" in cra_df_near.columns:
        cra_df_near["q (g/kg)"] = cra_df_near["q (kg/kg)"] * 1000.0

    cra_df_near["BJT"] = (cra_df_near["time"] + pd.Timedelta(hours=8)).dt.strftime(
        "%Y/%m/%d %H:%M:%S"
    )

    cols_near = ["BJT", "time", "Level (hPa)"]
    if "Temp (K)" in cra_df_near.columns:
        cols_near.append("Temp (K)")
    if "q (g/kg)" in cra_df_near.columns:
        cols_near.append("q (g/kg)")

    cra_tq_near = cra_df_near[cols_near]

    print("\n[CRA] 最近格点 T & q 示例：")
    print(cra_tq_near.head())
    print(f"[CRA] 最近格点总行数：{len(cra_tq_near)}")

    # 导出 Excel（如需要，取消注释）
    # cra_tq_near.to_excel(CRA_OUT_EXCEL_NEAR, index=False)
    print(f"[CRA] 最近格点如需导出，可用：cra_tq_near.to_excel(r'{CRA_OUT_EXCEL_NEAR}', index=False)")

if not cra_records_4pt and not cra_records_near:
    print("\n[CRA] 未生成任何数据，请检查 CRA_PARENT_DIR 和变量目录。")

# ============================================================
# ERA5 垂直插值：200–1000 hPa，每 25 hPa
# 已有层不改值，禁止外推
# 依赖：era5_tq_4pt, era5_tq_near 已经存在
# ============================================================

TARGET_PRESSURES = np.arange(300, 850 + 1, 25)  # 200, 225, ..., 1000 hPa

def vertical_interp_pressure(df, target_levels):
    """
    对给定的 ERA5 T-q 表按 pressure_level 做垂直插值：
    - 每个 valid_time 单独插值
    - 对已有层不改值，只填补缺失层
    - 线性插值，禁止外推（limit_area='inside'）
    df 需要至少包含：
      ['valid_time', 'pressure_level', 'Temp (K)', 'q (g/kg)', 'BJT']
    """
    required = {'valid_time', 'pressure_level', 'Temp (K)', 'q (g/kg)', 'BJT'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"插值函数需要的列缺失: {missing}")

    out_list = []

    # 确保 pressure_level 为数值型
    df = df.copy()
    df['pressure_level'] = pd.to_numeric(df['pressure_level'])

    for vt, g in df.groupby('valid_time'):
        # 按气压从小到大排序（或你想从大到小都行，关键是单调）
        g = g.sort_values('pressure_level')

        # 同一时间内，保证每个 pressure_level 只有一条（如果有重复，保留第一条）
        g = g.drop_duplicates(subset='pressure_level', keep='first')

        # 目标层索引
        idx = pd.Index(target_levels, name='pressure_level')

        base = pd.DataFrame(index=idx)

        # 对 Temp (K) 和 q (g/kg) 分别插值
        for col in ['Temp (K)', 'q (g/kg)']:
            s = g.set_index('pressure_level')[col]

            # 先把已有层对齐到目标层，其余为 NaN
            s2 = s.reindex(idx)

            # 线性插值，禁止外推（只在现有数据范围内部插）
            s2_interp = s2.interpolate(
                method='index',      # 用气压层值当作“坐标”
                limit_area='inside'  # 只在内部插值，边界不外推
            )

            base[col] = s2_interp

        # 时间和 BJT 都是 group 内常数
        base['valid_time'] = vt
        base['BJT'] = g['BJT'].iloc[0]

        base = base.reset_index()  # 把 pressure_level 从索引变成列
        out_list.append(base)

    result = pd.concat(out_list, ignore_index=True)

    # 列顺序
    cols_order = ['BJT', 'valid_time', 'pressure_level', 'Temp (K)', 'q (g/kg)']
    result = result[cols_order]

    return result

# 只有当原始表存在时才做插值
era5_tq_4pt_interp = None
era5_tq_near_interp = None

if isinstance(era5_tq_4pt, pd.DataFrame) and not era5_tq_4pt.empty:
    era5_tq_4pt_interp = vertical_interp_pressure(era5_tq_4pt, TARGET_PRESSURES)
    print("\n[ERA5] 四点平均插值后示例：")
    print(era5_tq_4pt_interp.head())

if isinstance(era5_tq_near, pd.DataFrame) and not era5_tq_near.empty:
    era5_tq_near_interp = vertical_interp_pressure(era5_tq_near, TARGET_PRESSURES)
    print("\n[ERA5] 最近格点插值后示例：")
    print(era5_tq_near_interp.head())


# 首先确保时间列是datetime格式
def convert_to_datetime(df):
    """使用.loc进行安全的赋值操作"""
    df = df.copy()  # 仍然建议先复制
    
    if 'BJT' in df.columns:
        df.loc[:, 'BJT'] = pd.to_datetime(df['BJT'])
   
    return df

# 转换所有DataFrame的时间列
cra_tq_near = convert_to_datetime(cra_tq_near)
cra_tq_4pt = convert_to_datetime(cra_tq_4pt)
era5_tq_4pt = convert_to_datetime(era5_tq_4pt_interp)
era5_tq_near = convert_to_datetime(era5_tq_near_interp)

# 定义时间段
period1_start = '2024-06-06'
period1_end = '2024-06-09'
period2_start = '2024-08-05'
period2_end = '2024-08-11'

def filter_time_periods(df, period1_start, period1_end, period2_start, period2_end):
    """
    筛选两个时间段的数据。
    
    # 关键修改：将输入的字符串转换为 Pandas 的 Timestamp 对象
    """
    # 将输入的时间字符串转换为 Pandas 的 Timestamp 对象，以便进行比较
    period1_start_ts = pd.to_datetime(period1_start)
    period1_end_ts = pd.to_datetime(period1_end)
    period2_start_ts = pd.to_datetime(period2_start)
    period2_end_ts = pd.to_datetime(period2_end)

    # 使用转换后的 Timestamp 对象进行筛选
    mask = (
        ((df['BJT'] >= period1_start_ts) & (df['BJT'] <= period1_end_ts)) |
        ((df['BJT'] >= period2_start_ts) & (df['BJT'] <= period2_end_ts))
    )
    return df[mask]

# 筛选所有DataFrame
cra_tq_near_filtered = filter_time_periods(cra_tq_near, period1_start, period1_end, period2_start, period2_end)
cra_tq_4pt_filtered = filter_time_periods(cra_tq_4pt, period1_start, period1_end, period2_start, period2_end)
era5_tq_4pt_filtered = filter_time_periods(era5_tq_4pt, period1_start, period1_end, period2_start, period2_end)
era5_tq_near_filtered = filter_time_periods(era5_tq_near, period1_start, period1_end, period2_start, period2_end)

# ============================================================
# ERA5 四个结果表：按 BJT 升序、pressure_level 降序 排序
# ============================================================
def sort_era5_df(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    # 确保 pressure_level 为数值型
    df['pressure_level'] = pd.to_numeric(df['pressure_level'])
    # 按 BJT 升序、pressure_level 降序排序
    df = df.sort_values(
        by=['BJT', 'pressure_level'],
        ascending=[True, False]
    ).reset_index(drop=True)
    return df

era5_tq_4pt_filtered   = sort_era5_df(era5_tq_4pt_filtered )
era5_tq_near_filtered  = sort_era5_df(era5_tq_near_filtered )

print("\n[ERA5] 四个结果表已经按 BJT 升序、pressure_level 降序 排序完成。")
cra_tq_near_filtered = cra_tq_near_filtered[(cra_tq_near_filtered['Level (hPa)']>=300)&(cra_tq_near_filtered['Level (hPa)']<=850)]

cra_tq_4pt_filtered = cra_tq_4pt_filtered[(cra_tq_4pt_filtered['Level (hPa)']>=300)&(cra_tq_4pt_filtered['Level (hPa)']<=850)]

# with pd.ExcelWriter('E:\\Beijing2024\\再分析nearest_4pt ave.xlsx', engine='openpyxl') as writer:       
#     cra_tq_near_filtered[(cra_tq_near_filtered['Level (hPa)']>=300)&(cra_tq_near_filtered['Level (hPa)']<=850)].to_excel(writer, sheet_name='cra_nearest', index=False)
#     era5_tq_near_filtered.to_excel(writer, sheet_name='era5_nearest', index=False)
 
#     cra_tq_4pt_filtered[(cra_tq_4pt_filtered['Level (hPa)']>=300)&(cra_tq_4pt_filtered['Level (hPa)']<=850)].to_excel(writer, sheet_name='cra_4pt ave', index=False)
#     era5_tq_4pt_filtered.to_excel(writer, sheet_name='era5_4pt ave', index=False)
    
#     workbook = writer.book
    
#     # 遍历所有工作表并设置列宽
#     for worksheet in workbook.worksheets:
#         for column in worksheet.columns:
#             max_length = 20
#             # 设置列宽
#             worksheet.column_dimensions[column[0].column_letter].width = max_length


"""
MWR vs FY4B / ERA5 / CRA / Radiosonde comparison (300–850 hPa)
- Two periods
- 3 layers: 300–500, 500–700, 700–850 hPa
- Metrics: STD, R, RMSE, Tian et al. (2017) metric T
- Taylor diagrams + bias-profile uncertainty plot
- Export: one Excel with 2 sheets: PairedData, Metrics

Assumes these DataFrames already exist in memory:
  fy4B_T, fy4B_q
  temp_p_df, q_p_gkg_df
  cra_tq_near_filtered, cra_tq_4pt_filtered
  era5_tq_near_filtered, era5_tq_4pt_filtered
  radiosonde
"""


# =========================
# 0) USER SETTINGS
# =========================
OUTPUT_DIR = r"E:\Beijing2024\MWR_compare_outputs-rh cbh"   # <<< change if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# two periods (inclusive end day)
periods = {
    "P1_20240606_20240609": ("2024-06-06", "2024-06-09"),
    "P2_20240805_20240810": ("2024-08-05", "2024-08-10"),
}

# time alignment to MWR base
TIME_FREQ = "60min"            # alignment frequency
TIME_ALIGN = "round"        # "round" | "floor" | "ceil"
MAX_TIME_DIFF = pd.Timedelta("45min")  # for asof matching if you enable it

# pressure range & layers
P_MIN, P_MAX = 300.0, 850.0

LAYER_DEF = {
    "300-500": (300.0, 500.0),
    "500-700": (500.0, 700.0),
    "700-850": (700.0, 850.0),
}

MIN_PAIRS = 15       # minimum paired samples to compute metrics
BOOT_N = 400         # bootstrap resamples for CI

# plotting style
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 12
# =========================
# 固定数据集颜色（全局）
# =========================
COLOR_MAP = {
    # 参考（MWR）
    "MWR": "red",

    # 探空
    "Radiosonde": "gold",

    # ERA5
    "ERA5_near": "blue",
    "ERA5_4pt": "dodgerblue",

    # CRA
    "CRA_near": "violet",
    "CRA_4pt": "magenta",

    # FY4B 三个环带
    "FY4B_<=6km": "darkgreen",
    "FY4B_20-25km": "lime",
    "FY4B_30-35km": "chartreuse",
}
# =========================
# 固定数据集 Marker（全局统一）
# =========================
def infer_marker(dname: str) -> str:
    dn = str(dname).lower()

    if "fy4b" in dn:
        return "^"
    if "radiosonde" in dn or "sonde" in dn:
        return "*"
    if "era5" in dn:
        return "."
    if "cra" in dn:
        return "."
    if "mwr" in dn:
        return "s"
    return "o"

# =========================
# 1) HELPERS
# =========================
def _to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def align_time_series(dt: pd.Series, freq=TIME_FREQ, how=TIME_ALIGN) -> pd.Series:
    dt = _to_datetime(dt)
    if how == "round":
        return dt.dt.round(freq)
    if how == "floor":
        return dt.dt.floor(freq)
    if how == "ceil":
        return dt.dt.ceil(freq)
    raise ValueError("TIME_ALIGN must be 'round'/'floor'/'ceil'")

def mwr_wide_to_long(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Robust MWR wide -> long:
      output columns: BJT_aligned, pressure_hPa, value_name
    Supports:
      - time in DatetimeIndex
      - or time in a column (BJT/Timestamp_BJT/time/datetime/...)
    """
    df = df_wide.copy()

    # ---- 1) get time column as 'BJT' ----
    time_candidates = ["BJT", "Timestamp_BJT", "timestamp_bjt", "time", "Time", "datetime", "Datetime", "DATE", "Date"]

    if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
        df = df.reset_index()
        # reset_index 后第一列通常就是原 index（名字可能是 index 或原 index.name）
        if "BJT" not in df.columns:
            df = df.rename(columns={df.columns[0]: "BJT"})
        df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")

    else:
        # 优先找候选时间列
        found = None
        for c in time_candidates:
            if c in df.columns:
                found = c
                break

        if found is None:
            # 最后兜底：找“可大量转成时间”的列
            for c in df.columns:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() >= 0.7 * len(df):
                    found = c
                    df[c] = parsed
                    break

        if found is None:
            raise ValueError("MWR表既没有DatetimeIndex，也没有可识别的时间列（如BJT/time）。")

        if found != "BJT":
            df = df.rename(columns={found: "BJT"})
        df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")

    df = df.dropna(subset=["BJT"])

    # ---- 2) detect pressure columns (anything convertible to float) ----
    pres_cols = []
    pres_vals = []
    for c in df.columns:
        if c == "BJT":
            continue
        try:
            p = float(str(c).strip())
            if (p >= P_MIN) and (p <= P_MAX):
                pres_cols.append(c)
                pres_vals.append(p)
        except Exception:
            continue

    if len(pres_cols) == 0:
        raise ValueError("未找到气压列：列名需要能转成数值(如 850.0, 825.0 ...)，且在300–850hPa内。")

    # ---- 3) melt ----
    long_df = df.melt(id_vars=["BJT"], value_vars=pres_cols,
                      var_name="pressure_hPa", value_name=value_name)

    long_df["pressure_hPa"] = long_df["pressure_hPa"].astype(str).str.strip()
    long_df["pressure_hPa"] = pd.to_numeric(long_df["pressure_hPa"], errors="coerce")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    long_df = long_df.dropna(subset=["pressure_hPa", value_name])

    long_df["BJT_aligned"] = align_time_series(long_df["BJT"])
    return long_df[["BJT_aligned", "pressure_hPa", value_name]]

def sat_bin_mean(df_long: pd.DataFrame, value_col: str, out_name_prefix: str) -> dict:
    """
    FY4B long: BJT, pressure_hPa, distance_km, value_col
    Return dict of {dataset_name: long_df(BJT_aligned, pressure_hPa, value)}
    """
    bins = {
        f"{out_name_prefix}_<=6km": (0.0, 6.0),
        f"{out_name_prefix}_20-25km": (20.0, 25.0),
        f"{out_name_prefix}_30-35km": (30.0, 35.0),
    }

    d = {}
    base = df_long.copy()
    base["BJT"] = _to_datetime(base["BJT"])
    base["BJT_aligned"] = align_time_series(base["BJT"])
    base["pressure_hPa"] = pd.to_numeric(base["pressure_hPa"], errors="coerce")
    base["distance_km"] = pd.to_numeric(base["distance_km"], errors="coerce")
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce")

    for name, (lo, hi) in bins.items():
        sub = base[(base["distance_km"] >= lo) & (base["distance_km"] <= hi)].copy()
        if sub.empty:
            d[name] = pd.DataFrame(columns=["BJT_aligned", "pressure_hPa", value_col])
            continue
        # average over available points within the ring for each time-pressure
        g = (sub.groupby(["BJT_aligned", "pressure_hPa"], as_index=False)[value_col]
             .mean())
        d[name] = g.dropna()
    return d

def reanalysis_to_long(df: pd.DataFrame, time_col: str, p_col: str,
                       t_col: str = None, q_col: str = None,
                       out_t: str = "T_K", out_q: str = "q_gkg") -> dict:
    """
    Convert reanalysis long to standardized long tables.
    Return {"T": df_T, "q": df_q} with columns:
      BJT_aligned, pressure_hPa, value
    """
    x = df.copy()
    x[time_col] = _to_datetime(x[time_col])
    x["BJT_aligned"] = align_time_series(x[time_col])
    x["pressure_hPa"] = pd.to_numeric(x[p_col], errors="coerce")

    out = {}
    if t_col is not None:
        xt = x[["BJT_aligned", "pressure_hPa", t_col]].rename(columns={t_col: out_t}).copy()
        xt[out_t] = pd.to_numeric(xt[out_t], errors="coerce")
        out["T"] = xt.dropna()
    if q_col is not None:
        xq = x[["BJT_aligned", "pressure_hPa", q_col]].rename(columns={q_col: out_q}).copy()
        xq[out_q] = pd.to_numeric(xq[out_q], errors="coerce")
        out["q"] = xq.dropna()
    return out

def radiosonde_to_long(rs: pd.DataFrame) -> dict:
    """
    radiosonde columns (given):
      time, pressure_hPa, temperature_C, mixing ratio_g/kg
    Convert to:
      T_K, q_gkg (specific humidity g/kg from mixing ratio)
    """
    x = rs.copy()
    # time column name in your radiosonde df is 'time'
    x["time"] = _to_datetime(x["time"])
    x["BJT_aligned"] = align_time_series(x["time"])
    x["pressure_hPa"] = pd.to_numeric(x["pressure_hPa"], errors="coerce")

    # temperature
    if "temperature_C" in x.columns:
        x["T_K"] = pd.to_numeric(x["temperature_C"], errors="coerce") + 273.15
    else:
        x["T_K"] = np.nan

    # mixing ratio -> specific humidity
    # w (kg/kg) = w_gkg / 1000; q = w/(1+w)
    if "mixing ratio_g/kg" in x.columns:
        w = pd.to_numeric(x["mixing ratio_g/kg"], errors="coerce") / 1000.0
        q = w / (1.0 + w)
        x["q_gkg"] = q * 1000.0
    else:
        x["q_gkg"] = np.nan

    out = {
        "T": x[["BJT_aligned", "pressure_hPa", "T_K"]].dropna(),
        "q": x[["BJT_aligned", "pressure_hPa", "q_gkg"]].dropna(),
    }
    return out

def filter_period_pressure(df: pd.DataFrame, start: str, end: str,
                           p_list: np.ndarray, value_col: str) -> pd.DataFrame:
    """
    Keep [start, end] inclusive by day, keep pressures in p_list, keep valid values.
    """
    st = pd.Timestamp(start)
    ed = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    y = df.copy()
    y = y[(y["BJT_aligned"] >= st) & (y["BJT_aligned"] <= ed)]
    y = y[y["pressure_hPa"].isin(p_list)]
    y[value_col] = pd.to_numeric(y[value_col], errors="coerce")
    return y.dropna(subset=[value_col])

def add_layer(p):
    for lname, (lo, hi) in LAYER_DEF.items():
        if (p >= lo) and (p <= hi):
            return lname
    return np.nan

def compute_metrics(ref: np.ndarray, mod: np.ndarray) -> dict:
    """
    STD, R, RMSE + Tian et al. (2017) metric T
    """
    ref = np.asarray(ref, dtype=float)
    mod = np.asarray(mod, dtype=float)

    m = np.isfinite(ref) & np.isfinite(mod)
    ref = ref[m]
    mod = mod[m]
    n = ref.size
    if n < MIN_PAIRS:
        return dict(n=n, std_ref=np.nan, std_mod=np.nan, r=np.nan,
                    bias=np.nan, mse=np.nan, rmse=np.nan, T=np.nan)

    bias = float(np.mean(mod - ref))
    mse = float(np.mean((mod - ref) ** 2))
    rmse = float(np.sqrt(mse))

    std_ref = float(np.std(ref, ddof=0))
    std_mod = float(np.std(mod, ddof=0))

    # correlation
    if std_ref == 0.0 and std_mod == 0.0:
        r = 1.0 if mse == 0.0 else 0.0
    elif std_ref == 0.0 or std_mod == 0.0:
        r = 0.0
    else:
        r = float(np.corrcoef(mod, ref)[0, 1])

    # Tian et al. 2017 metric T
    # T = ((1+R)/2) * (1 - MSE/(Bias^2 + sf^2 + sr^2))
    sf2 = std_mod ** 2
    sr2 = std_ref ** 2
    denom = (bias ** 2) + sf2 + sr2

    if denom == 0.0:
        T = 1.0 if mse == 0.0 else 0.0
    else:
        term1 = (1.0 + r) / 2.0
        term2 = 1.0 - (mse / denom)
        # keep within [0,1] to be stable in edge cases
        term1 = float(np.clip(term1, 0.0, 1.0))
        term2 = float(np.clip(term2, 0.0, 1.0))
        T = term1 * term2

    return dict(n=n, std_ref=std_ref, std_mod=std_mod, r=r,
                bias=bias, mse=mse, rmse=rmse, T=T)


def build_pairs(ref_df: pd.DataFrame, ref_col: str,
                tar_df: pd.DataFrame, tar_col: str,
                rh_hourly: pd.DataFrame = None,
                cbh_hourly: pd.DataFrame = None,
                rain_hourly: pd.DataFrame = None) -> pd.DataFrame:

    """
    Exact matching on (BJT_aligned, pressure_hPa).
    同时 merge RH(按小时) 与 CBH(按小时)，并产生 rh_bin 与 cbh_bin。
    """
    a = ref_df.rename(columns={ref_col: "ref"}).copy()
    b = tar_df.rename(columns={tar_col: "mod"}).copy()
    out = pd.merge(a, b, on=["BJT_aligned", "pressure_hPa"], how="inner")

    # --- RH ---
    if rh_hourly is not None and (not rh_hourly.empty):
        out = pd.merge(out, rh_hourly[["BJT_aligned", "RH"]], on="BJT_aligned", how="left")
        out["rh_bin"] = out["RH"].apply(assign_rh_bin)
    else:
        out["RH"] = np.nan
        out["rh_bin"] = np.nan

    # --- CBH ---
    if cbh_hourly is not None and (not cbh_hourly.empty):
        out = pd.merge(out, cbh_hourly[["BJT_aligned", "CBH"]], on="BJT_aligned", how="left")
        out["cbh_bin"] = out["CBH"].apply(assign_cbh_bin)
    else:
        out["CBH"] = np.nan
        out["cbh_bin"] = np.nan
    # --- RAIN ---
    if rain_hourly is not None and (not rain_hourly.empty):
        # 期望 rain_hourly 有列：BJT_aligned, RainRate_mmh, RainAmt_mm
        cols = ["BJT_aligned"]
        if "RainRate_mmh" in rain_hourly.columns:
            cols.append("RainRate_mmh")
        if "RainAmt_mm" in rain_hourly.columns:
            cols.append("RainAmt_mm")

        out = pd.merge(out, rain_hourly[cols], on="BJT_aligned", how="left")

        if "RainRate_mmh" not in out.columns:
            out["RainRate_mmh"] = np.nan
        if "RainAmt_mm" not in out.columns:
            out["RainAmt_mm"] = np.nan

        out["rain_bin"] = out.apply(lambda r: assign_rain_bin(r.get("RainRate_mmh", np.nan),
                                                              r.get("RainAmt_mm", np.nan)),
                                    axis=1)
    else:
        out["RainRate_mmh"] = np.nan
        out["RainAmt_mm"] = np.nan
        out["rain_bin"] = np.nan

    out["layer"] = out["pressure_hPa"].apply(add_layer)
    out = out.dropna(subset=["layer", "ref", "mod"])
    out["bias"] = out["mod"] - out["ref"]
    return out
# =========================
# RAIN helpers (NEW)
# =========================
RAIN_COL = "rainfall_rate_mmh"   # met_1min 里降水强度列（mm/h）

# 你说“按降水量分类输出”，但你现在字段是 rate(mm/h)；
# 这里默认用“小时平均降水强度”分箱（也可切换为小时累积降水量 mm）。
RAIN_BIN_MODE = "rate"   # "rate" 或 "amount"

# 分箱（可按你需要修改阈值）
# rate: mm/h；amount: mm（小时累积）
RAIN_BINS = [
    ("RAIN_0",        0.0,   0.0),     # 无降水
    ("RAIN_0.01-19.99",    0.01,   19.99),
    ("RAIN_20+",     20,  np.inf),
]

# =========================
# RH helpers (NEW)# CBH helpers (NEW)
# =========================

CBH_COL = "CBH"   # <<< 你CBH数据里云底高列名（单位：m），按你的实际列名改

CBH_BINS = [
    ("CBH_0-1500",        0.0, 1500.0),
    ("CBH_1500-5000",  1500.0, 5000.0),
    ("CBH_5000-10000", 5000.0, 10000.0),
]

RH_COL = "RH_mean_8_320"   # 你的湿度列名（百分比 0-100）

RH_BINS = [
    ("RH_0-60",   0.0, 60.0),
    ("RH_60-80", 60.0, 80.0),
    ("RH_80-100", 80.0, 100.0),
]
def cbh_to_aligned(cbh_obj, value_col=CBH_COL, agg="median") -> pd.DataFrame:
    """
    CBH(1-min or irregular) -> hourly aligned:
    return columns: BJT_aligned, CBH
    支持 Series / DataFrame / DatetimeIndex。
    agg: "mean" | "median" | "min" （云底高常用 median 或 min）
    """
    if isinstance(cbh_obj, pd.Series):
        df = cbh_obj.to_frame(name=value_col).copy()
        df = df.reset_index().rename(columns={df.columns[0]: "BJT"})
    else:
        df = cbh_obj.copy()
        if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
            df = df.reset_index()
            if "BJT" not in df.columns:
                df = df.rename(columns={df.columns[0]: "BJT"})
        else:
            time_candidates = ["BJT", "Timestamp_BJT", "timestamp_bjt", "time", "Time", "datetime", "Datetime", "DATE", "Date"]
            found = None
            for c in time_candidates:
                if c in df.columns:
                    found = c
                    break
            if found is None:
                for c in df.columns:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().sum() >= 0.7 * len(df):
                        found = c
                        df[c] = parsed
                        break
            if found is None:
                raise ValueError("CBH 数据找不到时间列（BJT/time/...），也不是 DatetimeIndex。")
            if found != "BJT":
                df = df.rename(columns={found: "BJT"})

    if value_col not in df.columns:
        raise ValueError(f"CBH 数据里找不到云底高列: {value_col}")

    df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")
    df = df.dropna(subset=["BJT"])
    df["CBH"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["CBH"])

    # 对齐到 BJT_aligned（例如整点）
    df["BJT_aligned"] = align_time_series(df["BJT"])

    # 1分钟 -> 1小时聚合
    if agg == "mean":
        out = df.groupby("BJT_aligned", as_index=False)["CBH"].mean()
    elif agg == "median":
        out = df.groupby("BJT_aligned", as_index=False)["CBH"].median()
    elif agg == "min":
        out = df.groupby("BJT_aligned", as_index=False)["CBH"].min()
    else:
        raise ValueError("agg must be one of: mean/median/min")

    return out

def assign_cbh_bin(cbh: float):
    """分箱： [0,1500), [1500,5000), [5000,10000]"""
    if not np.isfinite(cbh):
        return np.nan
    if cbh < 1500:
        return "CBH_0-1500"
    elif cbh < 5000:
        return "CBH_1500-5000"
    elif cbh <= 10000:
        return "CBH_5000-10000"
    return np.nan

def rh_to_aligned(rh_obj, value_col=RH_COL) -> pd.DataFrame:
    """
    rh_mean_8_320 (1-min) -> hourly aligned:
    return columns: BJT_aligned, RH
    支持：
      - Series: index 是时间，name/列名是 rh_mean_8_320
      - DataFrame: 有时间列(BJT/time/...) + rh_mean_8_320
      - DataFrame: index 是时间 + rh_mean_8_320 列
    """
    if isinstance(rh_obj, pd.Series):
        df = rh_obj.to_frame(name=value_col).copy()
        df = df.reset_index().rename(columns={df.columns[0]: "BJT"})
    else:
        df = rh_obj.copy()
        if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
            df = df.reset_index()
            if "BJT" not in df.columns:
                df = df.rename(columns={df.columns[0]: "BJT"})
        else:
            # 找时间列
            time_candidates = ["BJT", "Timestamp_BJT", "timestamp_bjt", "time", "Time", "datetime", "Datetime", "DATE", "Date"]
            found = None
            for c in time_candidates:
                if c in df.columns:
                    found = c
                    break
            if found is None:
                # 兜底：可大量转成时间的列
                for c in df.columns:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().sum() >= 0.7 * len(df):
                        found = c
                        df[c] = parsed
                        break
            if found is None:
                raise ValueError("RH 数据找不到时间列（BJT/time/...），也不是 DatetimeIndex。")
            if found != "BJT":
                df = df.rename(columns={found: "BJT"})

    if value_col not in df.columns:
        raise ValueError(f"RH 数据里找不到湿度列: {value_col}")

    df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")
    df = df.dropna(subset=["BJT"])
    df["RH"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["RH"])

    # 对齐到 BJT_aligned（例如整点）
    df["BJT_aligned"] = align_time_series(df["BJT"])

    # 1分钟 -> 1小时：对每个小时取平均RH
    out = df.groupby("BJT_aligned", as_index=False)["RH"].mean()
    # 清理范围外（可选）
    #out = out[(out["RH"] >= 0) & (out["RH"] <= 100)]
    return out

def assign_rh_bin(rh: float):
    """分箱： [0,60), [60,80), [80,100]（60算第二档，80算第三档）"""
    if not np.isfinite(rh):
        return np.nan
    if rh < 60:
        return "RH_0-60"
    elif rh < 80:
        return "RH_60-80"
    elif rh <= 100:
        return "RH_80-100"
    return np.nan
def rain_to_aligned(met_obj, value_col=RAIN_COL) -> pd.DataFrame:
    """
    met_1min (1-min or irregular) -> hourly aligned rainfall:
    return columns:
      BJT_aligned, RainRate_mmh, RainAmt_mm

    RainRate_mmh: 小时平均降水强度 (mm/h)
    RainAmt_mm:   小时累积降水量 (mm)，用 sum(rate/60) 近似积分
    支持 Series / DataFrame / DatetimeIndex。
    """
    if isinstance(met_obj, pd.Series):
        df = met_obj.to_frame(name=value_col).copy()
        df = df.reset_index().rename(columns={df.columns[0]: "BJT"})
    else:
        df = met_obj.copy()
        if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
            df = df.reset_index()
            if "BJT" not in df.columns:
                df = df.rename(columns={df.columns[0]: "BJT"})
        else:
            time_candidates = ["BJT", "Timestamp_BJT", "timestamp_bjt", "time", "Time",
                               "datetime", "Datetime", "DATE", "Date"]
            found = None
            for c in time_candidates:
                if c in df.columns:
                    found = c
                    break
            if found is None:
                for c in df.columns:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().sum() >= 0.7 * len(df):
                        found = c
                        df[c] = parsed
                        break
            if found is None:
                raise ValueError("降水数据找不到时间列（BJT/time/...），也不是 DatetimeIndex。")
            if found != "BJT":
                df = df.rename(columns={found: "BJT"})

    if value_col not in df.columns:
        raise ValueError(f"降水数据里找不到列: {value_col}")

    df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")
    df = df.dropna(subset=["BJT"])

    rr = pd.to_numeric(df[value_col], errors="coerce")
    rr = rr.clip(lower=0)  # 降水强度不应为负
    df["RainRate_mmh_1min"] = rr
    df = df.dropna(subset=["RainRate_mmh_1min"])

    # 对齐到整点（与你其他一致）
    df["BJT_aligned"] = align_time_series(df["BJT"])

    # 小时平均降水强度（mm/h）
    out_rate = df.groupby("BJT_aligned", as_index=False)["RainRate_mmh_1min"].mean()
    out_rate = out_rate.rename(columns={"RainRate_mmh_1min": "RainRate_mmh"})

    # 小时累积降水量（mm）：sum(rate/60)，一分钟一个采样近似积分
    out_amt = df.groupby("BJT_aligned", as_index=False)["RainRate_mmh_1min"].sum()
    out_amt["RainAmt_mm"] = out_amt["RainRate_mmh_1min"] / 60.0
    out_amt = out_amt[["BJT_aligned", "RainAmt_mm"]]

    out = pd.merge(out_rate, out_amt, on="BJT_aligned", how="outer").sort_values("BJT_aligned")
    return out


def assign_rain_bin(rate_mmh: float, amount_mm: float = None):
    """
    返回 rain_bin 字符串。
    - RAIN_BIN_MODE="rate": 按小时平均强度 mm/h 分箱
    - RAIN_BIN_MODE="amount": 按小时累积降水量 mm 分箱
    """
    x = None
    if RAIN_BIN_MODE == "amount":
        x = amount_mm
    else:
        x = rate_mmh

    if x is None or (not np.isfinite(x)):
        return np.nan

    # 精确 0 归到 RAIN_0
    if x == 0:
        return "RAIN_0"

    for name, lo, hi in RAIN_BINS:
        if name == "RAIN_0":
            continue
        if (x > lo) and (x <= hi):
            return name
        # 对于 (0,0.1] 这种你也可改成 >=
        if (lo == 0.0) and (x > 0.0) and (x <= hi):
            return name
    return np.nan

def filter_union_periods(df: pd.DataFrame, periods_dict: dict, value_col: str) -> pd.DataFrame:
    """仍然用你原来的两个时间段做“时间范围限制”（但不再用它来分组出图）"""
    if df.empty:
        return df
    masks = []
    for _, (start, end) in periods_dict.items():
        st = pd.Timestamp(start)
        ed = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        masks.append((df["BJT_aligned"] >= st) & (df["BJT_aligned"] <= ed))
    m = np.logical_or.reduce(masks) if masks else np.ones(len(df), dtype=bool)
    y = df.loc[m].copy()
    y[value_col] = pd.to_numeric(y[value_col], errors="coerce")
    return y.dropna(subset=[value_col])





# ---- Taylor diagram helper (styled version, backward compatible) ----
class TaylorDiagram:
    def __init__(self, ref_color="red", ref_std=1.0, fig=None, rect=111, label="Ref:MWR", srange=(0, 1.8),
                 *,
                 # 字体：不传则自动兜底；传入则强制用你给的路径
                 font_times_path=None,
                 font_chinese_path=None,
                 
                 # 样式参数（不影响外部调用；默认与原类一致）
                 r_interval=0.25,
                 major_corrs=(0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
                 minor_corrs=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                              0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
                              0.97, 0.98, 0.99, 1.0),
                 rms_levels=(0.5, 0.75, 1.0, 1.25, 1.5),    # 蓝色：以(1,0)为圆心的CRMSD圆
                 std_levels=(0.5, 0.75, 1.0, 1.25, 1.5),         # 红色：以(0,0)为圆心的STD圆
                 corr_guides=(0.4, 0.6, 0.8, 0.9, 0.95, 0.99)  # 绿色：相关系数辅助线
                 ):

        self.ref_std = ref_std

        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        self.fig = fig
        self.ax = fig.add_subplot(rect, polar=True)
        self.ax.plot([0], [1.0], marker="*", markersize=18, label=label,
             color=ref_color, markerfacecolor=ref_color, markeredgecolor=ref_color)

        # 增加主轴线（0°和90°）的线宽

        # ------------------ 字体：自动兜底避免 FileNotFoundError ------------------
        def _pick_existing(paths):
            for p in paths:
                if p and os.path.exists(p):
                    return p
            return None

        if font_times_path is None:
            font_times_path = _pick_existing([
                # Windows 常见
                r"C:\Windows\Fonts\times.ttf",
                r"C:\Windows\Fonts\times.ttf",
                # Linux 你之前的路径（如果确实存在）
                "/home/mw/input/font1842/Times.ttf",
            ])

        if font_chinese_path is None:
            font_chinese_path = _pick_existing([
                # Windows 常见：宋体多为 ttc
                r"C:\Windows\Fonts\simsun.ttc",
                r"C:\Windows\Fonts\simsun.ttf",
                # Linux 你之前的路径（如果确实存在）
                "/home/mw/input/font1842/SimSun.ttf",
            ])

        self._fp_times = FontProperties(fname=font_times_path) if font_times_path else FontProperties()
        self._fp_cn    = FontProperties(fname=font_chinese_path) if font_chinese_path else FontProperties()

        # ------------------ 坐标系（保持你原类的设定） ------------------
        self.ax.set_theta_zero_location("E")
        self.ax.set_theta_direction(1)
        self.ax.set_thetamin(0)
        self.ax.set_thetamax(90)
        # 0° 和 90° 边界轴线宽度（单位：points）
        self.ax.spines["start"].set_linewidth(2)  # 0°
        self.ax.spines["end"].set_linewidth(2)    # 90°

        # 如果你还想把外侧圆弧边界也加粗（可选）
        self.ax.spines["polar"].set_linewidth(2)

        # ------------------ 相关系数刻度（主/次刻度 + 外圈短刻度线） ------------------
        major_corrs = np.array(major_corrs, dtype=float)
        minor_corrs = np.array(minor_corrs, dtype=float)

        major_thetas = np.arccos(np.clip(major_corrs, -1, 1))
        minor_thetas = np.arccos(np.clip(minor_corrs, -1, 1))

        self.ax.set_xticks(major_thetas)
        # 显示与示例一致：直接显示 corr 值
        self.ax.set_xticklabels([f"{c:g}" for c in major_corrs])

        for lab in self.ax.get_xticklabels():
            lab.set_fontproperties(self._fp_times)
            lab.set_fontsize(20)

        # ------------------ r 范围 + 手工 r 标注（REF/数值） ------------------
        r_small, r_big = float(srange[0]), float(srange[1])
        self.ax.set_rlim(r_small, r_big)

        # 去掉默认 rgrid，使用你示例那种“在两条轴上标注”
        self.ax.set_rgrids([])

        rr = np.arange(r_small, r_big + 1e-9, r_interval)
        for r in rr:
            if abs(r - 1.0) < 1e-12:
                txt = "\nRef:MWR"
            else:
                txt = "\n" + (f"{r:g}")
            # theta=0 轴
            self.ax.text(0, r, s=txt, fontproperties=self._fp_times, fontsize=20,
                         ha="center", va="top")
            # theta=90° 边界
            self.ax.text(np.pi / 2, r, s=(f"{r:g}  "), fontproperties=self._fp_times, fontsize=20,
                         ha="right", va="center")

        # 不要默认网格
        self.ax.grid(False)

        # 外圈短刻度线（主/次）
        angle_linewidth, angle_length, angle_minor_length = 2, 0.02, 0.01
        tick = [self.ax.get_rmax(), self.ax.get_rmax() * (1 - angle_length)]
        tick_minor = [self.ax.get_rmax(), self.ax.get_rmax() * (1 - angle_minor_length)]
        self.ax.plot([0, 0], [0, self.ax.get_rmax()], lw=2, color="k")  # 0°水平线
        self.ax.plot([np.pi/2, np.pi/2], [0, self.ax.get_rmax()], lw=2, color="k")  # 90°垂直线
        for t in major_thetas:
            self.ax.plot([t, t], tick, lw=angle_linewidth, color="k")
        for t in minor_thetas:
            self.ax.plot([t, t], tick_minor, lw=angle_linewidth, color="k")

        # ------------------ 圆弧：CRMSD（蓝）+ STD（红） ------------------
        # 关键：用 ax.transData._b 把“笛卡尔坐标圆”画在极坐标图上（与你示例一致）
        for lv in rms_levels:
            c = Circle((1, 0), lv, transform=self.ax.transData._b,
                       facecolor=(0, 0, 0, 0), edgecolor="blue",
                       linestyle=(0, (5, 5)), linewidth=1)
            self.ax.add_artist(c)

        for lv in std_levels:
            c = Circle((0, 0), lv, transform=self.ax.transData._b,
                       facecolor=(0, 0, 0, 0), edgecolor="red",
                       linestyle=(0, (5, 5)), linewidth=1)
            self.ax.add_artist(c)

        # ------------------ 相关系数辅助线（绿虚线） ------------------
        for c in corr_guides:
            th = float(np.arccos(np.clip(c, -1, 1)))
            self.ax.plot([0, th], [0, self.ax.get_rmax()], lw=1, color="green", linestyle=(0, (5, 5)))

        # 轴注记（与你示例一致）
        #self.ax.set_ylabel("Standard Deviation", fontproperties=self._fp_times, color="red", fontweight='bold',labelpad=45, fontsize=20)
        #self.ax.set_xlabel("RMSE", fontproperties=self._fp_times, labelpad=45, color="blue",fontsize=20, fontweight='bold')
        self.ax.text(np.deg2rad(40), self.ax.get_rmax() + 0.00, s="Correlation Coefficients",
                      fontproperties=self._fp_times, fontsize=20,color="green", fontweight='bold',
                      ha="center", va="bottom", rotation=-45)
        
        # ---- 控制左右（x）与上下（y）位置：0~1 是轴宽/高的比例 ----
        self.ax.xaxis.set_label_coords(0.55, -0.07)  # x<0.5 往左；x>0.5 往右
        
        # REF 点（保持原接口/含义：std_ratio=1, corr=1 => theta=0, r=1）
        self.ax.plot([0], [1.0], marker="*", markersize=18, label=label, color=ref_color)

    def add_sample(self, std_ratio, corrcoef, label, marker="o", color=None):
        theta = np.arccos(np.clip(corrcoef, -1, 1))
        self.ax.plot([theta], [std_ratio], marker=marker, markersize=14,color=color,
                     linestyle="None", label=label)

# ---- bootstrap CI for mean bias profile ----
def bootstrap_mean_ci(x: np.ndarray, n_boot=BOOT_N, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < MIN_PAIRS:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(2025)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        means[i] = np.mean(samp)
    lo = np.quantile(means, alpha/2)
    hi = np.quantile(means, 1 - alpha/2)
    return float(np.mean(x)), float(lo), float(hi)

# =========================
# 2) BUILD STANDARDIZED LONG TABLES
# =========================
# MWR reference
mwr_T = mwr_wide_to_long(temp_p_df, "T_K")        # BJT_aligned, pressure_hPa, T_K
mwr_q = mwr_wide_to_long(q_p_gkg_df, "q_gkg")     # BJT_aligned, pressure_hPa, q_gkg

# Use MWR pressure list as the standard pressures (300–850)
mwr_pressures = np.array(sorted([p for p in mwr_T["pressure_hPa"].unique() if (p >= P_MIN and p <= P_MAX)]), dtype=float)

# FY4B rings
fy4B_T_bins = sat_bin_mean(fy4B_T, "T_K", "FY4B_T")
fy4B_q_bins = sat_bin_mean(fy4B_q, "q_gkg", "FY4B_q")

# CRA (near + 4pt)
cra_near = reanalysis_to_long(
    cra_tq_near_filtered, time_col="BJT", p_col="Level (hPa)",
    t_col="Temp (K)", q_col="q (g/kg)",
    out_t="T_K", out_q="q_gkg"
)
cra_4pt = reanalysis_to_long(
    cra_tq_4pt_filtered, time_col="BJT", p_col="Level (hPa)",
    t_col="Temp (K)", q_col="q (g/kg)",
    out_t="T_K", out_q="q_gkg"
)

# ERA5 (near + 4pt)
era5_near = reanalysis_to_long(
    era5_tq_near_filtered, time_col="BJT", p_col="pressure_level",
    t_col="Temp (K)", q_col="q (g/kg)",
    out_t="T_K", out_q="q_gkg"
)
era5_4pt = reanalysis_to_long(
    era5_tq_4pt_filtered, time_col="BJT", p_col="pressure_level",
    t_col="Temp (K)", q_col="q (g/kg)",
    out_t="T_K", out_q="q_gkg"
)

# Radiosonde
rs = radiosonde_to_long(radiosonde)

# Build dataset dictionaries for each variable
DATASETS_T = {}
DATASETS_q = {}

# FY4B
for k, df in fy4B_T_bins.items():
    DATASETS_T[k.replace("FY4B_T", "FY4B")] = df.rename(columns={"T_K": "T_K"})
for k, df in fy4B_q_bins.items():
    DATASETS_q[k.replace("FY4B_q", "FY4B")] = df.rename(columns={"q_gkg": "q_gkg"})

# CRA/ERA5
DATASETS_T["CRA_near"] = cra_near["T"]
DATASETS_T["CRA_4pt"]  = cra_4pt["T"]
DATASETS_T["ERA5_near"] = era5_near["T"]
DATASETS_T["ERA5_4pt"]  = era5_4pt["T"]

DATASETS_q["CRA_near"] = cra_near["q"]
DATASETS_q["CRA_4pt"]  = cra_4pt["q"]
DATASETS_q["ERA5_near"] = era5_near["q"]
DATASETS_q["ERA5_4pt"]  = era5_4pt["q"]

# Radiosonde
DATASETS_T["Radiosonde"] = rs["T"]
DATASETS_q["Radiosonde"] = rs["q"]

# =========================
# 3) MAIN LOOP: RH bins × variables × datasets
# （仍然只在“时间范围上”限制为原来的两个时间段，但不再按 period 分组）
# =========================
paired_all = []
metrics_all = []
metrics_all_cbh = []
metrics_all_rain = []

# 先把 RH 做成小时对齐
rh_hourly = rh_to_aligned(rh_mean_8_320, value_col=RH_COL)

# 只保留原两个时间段范围内的 RH（可选但建议，避免其他时间混入）
rh_hourly = filter_union_periods(rh_hourly.rename(columns={"RH": "RHval"}), periods, "RHval") \
              .rename(columns={"RHval": "RH"})
# === NEW: CBH hourly ===
# 你需要保证 cbh_1min（或你的 CBH 数据对象）已经在内存里
cbh_hourly = cbh_to_aligned(cbh_1min, value_col=CBH_COL, agg="median")  # <<< cbh_1min 与列名按你实际改
cbh_hourly = filter_union_periods(cbh_hourly.rename(columns={"CBH": "CBHval"}), periods, "CBHval") \
               .rename(columns={"CBHval": "CBH"})

print(f"CBH hourly rows: {len(cbh_hourly)}")
# === NEW: RAIN hourly (from met_1min) ===
# 你需要保证 met_1min 已经在内存里，且包含 rainfall_rate_mmh
rain_hourly = rain_to_aligned(met_1min, value_col=RAIN_COL)

# 只保留原两个时间段范围内的降水（关键：你说降水包含很多别的时间）
rain_hourly = filter_union_periods(
    rain_hourly.rename(columns={"RainRate_mmh": "RainRate_tmp"}),
    periods,
    "RainRate_tmp"
).rename(columns={"RainRate_tmp": "RainRate_mmh"})

# RainAmt_mm 也做一次数值清理（可选）
if "RainAmt_mm" in rain_hourly.columns:
    rain_hourly["RainAmt_mm"] = pd.to_numeric(rain_hourly["RainAmt_mm"], errors="coerce")

print(f"RAIN hourly rows: {len(rain_hourly)}")

# 同样：把 MWR 和各数据集都限制在原两个时段（但最终分组靠 rh_bin）
refT_all = filter_union_periods(mwr_T, periods, "T_K")
refq_all = filter_union_periods(mwr_q, periods, "q_gkg")

#print(f"\n=== RH-binned analysis within union of given periods ===")
print(f"MWR T rows: {len(refT_all)}, MWR q rows: {len(refq_all)}")
print(f"RH hourly rows: {len(rh_hourly)}")

for var, ref_df, ref_col, dsets in [
    ("T", refT_all, "T_K", DATASETS_T),
    ("q", refq_all, "q_gkg", DATASETS_q),
]:
    for dname, ddf0 in dsets.items():
        tar_col = "T_K" if var == "T" else "q_gkg"

        # 仍然限制在原两个时间段（不是分组，只是范围）
        ddf = filter_union_periods(ddf0, periods, tar_col)

        # 压力仍用 MWR pressure 列表
        ddf = ddf[ddf["pressure_hPa"].isin(mwr_pressures)].copy()
        ref_use = ref_df[ref_df["pressure_hPa"].isin(mwr_pressures)].copy()

        pairs = build_pairs(
            ref_use[["BJT_aligned", "pressure_hPa", ref_col]], ref_col,
            ddf[["BJT_aligned", "pressure_hPa", tar_col]], tar_col,
            rh_hourly=rh_hourly,
            cbh_hourly=cbh_hourly,
            rain_hourly=rain_hourly
        )

        if pairs.empty:
            continue

        # 保存 paired（用于导出和 bias profile）
        tmp = pairs.copy()
        tmp["variable"] = var
        tmp["dataset"] = dname
        paired_all.append(tmp[[
            "variable", "dataset",
            "rh_bin", "RH",
            "cbh_bin", "CBH",
            "rain_bin", "RainRate_mmh", "RainAmt_mm",
            "layer", "BJT_aligned", "pressure_hPa", "ref", "mod", "bias"
        ]])


        # 分 RH 档 + 分 layer 统计指标
        for rh_name, lo, hi in RH_BINS:
            sub_rh = pairs[pairs["rh_bin"] == rh_name]
            if sub_rh.empty:
                continue
            for layer_name in LAYER_DEF.keys():
                sub = sub_rh[sub_rh["layer"] == layer_name]
                mm = compute_metrics(sub["ref"].values, sub["mod"].values)
                metrics_all.append({
                    "rh_bin": rh_name,
                    "variable": var,
                    "dataset": dname,
                    "layer": layer_name,
                    **mm
                })
        # ===== NEW: CBH bins × layer metrics =====
        for cbh_name, lo, hi in CBH_BINS:
            sub_cbh = pairs[pairs["cbh_bin"] == cbh_name]
            if sub_cbh.empty:
                continue
            for layer_name in LAYER_DEF.keys():
                sub = sub_cbh[sub_cbh["layer"] == layer_name]
                mm = compute_metrics(sub["ref"].values, sub["mod"].values)
                metrics_all_cbh.append({
                    "cbh_bin": cbh_name,
                    "variable": var,
                    "dataset": dname,
                    "layer": layer_name,
                    **mm
                })
        # ===== NEW: RAIN bins × layer metrics =====
        for rain_name, lo, hi in RAIN_BINS:
            sub_rain = pairs[pairs["rain_bin"] == rain_name]
            if sub_rain.empty:
                continue
            for layer_name in LAYER_DEF.keys():
                sub = sub_rain[sub_rain["layer"] == layer_name]
                mm = compute_metrics(sub["ref"].values, sub["mod"].values)
                metrics_all_rain.append({
                    "rain_bin": rain_name,
                    "variable": var,
                    "dataset": dname,
                    "layer": layer_name,
                    **mm
                })

paired_df = pd.concat(paired_all, ignore_index=True) if paired_all else pd.DataFrame()
metrics_df = pd.DataFrame(metrics_all)
metrics_cbh_df = pd.DataFrame(metrics_all_cbh)
metrics_rain_df = pd.DataFrame(metrics_all_rain)
print("[OK] RAIN-binned metrics done.")

print("[OK] CBH-binned metrics done.")

print("[OK] RH-binned pairing & metrics done.")

# =========================
# 5) TAYLOR DIAGRAMS (period × var × layer)
# =========================

def plot_taylor_for(rh_bin_name, var, layer_name):
    subm = metrics_df[(metrics_df["rh_bin"] == rh_bin_name) &
                      (metrics_df["variable"] == var) &
                      (metrics_df["layer"] == layer_name)].copy()
    subm = subm.dropna(subset=["std_ref", "std_mod", "r"])
    if subm.empty:
        return

    subm["std_ratio"] = subm["std_mod"] / subm["std_ref"].replace(0, np.nan)

    fig = plt.figure(figsize=(8.5, 8.5))
    td = TaylorDiagram(ref_std=1.0, fig=fig, rect=111, label="MWR", srange=(0, 1.8), ref_color=COLOR_MAP.get("MWR", "red"))

    for _, row in subm.iterrows():
        label  = f"{row['dataset']}, T={row['T']:.3f}"
        marker = infer_marker(row["dataset"])
        color  = COLOR_MAP.get(str(row["dataset"]), None)
    
        td.add_sample(row["std_ratio"], row["r"], label=label, marker=marker, color=color)

        
        
        # 检查数据集名称是否包含关键词
        # for key, mark in marker_map.items():
        #     if key in dataset_name:
        #         marker = mark
        #         break
        color = COLOR_MAP.get(str(row["dataset"]), None)
        td.add_sample(row["std_ratio"], row["r"], label=label, marker=marker, color=color)
        

    title_var = "Temperature (K)" if var == "T" else "Specific humidity (g/kg)"
    plt.title(f"{rh_bin_name} | {layer_name} hPa {title_var}",
              fontweight="bold", pad=22)
    plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0.72), frameon=False)

    out_png = os.path.join(OUTPUT_DIR, f"Taylor_{rh_bin_name}_{var}_{layer_name}.tif")
    plt.savefig(
        os.path.join(out_png),
        dpi = 150, bbox_inches = "tight", format = "tif", facecolor = 'none'
    )
    plt.close(fig)

for rh_name, _, _ in RH_BINS:
    for var in ["T", "q"]:
        for layer in LAYER_DEF.keys():
            plot_taylor_for(rh_name, var, layer)

print("[OK] RH-binned Taylor diagrams saved.")


def plot_taylor_for_cbh(cbh_bin_name, var, layer_name):
    subm = metrics_cbh_df[(metrics_cbh_df["cbh_bin"] == cbh_bin_name) &
                          (metrics_cbh_df["variable"] == var) &
                          (metrics_cbh_df["layer"] == layer_name)].copy()
    subm = subm.dropna(subset=["std_ref", "std_mod", "r"])
    if subm.empty:
        return

    subm["std_ratio"] = subm["std_mod"] / subm["std_ref"].replace(0, np.nan)

    fig = plt.figure(figsize=(8.5, 8.5))
    td = TaylorDiagram(ref_std=1.0, fig=fig, rect=111, label="MWR", srange=(0, 1.8), ref_color=COLOR_MAP.get("MWR", "red"))

    for _, row in subm.iterrows():
        
        label  = f"{row['dataset']}, T={row['T']:.3f}"
        marker = infer_marker(row["dataset"])
        color  = COLOR_MAP.get(str(row["dataset"]), None)
    
        td.add_sample(row["std_ratio"], row["r"], label=label, marker=marker, color=color)
    
        # for key, mark in marker_map.items():
        #     if key in dataset_name:
        #         marker = mark
        #         break

        color = COLOR_MAP.get(str(row["dataset"]), None)
        td.add_sample(row["std_ratio"], row["r"], label=label, marker=marker, color=color)

    title_var = "Temperature (K)" if var == "T" else "Specific humidity (g/kg)"
    plt.title(f"{cbh_bin_name} | {layer_name} hPa {title_var}",
              fontweight="bold", pad=22)
    plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0.72), frameon=False)

    out_png = os.path.join(OUTPUT_DIR, f"Taylor_CBH_{cbh_bin_name}_{var}_{layer_name}.tif")
    plt.savefig(
        os.path.join(out_png),
        dpi = 150, bbox_inches = "tight", format = "tif", facecolor = 'none'
    )
    plt.close(fig)

# === NEW: CBH-binned Taylor diagrams ===
for cbh_name, _, _ in CBH_BINS:
    for var in ["T", "q"]:
        for layer in LAYER_DEF.keys():
            plot_taylor_for_cbh(cbh_name, var, layer)

print("[OK] CBH-binned Taylor diagrams saved.")

# =========================
# 5.3) RAIN-binned TAYLOR DIAGRAMS (NEW)
# =========================
def plot_taylor_for_rain(rain_bin_name, var, layer_name):
    subm = metrics_rain_df[(metrics_rain_df["rain_bin"] == rain_bin_name) &
                           (metrics_rain_df["variable"] == var) &
                           (metrics_rain_df["layer"] == layer_name)].copy()
    subm = subm.dropna(subset=["std_ref", "std_mod", "r"])
    if subm.empty:
        return

    subm["std_ratio"] = subm["std_mod"] / subm["std_ref"].replace(0, np.nan)

    fig = plt.figure(figsize=(9, 9))
    td = TaylorDiagram(ref_std=1.0, fig=fig, rect=111, label="MWR", srange=(0, 1.8), ref_color=COLOR_MAP.get("MWR", "red"))

    for _, row in subm.iterrows():
        label  = f"{row['dataset']}"
        marker = infer_marker(row["dataset"])
        color  = COLOR_MAP.get(str(row["dataset"]), None)

        td.add_sample(row["std_ratio"], row["r"], label=label, marker=marker, color=color)




        # for key, mark in marker_map.items():
        #     if key in dataset_name:
        #         marker = mark
        #         break

        color = COLOR_MAP.get(str(row["dataset"]), None)
        td.add_sample(row["std_ratio"], row["r"], label=label, marker=marker, color=color)


    title_var = "Temperature (K)" if var == "T" else "Specific humidity (g/kg)"
    unit_tag = "mm/h" if RAIN_BIN_MODE == "rate" else "mm"
    plt.title(f"{rain_bin_name} ({unit_tag}) | {layer_name} hPa {title_var}",
              fontweight="bold", pad=22, fontsize = 20)
    #plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0.72), frameon=False)

    out_png = os.path.join(OUTPUT_DIR, f"Taylor_RAIN_{rain_bin_name}_{var}_{layer_name}.tif")
    plt.savefig(
        os.path.join(out_png),
        dpi = 150, bbox_inches = "tight", format = "tif", facecolor = 'none'
    )
    plt.close(fig)

# 批量输出
for rain_name, _, _ in RAIN_BINS:
    for var in ["T", "q"]:
        for layer in LAYER_DEF.keys():
            plot_taylor_for_rain(rain_name, var, layer)

print("[OK] RAIN-binned Taylor diagrams saved.")

# =========================
# 6) BIAS PROFILE UNCERTAINTY PLOT (bootstrap 95% CI)
# =========================
def plot_bias_uncertainty(rh_bin_name, var):
    x = paired_df[(paired_df["rh_bin"] == rh_bin_name) &
                  (paired_df["variable"] == var)].copy()
    if x.empty:
        return

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111)
    linestyle_map = {
            "FY4B_<=6km": (0, (5, 1)),
            "FY4B_20-25km": (0, (5, 1)),
            "FY4B_30-35km": (0, (5, 1)),
            "CRA_near": "dotted",
            "CRA_4pt": "dotted",
            "ERA5_near": "dotted",
            "ERA5_4pt": "dotted",
            "Radiosonde": "solid"  # 或使用 "*-" 添加标记
        }
    for dname in sorted(x["dataset"].unique()):
        xd = x[x["dataset"] == dname]
        means, los, his, ps = [], [], [], []

        for p in sorted(mwr_pressures, reverse=True):
            bp = xd[xd["pressure_hPa"] == p]["bias"].values
            m, lo, hi = bootstrap_mean_ci(bp)
            if np.isfinite(m):
                ps.append(p); means.append(m); los.append(lo); his.append(hi)

        if len(ps) < 6:
            continue

        ps = np.array(ps)
        means = np.array(means)
        los = np.array(los)
        his = np.array(his)
        linewidth  = 2.5 if dname == "Radiosonde" else 1.5
        c = COLOR_MAP.get(str(dname), None)

        ax.plot(means, ps, color=c, linewidth=linewidth , label=dname, linestyle = linestyle_map.get(dname, "-"))  # 如果没有匹配到，则默认使用实线'-')
        ax.fill_betweenx(ps, los,  his, alpha=0.25, color=c)

    ax.axvline(0, linewidth=2,color = "black", linestyle = (0, (5, 10)))
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (hPa)", fontweight="bold")
    if var == "T":
        ax.set_xlabel("Bias = (dataset - MWR) (K)", fontweight="bold")
        title_var = "Temperature"
    else:
        ax.set_xlabel("Bias = (dataset - MWR) (g/kg)", fontweight="bold")
        title_var = "Specific humidity"

    ax.set_title(f"{rh_bin_name} {title_var} Bias Profile",
                 fontweight="bold", pad=14)
    #ax.legend(loc="upper right", frameon=False)
    # 设置四周边框宽度为2
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='major', width=2,labelsize = 18)
    #ax.set_facecolor('lightgray')
    out_png = os.path.join(OUTPUT_DIR, f"BiasUncertainty_{rh_bin_name}_{var}.tif")
    plt.savefig(
        os.path.join(out_png),
        dpi = 150, bbox_inches = "tight", format = "tif", facecolor = 'none'
    )
    plt.close(fig)

for rh_name, _, _ in RH_BINS:
    plot_bias_uncertainty(rh_name, "T")
    plot_bias_uncertainty(rh_name, "q")

print("[OK] RH-binned bias uncertainty plots saved.")


def plot_bias_uncertainty_cbh(cbh_bin_name, var):
    x = paired_df[(paired_df["cbh_bin"] == cbh_bin_name) &
                  (paired_df["variable"] == var)].copy()
    if x.empty:
        return

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111)

    linestyle_map = {
        "FY4B_<=6km": (0, (5, 1)),
        "FY4B_20-25km": (0, (5, 1)),
        "FY4B_30-35km": (0, (5, 1)),
        "CRA_near": "dotted",
        "CRA_4pt": "dotted",
        "ERA5_near": "dotted",
        "ERA5_4pt": "dotted",
        "Radiosonde": "solid"
    }

    for dname in sorted(x["dataset"].unique()):
        xd = x[x["dataset"] == dname]
        means, los, his, ps = [], [], [], []

        for p in sorted(mwr_pressures, reverse=True):
            bp = xd[xd["pressure_hPa"] == p]["bias"].values
            m, lo, hi = bootstrap_mean_ci(bp)
            if np.isfinite(m):
                ps.append(p); means.append(m); los.append(lo); his.append(hi)

        if len(ps) < 6:
            continue

        ps = np.array(ps)
        means = np.array(means)
        los = np.array(los)
        his = np.array(his)

        linewidth = 2.5 if dname == "Radiosonde" else 1.5
        c = COLOR_MAP.get(str(dname), None)
        ax.plot(means, ps, linewidth=linewidth, label=dname,color = c,
                linestyle=linestyle_map.get(dname, "-"))
        ax.fill_betweenx(ps, los, his, alpha=0.25, color = c)

    ax.axvline(0, linewidth=2, color="black", linestyle=(0, (5, 10)))
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (hPa)", fontweight="bold")

    if var == "T":
        ax.set_xlabel("Bias = (dataset - MWR) (K)", fontweight="bold")
        title_var = "Temperature"
    else:
        ax.set_xlabel("Bias = (dataset - MWR) (g/kg)", fontweight="bold")
        title_var = "Specific humidity"

    ax.set_title(f"{cbh_bin_name} {title_var} Bias Profile",
                 fontweight="bold", pad=14)
    #ax.legend(loc="upper right", frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='major', width=2, labelsize=18)
    #ax.set_facecolor('lightgray')
    out_png = os.path.join(OUTPUT_DIR, f"BiasUncertainty_CBH_{cbh_bin_name}_{var}.tif")
    plt.savefig(
        os.path.join(out_png),
        dpi = 150, bbox_inches = "tight", format = "tif", facecolor = 'none'
    )
    plt.close(fig)
# === NEW: CBH-binned bias uncertainty plots ===
for cbh_name, _, _ in CBH_BINS:
    plot_bias_uncertainty_cbh(cbh_name, "T")
    plot_bias_uncertainty_cbh(cbh_name, "q")

print("[OK] CBH-binned bias uncertainty plots saved.")
# =========================
# 6.3) RAIN-binned BIAS PROFILE UNCERTAINTY PLOT (NEW)
# =========================
def plot_bias_uncertainty_rain(rain_bin_name, var):
    x = paired_df[(paired_df["rain_bin"] == rain_bin_name) &
                  (paired_df["variable"] == var)].copy()
    if x.empty:
        return

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111)

    linestyle_map = {
        "FY4B_<=6km": (0, (5, 1)),
        "FY4B_20-25km": (0, (5, 1)),
        "FY4B_30-35km": (0, (5, 1)),
        "CRA_near": "dotted",
        "CRA_4pt": "dotted",
        "ERA5_near": "dotted",
        "ERA5_4pt": "dotted",
        "Radiosonde": "solid"
    }

    for dname in sorted(x["dataset"].unique()):
        xd = x[x["dataset"] == dname]
        means, los, his, ps = [], [], [], []

        for p in sorted(mwr_pressures, reverse=True):
            bp = xd[xd["pressure_hPa"] == p]["bias"].values
            m, lo, hi = bootstrap_mean_ci(bp)
            if np.isfinite(m):
                ps.append(p); means.append(m); los.append(lo); his.append(hi)

        if len(ps) < 6:
            continue

        ps = np.array(ps)
        means = np.array(means)
        los = np.array(los)
        his = np.array(his)

        linewidth = 2.5 if dname == "Radiosonde" else 1.5
        c = COLOR_MAP.get(str(dname), None)
        ax.plot(means, ps, linewidth=linewidth, label=dname,color = c,
                linestyle=linestyle_map.get(dname, "-"))
        ax.fill_betweenx(ps, los, his, alpha=0.25, color = c)

    ax.axvline(0, linewidth=2, color="black", linestyle=(0, (5, 10)))
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (hPa)", fontweight="bold")

    if var == "T":
        ax.set_xlabel("Bias = (dataset - MWR) (K)", fontweight="bold")
        title_var = "Temperature"
    else:
        ax.set_xlabel("Bias = (dataset - MWR) (g/kg)", fontweight="bold")
        title_var = "Specific humidity"

    unit_tag = "mm/h" if RAIN_BIN_MODE == "rate" else "mm"
    ax.set_title(f"{rain_bin_name} ({unit_tag}) {title_var} Bias Profile",
                 fontweight="bold", pad=14)
    #ax.legend(loc="upper right", frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='major', width=2, labelsize=18)
    #ax.set_facecolor('lightgray')
    out_png = os.path.join(OUTPUT_DIR, f"BiasUncertainty_RAIN_{rain_bin_name}_{var}.tif")
    plt.savefig(
        os.path.join(out_png),
        dpi = 150, bbox_inches = "tight", format = "tif", facecolor = 'none'
    )
    plt.close(fig)

# 批量输出
for rain_name, _, _ in RAIN_BINS:
    plot_bias_uncertainty_rain(rain_name, "T")
    plot_bias_uncertainty_rain(rain_name, "q")

print("[OK] RAIN-binned bias uncertainty plots saved.")
def export_dataset_legend_only_colored(
    out_path,
    *,
    datasets,
    mode="taylor",         # "taylor" | "bias" | "combined"
    include_ref=True,
    ref_label="MWR (Ref)",
    ref_key="MWR",         # 用于从 COLOR_MAP 取参考色的 key
    font_family="Times New Roman",
    fontsize=14,
    ncol=1,
    fig_w=5.2,
    fig_h=4.0,
    dpi=300,
    frameon=False,
    title=None,
    title_fontsize=None,
    sort_datasets=False,
    color_map=None,        # 传入固定颜色字典；不传则自动用全局 COLOR_MAP
    fy_style=None          # 可选：你给的 FY_STYLE（≤6 km / 20–25 km / 30–35 km）
):
    """
    生成“只有图例”的图片，并且把每个数据集颜色固定下来（不再用默认颜色循环）。
    颜色来源优先级：
      1) color_map[dname] 精确匹配
      2) 如果是 FY4B_* 并且给了 fy_style，则按环带匹配
      3) 仍找不到则 color=None（交给 Matplotlib 默认，但一般你会给全）
    """

    import os
    import matplotlib.pyplot as plt

    # -------- 颜色表：默认取全局 COLOR_MAP --------
    if color_map is None:
        color_map = globals().get("COLOR_MAP", {}) or {}

    def _infer_marker(dname: str) -> str:
        return infer_marker(dname)





    linestyle_map = {
        "FY4B_<=6km": (0, (5, 1)),
        "FY4B_20-25km": (0, (5, 1)),
        "FY4B_30-35km": (0, (5, 1)),
        "CRA_near": "dotted",
        "CRA_4pt": "dotted",
        "ERA5_near": "dotted",
        "ERA5_4pt": "dotted",
        "Radiosonde": "solid",
    }



    def _infer_linestyle(dname: str):
        return linestyle_map.get(str(dname), "-")

    def _infer_linewidth(dname: str) -> float:
        return 2.5 if str(dname) == "Radiosonde" else 1.8

    def _infer_color(dname: str):
        dname = str(dname)

        # 1) 精确匹配
        if dname in color_map:
            return color_map[dname]

        # 2) FY4B 环带：允许用你给的 FY_STYLE（≤6 km / 20–25 km / 30–35 km）
        if fy_style is not None and ("FY4B" in dname):
            dn = dname.replace(" ", "")
            if "<=6km" in dn or "≤6km" in dn:
                return fy_style.get("≤6 km", None)
            if "20-25km" in dn or "20–25km" in dn:
                return fy_style.get("20–25 km", None)
            if "30-35km" in dn or "30–35km" in dn:
                return fy_style.get("30–35 km", None)

        # 3) 实在没有就返回 None（不建议；你希望固定就把 COLOR_MAP 配全）
        return None

    # -------- 是否排序 --------
    if sort_datasets:
        datasets = sorted(list(datasets), key=lambda x: str(x))

    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")

    # -------- 参考（MWR）句柄 --------
    if include_ref and mode in ("taylor", "combined"):
        c_ref = color_map.get(ref_key, "red")
        ax.plot([], [], linestyle="None", marker="*", markersize=14,
                label=ref_label, color=c_ref, markerfacecolor=c_ref, markeredgecolor=c_ref)

    # -------- 数据集句柄 --------
    for d in datasets:
        d = str(d)
        mk = _infer_marker(d)
        ls = _infer_linestyle(d)
        lw = _infer_linewidth(d)
        c  = _infer_color(d)

        if mode == "taylor":
            ax.plot([], [], linestyle="None", marker=mk, markersize=10,
                    label=d, color=c, markerfacecolor=c, markeredgecolor=c)
        elif mode == "bias":
            ax.plot([], [], linestyle=ls, linewidth=lw,
                    label=d, color=c)
        elif mode == "combined":
            ax.plot([], [], linestyle=ls, linewidth=lw, marker=mk, markersize=8,
                    label=d, color=c, markerfacecolor=c, markeredgecolor=c)
        else:
            raise ValueError("mode must be 'taylor' | 'bias' | 'combined'")

    leg = ax.legend(
        loc="center", ncol=ncol, frameon=frameon, fontsize=fontsize,
        handlelength=2.4, handletextpad=0.8, columnspacing=1.2, borderaxespad=0.0
    )

    if title:
        if title_fontsize is None:
            title_fontsize = fontsize + 1
        leg.set_title(title, prop={"family": font_family, "size": title_fontsize})

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


datasets_bias = sorted(paired_df["dataset"].dropna().unique())
export_dataset_legend_only_colored(
    os.path.join(OUTPUT_DIR, "LEGEND_Bias_colored.png"),
    datasets=datasets_bias,
    mode="bias",
    include_ref=False,
    title="Datasets (Bias)"
)
datasets_taylor = sorted(metrics_df["dataset"].dropna().unique())
export_dataset_legend_only_colored(
    os.path.join(OUTPUT_DIR, "LEGEND_Taylor_colored.png"),
    datasets=datasets_taylor,
    mode="taylor",
    include_ref=True,
    title="Datasets (Taylor)"
)

# # =========================
# # 7) EXPORT PAIRED DATA: ONE EXCEL PER (BIN × LAYER), with T & q in SAME workbook
# #   - RH bins:    rh_bin × layer  -> one xlsx
# #   - CBH bins:   cbh_bin × layer -> one xlsx
# #   - In each xlsx: sheets by group × variable (T/q)
# #   - Add one Metrics sheet (filtered)
# # =========================

# import os
# import re
# import pandas as pd
# import numpy as np

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---- dataset -> group mapping (keep your original logic) ----
# group_map = {
#     'FY4B_<=6km': 'Satellite',
#     'FY4B_20-25km': 'Satellite',
#     'FY4B_30-35km': 'Satellite',
#     'Radiosonde': 'Radiosonde',
#     'CRA_near': 'Reanalysis',
#     'CRA_4pt': 'Reanalysis',
#     'ERA5_near': 'Reanalysis',
#     'ERA5_4pt': 'Reanalysis'
# }

# # ensure group column
# paired_df = paired_df.copy()
# paired_df["group"] = paired_df["dataset"].map(group_map).fillna("Other")

# # ---- columns to export (you can add/remove) ----
# # 每个文件已经按 layer 分开了，但保留 layer 列也没问题
# export_cols = [
#     "dataset", "group", "layer",
#     "BJT_aligned", "pressure_hPa",
#     "RH", "rh_bin",
#     "CBH", "cbh_bin",
#     "ref", "mod", "bias",
#     "RainRate_mmh", "RainAmt_mm", "rain_bin",

# ]

# # keep only existing cols (avoid KeyError if some cols missing)
# export_cols = [c for c in export_cols if c in paired_df.columns]

# # ---- helpers ----
# EXCEL_MAX_ROWS = 1048576

# def _sanitize_filename(s: str) -> str:
#     s = str(s)
#     s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
#     s = s.replace(" ", "")
#     return s[:180]

# def _safe_sheet(s: str) -> str:
#     s = str(s)
#     s = re.sub(r"[\[\]\*:/\\\?]+", "_", s)
#     return s[:31]

# def _write_df_chunked(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_base: str):
#     """
#     If df exceeds Excel row limit, split into multiple sheets: sheet_base, sheet_base_2, ...
#     """
#     if df.empty:
#         return
#     n = len(df)
#     if n <= EXCEL_MAX_ROWS:
#         df.to_excel(writer, sheet_name=_safe_sheet(sheet_base), index=False)
#         return

#     # split
#     n_parts = int(np.ceil(n / EXCEL_MAX_ROWS))
#     for i in range(n_parts):
#         a = i * EXCEL_MAX_ROWS
#         b = min((i + 1) * EXCEL_MAX_ROWS, n)
#         part = df.iloc[a:b].copy()
#         sheet_name = f"{sheet_base}_{i+1}"
#         part.to_excel(writer, sheet_name=_safe_sheet(sheet_name), index=False)

# def _export_one_book(df_sub: pd.DataFrame,
#                      metrics_sub: pd.DataFrame,
#                      out_path: str,
#                      title_note: str = ""):
#     """
#     df_sub: paired data already filtered to a specific (bin × layer)
#     metrics_sub: filtered metrics (same bin × layer), can be empty
#     """
#     if df_sub.empty:
#         return

#     # sort for readability
#     sort_cols = [c for c in ["variable", "group", "dataset", "BJT_aligned", "pressure_hPa"] if c in df_sub.columns]
#     if sort_cols:
#         df_sub = df_sub.sort_values(sort_cols)

#     # group order
#     group_order = ["Satellite", "Radiosonde", "Reanalysis", "Other"]
#     groups = [g for g in group_order if g in df_sub["group"].unique()] + \
#              [g for g in sorted(df_sub["group"].unique()) if g not in group_order]

#     with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#         # optional: a small info sheet
#         info = pd.DataFrame({
#             "key": ["note"],
#             "value": [title_note]
#         })
#         info.to_excel(writer, sheet_name="Info", index=False)

#         # write by group × variable (T/q) in SAME workbook
#         for grp in groups:
#             for var in ["T", "q"]:
#                 d = df_sub[(df_sub["group"] == grp) & (df_sub["variable"] == var)].copy()
#                 if d.empty:
#                     continue

#                 # choose columns
#                 cols = export_cols.copy()
#                 # variable列在sheet层面固定，可保留也可去掉；这里保留更清楚
#                 if "variable" in df_sub.columns and "variable" not in cols:
#                     cols = ["variable"] + cols
#                 if "variable" in cols:
#                     pass
#                 else:
#                     cols = (["variable"] + cols) if "variable" in df_sub.columns else cols

#                 # only existing columns
#                 cols = [c for c in cols if c in d.columns]
#                 d = d[cols]

#                 sheet = f"{grp[:3]}_{var}"  # Sat_T / Sat_q / Rea_T ...
#                 sheet = sheet.replace("Sat", "Sat").replace("Rad", "Rad").replace("Rea", "Rea")
#                 _write_df_chunked(writer, d, sheet)

#         # metrics sheet (包含T和q，variable列区分)
#         if metrics_sub is not None and (not metrics_sub.empty):
#             _write_df_chunked(writer, metrics_sub, "Metrics")

# # -------------------------
# # A) RH-binned export: one file per (rh_bin × layer)
# # -------------------------
# rh_out_dir = os.path.join(OUTPUT_DIR, "Export_RH_bins")
# os.makedirs(rh_out_dir, exist_ok=True)

# for rh_name, _, _ in RH_BINS:
#     for layer_name in LAYER_DEF.keys():
#         df_sub = paired_df[(paired_df.get("rh_bin") == rh_name) &
#                            (paired_df.get("layer") == layer_name)].copy()
#         if df_sub.empty:
#             continue

#         # metrics filtered (RH metrics_df uses column "rh_bin")
#         if "rh_bin" in metrics_df.columns:
#             metrics_sub = metrics_df[(metrics_df["rh_bin"] == rh_name) &
#                                      (metrics_df["layer"] == layer_name)].copy()
#         else:
#             metrics_sub = pd.DataFrame()

#         out_fn = _sanitize_filename(f"Paired_RH_{rh_name}_{layer_name}.xlsx")
#         out_path = os.path.join(rh_out_dir, out_fn)

#         note = f"RH bin = {rh_name}; layer = {layer_name}. T & q are in this workbook."
#         _export_one_book(df_sub, metrics_sub, out_path, title_note=note)

# print(f"[OK] RH-binned export done: {rh_out_dir}")

# # -------------------------
# # B) CBH-binned export: one file per (cbh_bin × layer)
# # -------------------------
# cbh_out_dir = os.path.join(OUTPUT_DIR, "Export_CBH_bins")
# os.makedirs(cbh_out_dir, exist_ok=True)

# # 如果你没有 metrics_cbh_df，就把下面 metrics_sub 设成空表即可
# if "metrics_cbh_df" not in globals():
#     metrics_cbh_df = pd.DataFrame()

# for cbh_name, _, _ in CBH_BINS:
#     for layer_name in LAYER_DEF.keys():
#         df_sub = paired_df[(paired_df.get("cbh_bin") == cbh_name) &
#                            (paired_df.get("layer") == layer_name)].copy()
#         if df_sub.empty:
#             continue

#         # metrics filtered (CBH metrics uses column "cbh_bin")
#         if (not metrics_cbh_df.empty) and ("cbh_bin" in metrics_cbh_df.columns):
#             metrics_sub = metrics_cbh_df[(metrics_cbh_df["cbh_bin"] == cbh_name) &
#                                          (metrics_cbh_df["layer"] == layer_name)].copy()
#         else:
#             metrics_sub = pd.DataFrame()

#         out_fn = _sanitize_filename(f"Paired_CBH_{cbh_name}_{layer_name}.xlsx")
#         out_path = os.path.join(cbh_out_dir, out_fn)

#         note = f"CBH bin = {cbh_name}; layer = {layer_name}. T & q are in this workbook."
#         _export_one_book(df_sub, metrics_sub, out_path, title_note=note)

# print(f"[OK] CBH-binned export done: {cbh_out_dir}")

# # -------------------------
# # C) RAIN-binned export: one file per (rain_bin × layer)
# # -------------------------
# rain_out_dir = os.path.join(OUTPUT_DIR, "Export_RAIN_bins")
# os.makedirs(rain_out_dir, exist_ok=True)

# if "metrics_rain_df" not in globals():
#     metrics_rain_df = pd.DataFrame()

# for rain_name, _, _ in RAIN_BINS:
#     for layer_name in LAYER_DEF.keys():
#         df_sub = paired_df[(paired_df.get("rain_bin") == rain_name) &
#                            (paired_df.get("layer") == layer_name)].copy()
#         if df_sub.empty:
#             continue

#         if (not metrics_rain_df.empty) and ("rain_bin" in metrics_rain_df.columns):
#             metrics_sub = metrics_rain_df[(metrics_rain_df["rain_bin"] == rain_name) &
#                                           (metrics_rain_df["layer"] == layer_name)].copy()
#         else:
#             metrics_sub = pd.DataFrame()

#         out_fn = _sanitize_filename(f"Paired_RAIN_{rain_name}_{layer_name}.xlsx")
#         out_path = os.path.join(rain_out_dir, out_fn)

#         note = (f"RAIN bin = {rain_name}; layer = {layer_name}. "
#                 f"RAIN_BIN_MODE={RAIN_BIN_MODE}. T & q are in this workbook.")
#         _export_one_book(df_sub, metrics_sub, out_path, title_note=note)

# print(f"[OK] RAIN-binned export done: {rain_out_dir}")








































# # -*- coding: utf-8 -*-
# """
# MWR vs FY4B / ERA5 / CRA / Radiosonde comparison (300–850 hPa)
# - Two periods
# - 3 layers: 300–500, 500–700, 700–850 hPa
# - Metrics: STD, R, RMSE, Tian et al. (2017) metric T
# - Taylor diagrams + bias-profile uncertainty plot
# - Export: one Excel with 2 sheets: PairedData, Metrics

# Assumes these DataFrames already exist in memory:
#   fy4B_T, fy4B_q                 # FY4B long: BJT, pressure_hPa, distance_km, T_K / q_gkg
#   temp_p_df, q_p_gkg_df          # MWR wide: time + pressure columns
#   cra_tq_near_filtered, cra_tq_4pt_filtered
#   era5_tq_near_filtered, era5_tq_4pt_filtered
#   radiosonde                     # columns: time, pressure_hPa, temperature_C, mixing ratio_g/kg

# Also assumes these exist if you enable RH/CBH/RAIN binning:
#   rh_mean_8_320   # RH 1-min
#   cbh_1min        # CBH 1-min (or similar)
#   met_1min        # met 1-min with rainfall_rate_mmh
# """



# # =========================
# # 0) USER SETTINGS
# # =========================
# OUTPUT_DIR = r"E:\Beijing2024\MWR_compare_outputs-rh_cbh——cra ref"   # <<< change if needed
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -------------------------
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # REF SWITCH (唯一开关：改这里即可)
# # -------------------------
# # 可选： "MWR" | "CRA_near" | "CRA_4pt"
# REF_DATASET = "CRA_near"
# # 例：REF_DATASET = "CRA_near"
# # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# # -------------------------

# # two periods (inclusive end day)
# periods = {
#     "P1_20240606_20240609": ("2024-06-06", "2024-06-09"),
#     "P2_20240805_20240810": ("2024-08-05", "2024-08-10"),
# }

# # time alignment to MWR base
# TIME_FREQ = "60min"            # alignment frequency
# TIME_ALIGN = "round"           # "round" | "floor" | "ceil"
# MAX_TIME_DIFF = pd.Timedelta("45min")  # for asof matching if you enable it (not used now)

# # pressure range & layers
# P_MIN, P_MAX = 300.0, 850.0

# LAYER_DEF = {
#     "300-500": (300.0, 500.0),
#     "500-700": (500.0, 700.0),
#     "700-850": (700.0, 850.0),
# }

# MIN_PAIRS = 3       # minimum paired samples to compute metrics
# BOOT_N = 400         # bootstrap resamples for CI

# # plotting style
# rcParams["font.family"] = "Times New Roman"
# rcParams["font.size"] = 12


# # =========================
# # 1) HELPERS
# # =========================
# def _to_datetime(s):
#     return pd.to_datetime(s, errors="coerce")

# def align_time_series(dt: pd.Series, freq=TIME_FREQ, how=TIME_ALIGN) -> pd.Series:
#     dt = _to_datetime(dt)
#     if how == "round":
#         return dt.dt.round(freq)
#     if how == "floor":
#         return dt.dt.floor(freq)
#     if how == "ceil":
#         return dt.dt.ceil(freq)
#     raise ValueError("TIME_ALIGN must be 'round'/'floor'/'ceil'")

# def mwr_wide_to_long(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
#     """
#     Robust MWR wide -> long:
#       output columns: BJT_aligned, pressure_hPa, value_name
#     Supports:
#       - time in DatetimeIndex
#       - or time in a column (BJT/Timestamp_BJT/time/datetime/...)
#     """
#     df = df_wide.copy()

#     # ---- 1) get time column as 'BJT' ----
#     time_candidates = ["BJT", "Timestamp_BJT", "timestamp_bjt", "time", "Time",
#                        "datetime", "Datetime", "DATE", "Date"]

#     if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
#         df = df.reset_index()
#         if "BJT" not in df.columns:
#             df = df.rename(columns={df.columns[0]: "BJT"})
#         df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")
#     else:
#         found = None
#         for c in time_candidates:
#             if c in df.columns:
#                 found = c
#                 break

#         if found is None:
#             for c in df.columns:
#                 parsed = pd.to_datetime(df[c], errors="coerce")
#                 if parsed.notna().sum() >= 0.7 * len(df):
#                     found = c
#                     df[c] = parsed
#                     break

#         if found is None:
#             raise ValueError("MWR表既没有DatetimeIndex，也没有可识别的时间列（如BJT/time）。")

#         if found != "BJT":
#             df = df.rename(columns={found: "BJT"})
#         df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")

#     df = df.dropna(subset=["BJT"])

#     # ---- 2) detect pressure columns ----
#     pres_cols = []
#     pres_vals = []
#     for c in df.columns:
#         if c == "BJT":
#             continue
#         try:
#             p = float(str(c).strip())
#             if (p >= P_MIN) and (p <= P_MAX):
#                 pres_cols.append(c)
#                 pres_vals.append(p)
#         except Exception:
#             continue

#     if len(pres_cols) == 0:
#         raise ValueError("未找到气压列：列名需可转成数值，且在300–850hPa内。")

#     # ---- 3) melt ----
#     long_df = df.melt(id_vars=["BJT"], value_vars=pres_cols,
#                       var_name="pressure_hPa", value_name=value_name)

#     long_df["pressure_hPa"] = long_df["pressure_hPa"].astype(str).str.strip()
#     long_df["pressure_hPa"] = pd.to_numeric(long_df["pressure_hPa"], errors="coerce")
#     long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
#     long_df = long_df.dropna(subset=["pressure_hPa", value_name])

#     long_df["BJT_aligned"] = align_time_series(long_df["BJT"])
#     return long_df[["BJT_aligned", "pressure_hPa", value_name]]

# def sat_bin_mean(df_long: pd.DataFrame, value_col: str, out_name_prefix: str) -> dict:
#     """
#     FY4B long: BJT, pressure_hPa, distance_km, value_col
#     Return dict of {dataset_name: long_df(BJT_aligned, pressure_hPa, value)}
#     """
#     bins = {
#         f"{out_name_prefix}_<=6km": (0.0, 6.0),
#         f"{out_name_prefix}_20-25km": (20.0, 25.0),
#         f"{out_name_prefix}_30-35km": (30.0, 35.0),
#     }

#     d = {}
#     base = df_long.copy()
#     base["BJT"] = _to_datetime(base["BJT"])
#     base["BJT_aligned"] = align_time_series(base["BJT"])
#     base["pressure_hPa"] = pd.to_numeric(base["pressure_hPa"], errors="coerce")
#     base["distance_km"] = pd.to_numeric(base["distance_km"], errors="coerce")
#     base[value_col] = pd.to_numeric(base[value_col], errors="coerce")

#     for name, (lo, hi) in bins.items():
#         sub = base[(base["distance_km"] >= lo) & (base["distance_km"] <= hi)].copy()
#         if sub.empty:
#             d[name] = pd.DataFrame(columns=["BJT_aligned", "pressure_hPa", value_col])
#             continue
#         g = (sub.groupby(["BJT_aligned", "pressure_hPa"], as_index=False)[value_col]
#              .mean())
#         d[name] = g.dropna()
#     return d

# def reanalysis_to_long(df: pd.DataFrame, time_col: str, p_col: str,
#                        t_col: str = None, q_col: str = None,
#                        out_t: str = "T_K", out_q: str = "q_gkg") -> dict:
#     """
#     Convert reanalysis long to standardized long tables.
#     Return {"T": df_T, "q": df_q} with columns:
#       BJT_aligned, pressure_hPa, value
#     """
#     x = df.copy()
#     x[time_col] = _to_datetime(x[time_col])
#     x["BJT_aligned"] = align_time_series(x[time_col])
#     x["pressure_hPa"] = pd.to_numeric(x[p_col], errors="coerce")

#     out = {}
#     if t_col is not None:
#         xt = x[["BJT_aligned", "pressure_hPa", t_col]].rename(columns={t_col: out_t}).copy()
#         xt[out_t] = pd.to_numeric(xt[out_t], errors="coerce")
#         out["T"] = xt.dropna()
#     if q_col is not None:
#         xq = x[["BJT_aligned", "pressure_hPa", q_col]].rename(columns={q_col: out_q}).copy()
#         xq[out_q] = pd.to_numeric(xq[out_q], errors="coerce")
#         out["q"] = xq.dropna()
#     return out

# def radiosonde_to_long(rs: pd.DataFrame) -> dict:
#     """
#     radiosonde columns:
#       time, pressure_hPa, temperature_C, mixing ratio_g/kg
#     Convert to:
#       T_K, q_gkg (specific humidity g/kg from mixing ratio)
#     """
#     x = rs.copy()
#     x["time"] = _to_datetime(x["time"])
#     x["BJT_aligned"] = align_time_series(x["time"])
#     x["pressure_hPa"] = pd.to_numeric(x["pressure_hPa"], errors="coerce")

#     if "temperature_C" in x.columns:
#         x["T_K"] = pd.to_numeric(x["temperature_C"], errors="coerce") + 273.15
#     else:
#         x["T_K"] = np.nan

#     # mixing ratio -> specific humidity
#     if "mixing ratio_g/kg" in x.columns:
#         w = pd.to_numeric(x["mixing ratio_g/kg"], errors="coerce") / 1000.0  # kg/kg
#         q = w / (1.0 + w)  # kg/kg
#         x["q_gkg"] = q * 1000.0
#     else:
#         x["q_gkg"] = np.nan

#     return {
#         "T": x[["BJT_aligned", "pressure_hPa", "T_K"]].dropna(),
#         "q": x[["BJT_aligned", "pressure_hPa", "q_gkg"]].dropna(),
#     }

# def filter_union_periods(df: pd.DataFrame, periods_dict: dict, value_col: str) -> pd.DataFrame:
#     """用原来的两个时间段做“时间范围限制”"""
#     if df.empty:
#         return df
#     masks = []
#     for _, (start, end) in periods_dict.items():
#         st = pd.Timestamp(start)
#         ed = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
#         masks.append((df["BJT_aligned"] >= st) & (df["BJT_aligned"] <= ed))
#     m = np.logical_or.reduce(masks) if masks else np.ones(len(df), dtype=bool)
#     y = df.loc[m].copy()
#     y[value_col] = pd.to_numeric(y[value_col], errors="coerce")
#     return y.dropna(subset=[value_col])

# def add_layer(p):
#     for lname, (lo, hi) in LAYER_DEF.items():
#         if (p >= lo) and (p <= hi):
#             return lname
#     return np.nan

# def compute_metrics(ref: np.ndarray, mod: np.ndarray) -> dict:
#     """
#     STD, R, RMSE + Tian et al. (2017) metric T
#     bias = mean(mod - ref)
#     """
#     ref = np.asarray(ref, dtype=float)
#     mod = np.asarray(mod, dtype=float)

#     m = np.isfinite(ref) & np.isfinite(mod)
#     ref = ref[m]
#     mod = mod[m]
#     n = ref.size
#     if n < MIN_PAIRS:
#         return dict(n=n, std_ref=np.nan, std_mod=np.nan, r=np.nan,
#                     bias=np.nan, mse=np.nan, rmse=np.nan, T=np.nan)

#     bias = float(np.mean(mod - ref))
#     mse = float(np.mean((mod - ref) ** 2))
#     rmse = float(np.sqrt(mse))

#     std_ref = float(np.std(ref, ddof=0))
#     std_mod = float(np.std(mod, ddof=0))

#     if std_ref == 0.0 and std_mod == 0.0:
#         r = 1.0 if mse == 0.0 else 0.0
#     elif std_ref == 0.0 or std_mod == 0.0:
#         r = 0.0
#     else:
#         r = float(np.corrcoef(mod, ref)[0, 1])

#     sf2 = std_mod ** 2
#     sr2 = std_ref ** 2
#     denom = (bias ** 2) + sf2 + sr2

#     if denom == 0.0:
#         T = 1.0 if mse == 0.0 else 0.0
#     else:
#         term1 = (1.0 + r) / 2.0
#         term2 = 1.0 - (mse / denom)
#         term1 = float(np.clip(term1, 0.0, 1.0))
#         term2 = float(np.clip(term2, 0.0, 1.0))
#         T = term1 * term2

#     return dict(n=n, std_ref=std_ref, std_mod=std_mod, r=r,
#                 bias=bias, mse=mse, rmse=rmse, T=T)

# def build_pairs(ref_df: pd.DataFrame, ref_col: str,
#                 tar_df: pd.DataFrame, tar_col: str,
#                 rh_hourly: pd.DataFrame = None,
#                 cbh_hourly: pd.DataFrame = None,
#                 rain_hourly: pd.DataFrame = None) -> pd.DataFrame:
#     """
#     Exact matching on (BJT_aligned, pressure_hPa).
#     同时 merge RH(按小时) 与 CBH(按小时) 与 RAIN(按小时)，并产生分箱。
#     """
#     a = ref_df.rename(columns={ref_col: "ref"}).copy()
#     b = tar_df.rename(columns={tar_col: "mod"}).copy()
#     out = pd.merge(a, b, on=["BJT_aligned", "pressure_hPa"], how="inner")

#     # --- RH ---
#     if rh_hourly is not None and (not rh_hourly.empty):
#         out = pd.merge(out, rh_hourly[["BJT_aligned", "RH"]], on="BJT_aligned", how="left")
#         out["rh_bin"] = out["RH"].apply(assign_rh_bin)
#     else:
#         out["RH"] = np.nan
#         out["rh_bin"] = np.nan

#     # --- CBH ---
#     if cbh_hourly is not None and (not cbh_hourly.empty):
#         out = pd.merge(out, cbh_hourly[["BJT_aligned", "CBH"]], on="BJT_aligned", how="left")
#         out["cbh_bin"] = out["CBH"].apply(assign_cbh_bin)
#     else:
#         out["CBH"] = np.nan
#         out["cbh_bin"] = np.nan

#     # --- RAIN ---
#     if rain_hourly is not None and (not rain_hourly.empty):
#         cols = ["BJT_aligned"]
#         if "RainRate_mmh" in rain_hourly.columns:
#             cols.append("RainRate_mmh")
#         if "RainAmt_mm" in rain_hourly.columns:
#             cols.append("RainAmt_mm")

#         out = pd.merge(out, rain_hourly[cols], on="BJT_aligned", how="left")

#         if "RainRate_mmh" not in out.columns:
#             out["RainRate_mmh"] = np.nan
#         if "RainAmt_mm" not in out.columns:
#             out["RainAmt_mm"] = np.nan

#         out["rain_bin"] = out.apply(lambda r: assign_rain_bin(r.get("RainRate_mmh", np.nan),
#                                                               r.get("RainAmt_mm", np.nan)), axis=1)
#     else:
#         out["RainRate_mmh"] = np.nan
#         out["RainAmt_mm"] = np.nan
#         out["rain_bin"] = np.nan

#     out["layer"] = out["pressure_hPa"].apply(add_layer)
#     out = out.dropna(subset=["layer", "ref", "mod"])
#     out["bias"] = out["mod"] - out["ref"]  # <<< 关键：bias = mod - ref（保留不变）
#     return out


# # =========================
# # RH / CBH / RAIN helpers
# # =========================
# RAIN_COL = "rainfall_rate_mmh"   # met_1min 里降水强度列（mm/h）
# RAIN_BIN_MODE = "rate"           # "rate" 或 "amount"

# RAIN_BINS = [
#     ("RAIN_0",          0.0,   0.0),
#     ("RAIN_0.01-19.99", 0.01,  np.inf),
#     #("RAIN_20+",        20,    np.inf),
# ]

# CBH_COL = "CBH"   # <<< 你CBH数据里云底高列名（单位：m）

# CBH_BINS = [
#     ("CBH_0-1500",      0.0,    1500.0),
#     ("CBH_1500-5000",   1500.0, 5000.0),
#     ("CBH_5000-10000",  5000.0, 10000.0),
# ]

# RH_COL = "RH_mean_8_320"   # 你的湿度列名（百分比 0-100）

# RH_BINS = [
#     ("RH_0-60",    0.0,  60.0),
#     ("RH_60-80",   60.0, 80.0),
#     ("RH_80-100",  80.0, 100.0),
# ]

# def cbh_to_aligned(cbh_obj, value_col=CBH_COL, agg="median") -> pd.DataFrame:
#     """
#     CBH(1-min or irregular) -> hourly aligned:
#     return columns: BJT_aligned, CBH
#     """
#     if isinstance(cbh_obj, pd.Series):
#         df = cbh_obj.to_frame(name=value_col).copy()
#         df = df.reset_index().rename(columns={df.columns[0]: "BJT"})
#     else:
#         df = cbh_obj.copy()
#         if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
#             df = df.reset_index()
#             if "BJT" not in df.columns:
#                 df = df.rename(columns={df.columns[0]: "BJT"})
#         else:
#             time_candidates = ["BJT", "Timestamp_BJT", "timestamp_bjt", "time", "Time",
#                                "datetime", "Datetime", "DATE", "Date"]
#             found = None
#             for c in time_candidates:
#                 if c in df.columns:
#                     found = c
#                     break
#             if found is None:
#                 for c in df.columns:
#                     parsed = pd.to_datetime(df[c], errors="coerce")
#                     if parsed.notna().sum() >= 0.7 * len(df):
#                         found = c
#                         df[c] = parsed
#                         break
#             if found is None:
#                 raise ValueError("CBH 数据找不到时间列（BJT/time/...），也不是 DatetimeIndex。")
#             if found != "BJT":
#                 df = df.rename(columns={found: "BJT"})

#     if value_col not in df.columns:
#         raise ValueError(f"CBH 数据里找不到云底高列: {value_col}")

#     df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")
#     df = df.dropna(subset=["BJT"])
#     df["CBH"] = pd.to_numeric(df[value_col], errors="coerce")
#     df = df.dropna(subset=["CBH"])
#     df["BJT_aligned"] = align_time_series(df["BJT"])

#     if agg == "mean":
#         out = df.groupby("BJT_aligned", as_index=False)["CBH"].mean()
#     elif agg == "median":
#         out = df.groupby("BJT_aligned", as_index=False)["CBH"].median()
#     elif agg == "min":
#         out = df.groupby("BJT_aligned", as_index=False)["CBH"].min()
#     else:
#         raise ValueError("agg must be one of: mean/median/min")
#     return out

# def assign_cbh_bin(cbh: float):
#     if not np.isfinite(cbh):
#         return np.nan
#     if cbh < 1500:
#         return "CBH_0-1500"
#     elif cbh < 5000:
#         return "CBH_1500-5000"
#     elif cbh <= 10000:
#         return "CBH_5000-10000"
#     return np.nan

# def rh_to_aligned(rh_obj, value_col=RH_COL) -> pd.DataFrame:
#     """
#     RH(1-min) -> hourly aligned:
#     return columns: BJT_aligned, RH
#     """
#     if isinstance(rh_obj, pd.Series):
#         df = rh_obj.to_frame(name=value_col).copy()
#         df = df.reset_index().rename(columns={df.columns[0]: "BJT"})
#     else:
#         df = rh_obj.copy()
#         if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
#             df = df.reset_index()
#             if "BJT" not in df.columns:
#                 df = df.rename(columns={df.columns[0]: "BJT"})
#         else:
#             time_candidates = ["BJT", "Timestamp_BJT", "timestamp_bjt", "time", "Time",
#                                "datetime", "Datetime", "DATE", "Date"]
#             found = None
#             for c in time_candidates:
#                 if c in df.columns:
#                     found = c
#                     break
#             if found is None:
#                 for c in df.columns:
#                     parsed = pd.to_datetime(df[c], errors="coerce")
#                     if parsed.notna().sum() >= 0.7 * len(df):
#                         found = c
#                         df[c] = parsed
#                         break
#             if found is None:
#                 raise ValueError("RH 数据找不到时间列（BJT/time/...），也不是 DatetimeIndex。")
#             if found != "BJT":
#                 df = df.rename(columns={found: "BJT"})

#     if value_col not in df.columns:
#         raise ValueError(f"RH 数据里找不到湿度列: {value_col}")

#     df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")
#     df = df.dropna(subset=["BJT"])
#     df["RH"] = pd.to_numeric(df[value_col], errors="coerce")
#     df = df.dropna(subset=["RH"])
#     df["BJT_aligned"] = align_time_series(df["BJT"])

#     out = df.groupby("BJT_aligned", as_index=False)["RH"].mean()
#     return out

# def assign_rh_bin(rh: float):
#     if not np.isfinite(rh):
#         return np.nan
#     if rh < 60:
#         return "RH_0-60"
#     elif rh < 80:
#         return "RH_60-80"
#     elif rh <= 100:
#         return "RH_80-100"
#     return np.nan

# def rain_to_aligned(met_obj, value_col=RAIN_COL) -> pd.DataFrame:
#     """
#     met_1min -> hourly aligned rainfall:
#     return columns: BJT_aligned, RainRate_mmh, RainAmt_mm
#     """
#     if isinstance(met_obj, pd.Series):
#         df = met_obj.to_frame(name=value_col).copy()
#         df = df.reset_index().rename(columns={df.columns[0]: "BJT"})
#     else:
#         df = met_obj.copy()
#         if isinstance(df.index, pd.DatetimeIndex) or np.issubdtype(df.index.dtype, np.datetime64):
#             df = df.reset_index()
#             if "BJT" not in df.columns:
#                 df = df.rename(columns={df.columns[0]: "BJT"})
#         else:
#             time_candidates = ["BJT", "Timestamp_BJT", "timestamp_bjt", "time", "Time",
#                                "datetime", "Datetime", "DATE", "Date"]
#             found = None
#             for c in time_candidates:
#                 if c in df.columns:
#                     found = c
#                     break
#             if found is None:
#                 for c in df.columns:
#                     parsed = pd.to_datetime(df[c], errors="coerce")
#                     if parsed.notna().sum() >= 0.7 * len(df):
#                         found = c
#                         df[c] = parsed
#                         break
#             if found is None:
#                 raise ValueError("降水数据找不到时间列（BJT/time/...），也不是 DatetimeIndex。")
#             if found != "BJT":
#                 df = df.rename(columns={found: "BJT"})

#     if value_col not in df.columns:
#         raise ValueError(f"降水数据里找不到列: {value_col}")

#     df["BJT"] = pd.to_datetime(df["BJT"], errors="coerce")
#     df = df.dropna(subset=["BJT"])

#     rr = pd.to_numeric(df[value_col], errors="coerce").clip(lower=0)
#     df["RainRate_mmh_1min"] = rr
#     df = df.dropna(subset=["RainRate_mmh_1min"])

#     df["BJT_aligned"] = align_time_series(df["BJT"])

#     out_rate = df.groupby("BJT_aligned", as_index=False)["RainRate_mmh_1min"].mean()
#     out_rate = out_rate.rename(columns={"RainRate_mmh_1min": "RainRate_mmh"})

#     out_amt = df.groupby("BJT_aligned", as_index=False)["RainRate_mmh_1min"].sum()
#     out_amt["RainAmt_mm"] = out_amt["RainRate_mmh_1min"] / 60.0
#     out_amt = out_amt[["BJT_aligned", "RainAmt_mm"]]

#     out = pd.merge(out_rate, out_amt, on="BJT_aligned", how="outer").sort_values("BJT_aligned")
#     return out

# def assign_rain_bin(rate_mmh: float, amount_mm: float = None):
#     x = amount_mm if RAIN_BIN_MODE == "amount" else rate_mmh
#     if x is None or (not np.isfinite(x)):
#         return np.nan
#     if x <= 0:
#         return "RAIN_0"
#     # 把 0<x<=19.99 都算作下雨，避免 0~0.01 漏掉
#     if x <= 19.99:
#         return "RAIN_0.01-19.99"
#     return "RAIN_20+"


# # =========================
# # Fixed style for Taylor plots (order -> fixed marker + fixed color)
# # =========================
# TAYLOR_FIXED_ORDER = [
#     "FY4B_all",
#     "FY4B_<=6km",
#     "FY4B_20-25km",
#     "FY4B_30-35km",
#     "Radiosonde",
#     "ERA5_near",
#     "ERA5_4pt",
#     "CRA_near",
#     "CRA_4pt",
#     "MWR",
# ]

# # 与 fixed_order 一一对应（长度必须一致）
# TAYLOR_FIXED_MARKERS = [
#     "v",   # FY4B_all
#     "^",   # FY4B_<=6km
#     "^",   # FY4B_20-25km
#     "^",   # FY4B_30-35km
#     "*",   # Radiosonde
#     ".",   # ERA5_near
#     ".",   # ERA5_4pt
#     ".",   # CRA_near
#     ".",   # CRA_4pt
#     "h",   # MWR
# ]

# _palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# TAYLOR_FIXED_COLORS = [_palette[i % len(_palette)] for i in range(len(TAYLOR_FIXED_ORDER))]

# TAYLOR_COLOR_MAP  = {ds: TAYLOR_FIXED_COLORS[i]  for i, ds in enumerate(TAYLOR_FIXED_ORDER)}
# TAYLOR_MARKER_MAP = {ds: TAYLOR_FIXED_MARKERS[i] for i, ds in enumerate(TAYLOR_FIXED_ORDER)}
# TAYLOR_ORDER_IDX  = {ds: i for i, ds in enumerate(TAYLOR_FIXED_ORDER)}

# def sort_by_fixed_order(df, col="dataset"):
#     # 未在 fixed_order 的放到最后
#     return (df.assign(_ord=df[col].map(TAYLOR_ORDER_IDX).fillna(999).astype(int))
#               .sort_values("_ord")
#               .drop(columns="_ord"))


# # ---- Taylor diagram helper (styled version, backward compatible) ----
# class TaylorDiagram:
#     def __init__(self, ref_std=1.0, fig=None, rect=111, label="Ref:MWR", srange=(0, 1.8),
#                  *,
#                  # 字体：不传则自动兜底；传入则强制用你给的路径
#                  font_times_path=None,
#                  font_chinese_path=None,
#                  # 样式参数（不影响外部调用；默认与原类一致）
#                  r_interval=0.25,
#                  major_corrs=(0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
#                  minor_corrs=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
#                               0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
#                               0.97, 0.98, 0.99, 1.0),
#                  rms_levels=(0.5, 0.75, 1.0, 1.25, 1.5),    # 蓝色：以(1,0)为圆心的CRMSD圆
#                  std_levels=(0.5, 0.75, 1.0, 1.25, 1.5),         # 红色：以(0,0)为圆心的STD圆
#                  corr_guides=(0.4, 0.6, 0.8, 0.9, 0.95, 0.99)  # 绿色：相关系数辅助线
#                  ):

#         self.ref_std = ref_std

#         if fig is None:
#             fig = plt.figure(figsize=(8, 8))
#         self.fig = fig
#         self.ax = fig.add_subplot(rect, polar=True)

#         # 增加主轴线（0°和90°）的线宽

#         # ------------------ 字体：自动兜底避免 FileNotFoundError ------------------
#         def _pick_existing(paths):
#             for p in paths:
#                 if p and os.path.exists(p):
#                     return p
#             return None

#         if font_times_path is None:
#             font_times_path = _pick_existing([
#                 # Windows 常见
#                 r"C:\Windows\Fonts\times.ttf",
#                 r"C:\Windows\Fonts\times.ttf",
#                 # Linux 你之前的路径（如果确实存在）
#                 "/home/mw/input/font1842/Times.ttf",
#             ])

#         if font_chinese_path is None:
#             font_chinese_path = _pick_existing([
#                 # Windows 常见：宋体多为 ttc
#                 r"C:\Windows\Fonts\simsun.ttc",
#                 r"C:\Windows\Fonts\simsun.ttf",
#                 # Linux 你之前的路径（如果确实存在）
#                 "/home/mw/input/font1842/SimSun.ttf",
#             ])

#         self._fp_times = FontProperties(fname=font_times_path) if font_times_path else FontProperties()
#         self._fp_cn    = FontProperties(fname=font_chinese_path) if font_chinese_path else FontProperties()

#         # ------------------ 坐标系（保持你原类的设定） ------------------
#         self.ax.set_theta_zero_location("E")
#         self.ax.set_theta_direction(1)
#         self.ax.set_thetamin(0)
#         self.ax.set_thetamax(90)
#         # 0° 和 90° 边界轴线宽度（单位：points）
#         self.ax.spines["start"].set_linewidth(2)  # 0°
#         self.ax.spines["end"].set_linewidth(2)    # 90°

#         # 如果你还想把外侧圆弧边界也加粗（可选）
#         self.ax.spines["polar"].set_linewidth(2)

#         # ------------------ 相关系数刻度（主/次刻度 + 外圈短刻度线） ------------------
#         major_corrs = np.array(major_corrs, dtype=float)
#         minor_corrs = np.array(minor_corrs, dtype=float)

#         major_thetas = np.arccos(np.clip(major_corrs, -1, 1))
#         minor_thetas = np.arccos(np.clip(minor_corrs, -1, 1))

#         self.ax.set_xticks(major_thetas)
#         # 显示与示例一致：直接显示 corr 值
#         self.ax.set_xticklabels([f"{c:g}" for c in major_corrs])

#         for lab in self.ax.get_xticklabels():
#             lab.set_fontproperties(self._fp_times)
#             lab.set_fontsize(20)

#         # ------------------ r 范围 + 手工 r 标注（REF/数值） ------------------
#         r_small, r_big = float(srange[0]), float(srange[1])
#         self.ax.set_rlim(r_small, r_big)

#         # 去掉默认 rgrid，使用你示例那种“在两条轴上标注”
#         self.ax.set_rgrids([])

#         rr = np.arange(r_small, r_big + 1e-9, r_interval)
#         for r in rr:
#             if abs(r - 1.0) < 1e-12:
#                 txt = "\nRef:MWR"
#             else:
#                 txt = "\n" + (f"{r:g}")
#             # theta=0 轴
#             self.ax.text(0, r, s=txt, fontproperties=self._fp_times, fontsize=20,
#                          ha="center", va="top")
#             # theta=90° 边界
#             self.ax.text(np.pi / 2, r, s=(f"{r:g}  "), fontproperties=self._fp_times, fontsize=20,
#                          ha="right", va="center")

#         # 不要默认网格
#         self.ax.grid(False)

#         # 外圈短刻度线（主/次）
#         angle_linewidth, angle_length, angle_minor_length = 2, 0.02, 0.01
#         tick = [self.ax.get_rmax(), self.ax.get_rmax() * (1 - angle_length)]
#         tick_minor = [self.ax.get_rmax(), self.ax.get_rmax() * (1 - angle_minor_length)]
#         self.ax.plot([0, 0], [0, self.ax.get_rmax()], lw=2, color="k")  # 0°水平线
#         self.ax.plot([np.pi/2, np.pi/2], [0, self.ax.get_rmax()], lw=2, color="k")  # 90°垂直线
#         for t in major_thetas:
#             self.ax.plot([t, t], tick, lw=angle_linewidth, color="k")
#         for t in minor_thetas:
#             self.ax.plot([t, t], tick_minor, lw=angle_linewidth, color="k")

#         # ------------------ 圆弧：CRMSD（蓝）+ STD（红） ------------------
#         # 关键：用 ax.transData._b 把“笛卡尔坐标圆”画在极坐标图上（与你示例一致）
#         for lv in rms_levels:
#             c = Circle((1, 0), lv, transform=self.ax.transData._b,
#                        facecolor=(0, 0, 0, 0), edgecolor="blue",
#                        linestyle=(0, (5, 5)), linewidth=1)
#             self.ax.add_artist(c)

#         for lv in std_levels:
#             c = Circle((0, 0), lv, transform=self.ax.transData._b,
#                        facecolor=(0, 0, 0, 0), edgecolor="red",
#                        linestyle=(0, (5, 5)), linewidth=1)
#             self.ax.add_artist(c)

#         # ------------------ 相关系数辅助线（绿虚线） ------------------
#         for c in corr_guides:
#             th = float(np.arccos(np.clip(c, -1, 1)))
#             self.ax.plot([0, th], [0, self.ax.get_rmax()], lw=1, color="green", linestyle=(0, (5, 5)))

#         # 轴注记（与你示例一致）
#         #self.ax.set_ylabel("Standard Deviation", fontproperties=self._fp_times, color="red", fontweight='bold',labelpad=45, fontsize=20)
#         #self.ax.set_xlabel("RMSE", fontproperties=self._fp_times, labelpad=45, color="blue",fontsize=20, fontweight='bold')
#         self.ax.text(np.deg2rad(40), self.ax.get_rmax() + 0.00, s="Correlation Coefficients",
#                       fontproperties=self._fp_times, fontsize=20,color="green", fontweight='bold',
#                       ha="center", va="bottom", rotation=-45)
        
#         # ---- 控制左右（x）与上下（y）位置：0~1 是轴宽/高的比例 ----
#         self.ax.xaxis.set_label_coords(0.55, -0.07)  # x<0.5 往左；x>0.5 往右
        
#         # REF 点（保持原接口/含义：std_ratio=1, corr=1 => theta=0, r=1）
#         self.ax.plot([0], [1.0], marker="*", markersize=18, label=label)

#     def add_sample(self, std_ratio, corrcoef, label, marker="o", color=None):
#         theta = np.arccos(np.clip(corrcoef, -1, 1))
#         self.ax.plot([theta], [std_ratio],
#                      marker=marker, markersize=14,
#                      linestyle="None", label=label,
#                      color=color)


# def bootstrap_mean_ci(x: np.ndarray, n_boot=BOOT_N, alpha=0.05):
#     x = np.asarray(x, dtype=float)
#     x = x[np.isfinite(x)]
#     n = x.size
#     if n < MIN_PAIRS:
#         return np.nan, np.nan, np.nan
#     rng = np.random.default_rng(2025)
#     means = np.empty(n_boot, dtype=float)
#     for i in range(n_boot):
#         samp = rng.choice(x, size=n, replace=True)
#         means[i] = np.mean(samp)
#     lo = np.quantile(means, alpha/2)
#     hi = np.quantile(means, 1 - alpha/2)
#     return float(np.mean(x)), float(lo), float(hi)


# # =========================
# # 2) BUILD STANDARDIZED LONG TABLES
# # =========================
# # MWR long
# mwr_T = mwr_wide_to_long(temp_p_df, "T_K")
# mwr_q = mwr_wide_to_long(q_p_gkg_df, "q_gkg")

# # 标准压力：仍使用 MWR（保持你原脚本功能/参数不变）
# mwr_pressures = np.array(sorted([p for p in mwr_T["pressure_hPa"].unique()
#                                  if (p >= P_MIN and p <= P_MAX)]), dtype=float)

# # FY4B rings
# fy4B_T_bins = sat_bin_mean(fy4B_T, "T_K", "FY4B_T")
# fy4B_q_bins = sat_bin_mean(fy4B_q, "q_gkg", "FY4B_q")

# # CRA (near + 4pt)
# cra_near = reanalysis_to_long(
#     cra_tq_near_filtered, time_col="BJT", p_col="Level (hPa)",
#     t_col="Temp (K)", q_col="q (g/kg)",
#     out_t="T_K", out_q="q_gkg"
# )
# cra_4pt = reanalysis_to_long(
#     cra_tq_4pt_filtered, time_col="BJT", p_col="Level (hPa)",
#     t_col="Temp (K)", q_col="q (g/kg)",
#     out_t="T_K", out_q="q_gkg"
# )

# # ERA5 (near + 4pt)
# era5_near = reanalysis_to_long(
#     era5_tq_near_filtered, time_col="BJT", p_col="pressure_level",
#     t_col="Temp (K)", q_col="q (g/kg)",
#     out_t="T_K", out_q="q_gkg"
# )
# era5_4pt = reanalysis_to_long(
#     era5_tq_4pt_filtered, time_col="BJT", p_col="pressure_level",
#     t_col="Temp (K)", q_col="q (g/kg)",
#     out_t="T_K", out_q="q_gkg"
# )

# # Radiosonde
# rs = radiosonde_to_long(radiosonde)

# # Dataset dictionaries (ALL, include MWR as a candidate)
# DATASETS_T_ALL = {}
# DATASETS_q_ALL = {}

# # FY4B
# for k, df in fy4B_T_bins.items():
#     DATASETS_T_ALL[k.replace("FY4B_T", "FY4B")] = df.rename(columns={"T_K": "T_K"})
# for k, df in fy4B_q_bins.items():
#     DATASETS_q_ALL[k.replace("FY4B_q", "FY4B")] = df.rename(columns={"q_gkg": "q_gkg"})

# # CRA/ERA5
# DATASETS_T_ALL["CRA_near"]  = cra_near["T"]
# DATASETS_T_ALL["CRA_4pt"]   = cra_4pt["T"]
# DATASETS_T_ALL["ERA5_near"] = era5_near["T"]
# DATASETS_T_ALL["ERA5_4pt"]  = era5_4pt["T"]

# DATASETS_q_ALL["CRA_near"]  = cra_near["q"]
# DATASETS_q_ALL["CRA_4pt"]   = cra_4pt["q"]
# DATASETS_q_ALL["ERA5_near"] = era5_near["q"]
# DATASETS_q_ALL["ERA5_4pt"]  = era5_4pt["q"]

# # Radiosonde
# DATASETS_T_ALL["Radiosonde"] = rs["T"]
# DATASETS_q_ALL["Radiosonde"] = rs["q"]

# # MWR (当 ref!=MWR 时，MWR 将作为对比数据集自动参与)
# DATASETS_T_ALL["MWR"] = mwr_T[["BJT_aligned", "pressure_hPa", "T_K"]].copy()
# DATASETS_q_ALL["MWR"] = mwr_q[["BJT_aligned", "pressure_hPa", "q_gkg"]].copy()

# # =========================
# # 2.1) SELECT REFERENCE (REF SWITCH)
# # =========================
# if REF_DATASET not in ["MWR", "CRA_near", "CRA_4pt"]:
#     raise ValueError("REF_DATASET must be one of: 'MWR' | 'CRA_near' | 'CRA_4pt'")

# REF_LABEL = REF_DATASET  # used in plots/labels

# # reference long tables
# if REF_DATASET == "MWR":
#     REF_T_RAW = mwr_T
#     REF_q_RAW = mwr_q
#     REF_T_COL = "T_K"
#     REF_q_COL = "q_gkg"
# else:
#     # CRA reference
#     if REF_DATASET == "CRA_near":
#         REF_T_RAW = cra_near["T"]
#         REF_q_RAW = cra_near["q"]
#     else:
#         REF_T_RAW = cra_4pt["T"]
#         REF_q_RAW = cra_4pt["q"]
#     REF_T_COL = "T_K"
#     REF_q_COL = "q_gkg"

# # 参考/对比都限制在原两个时段范围内（保持你原逻辑）
# refT_all = filter_union_periods(REF_T_RAW, periods, REF_T_COL)
# refq_all = filter_union_periods(REF_q_RAW, periods, REF_q_COL)

# # target datasets = ALL datasets - reference itself
# DATASETS_T = {k: v for k, v in DATASETS_T_ALL.items() if k != REF_DATASET}
# DATASETS_q = {k: v for k, v in DATASETS_q_ALL.items() if k != REF_DATASET}


# # =========================
# # 3) MAIN LOOP: RH/CBH/RAIN bins × variables × datasets
# # =========================
# paired_all = []
# metrics_all = []
# metrics_all_cbh = []
# metrics_all_rain = []

# # RH hourly
# rh_hourly = rh_to_aligned(rh_mean_8_320, value_col=RH_COL)
# rh_hourly = filter_union_periods(rh_hourly.rename(columns={"RH": "RHval"}), periods, "RHval") \
#               .rename(columns={"RHval": "RH"})

# # CBH hourly
# cbh_hourly = cbh_to_aligned(cbh_1min, value_col=CBH_COL, agg="median")
# cbh_hourly = filter_union_periods(cbh_hourly.rename(columns={"CBH": "CBHval"}), periods, "CBHval") \
#                .rename(columns={"CBHval": "CBH"})
# print(f"CBH hourly rows: {len(cbh_hourly)}")

# # RAIN hourly
# rain_hourly = rain_to_aligned(met_1min, value_col=RAIN_COL)
# rain_hourly = filter_union_periods(
#     rain_hourly.rename(columns={"RainRate_mmh": "RainRate_tmp"}),
#     periods,
#     "RainRate_tmp"
# ).rename(columns={"RainRate_tmp": "RainRate_mmh"})
# if "RainAmt_mm" in rain_hourly.columns:
#     rain_hourly["RainAmt_mm"] = pd.to_numeric(rain_hourly["RainAmt_mm"], errors="coerce")
# print(f"RAIN hourly rows: {len(rain_hourly)}")

# print(f"REF({REF_LABEL}) T rows: {len(refT_all)}, q rows: {len(refq_all)}")
# print(f"RH hourly rows: {len(rh_hourly)}")

# for var, ref_df, ref_col, dsets in [
#     ("T", refT_all, REF_T_COL, DATASETS_T),
#     ("q", refq_all, REF_q_COL, DATASETS_q),
# ]:
#     for dname, ddf0 in dsets.items():
#         tar_col = "T_K" if var == "T" else "q_gkg"

#         ddf = filter_union_periods(ddf0, periods, tar_col)

#         # 压力仍用 mwr_pressures（保持原功能/参数不变）
#         ddf = ddf[ddf["pressure_hPa"].isin(mwr_pressures)].copy()
#         ref_use = ref_df[ref_df["pressure_hPa"].isin(mwr_pressures)].copy()

#         pairs = build_pairs(
#             ref_use[["BJT_aligned", "pressure_hPa", ref_col]], ref_col,
#             ddf[["BJT_aligned", "pressure_hPa", tar_col]], tar_col,
#             rh_hourly=rh_hourly,
#             cbh_hourly=cbh_hourly,
#             rain_hourly=rain_hourly
#         )

#         if pairs.empty:
#             continue

#         tmp = pairs.copy()
#         tmp["variable"] = var
#         tmp["dataset"] = dname
#         paired_all.append(tmp[[
#             "variable", "dataset",
#             "rh_bin", "RH",
#             "cbh_bin", "CBH",
#             "rain_bin", "RainRate_mmh", "RainAmt_mm",
#             "layer", "BJT_aligned", "pressure_hPa", "ref", "mod", "bias"
#         ]])

#         # RH bins × layer
#         for rh_name, lo, hi in RH_BINS:
#             sub_rh = pairs[pairs["rh_bin"] == rh_name]
#             if sub_rh.empty:
#                 continue
#             for layer_name in LAYER_DEF.keys():
#                 sub = sub_rh[sub_rh["layer"] == layer_name]
#                 mm = compute_metrics(sub["ref"].values, sub["mod"].values)
#                 metrics_all.append({
#                     "rh_bin": rh_name,
#                     "variable": var,
#                     "dataset": dname,
#                     "layer": layer_name,
#                     **mm
#                 })

#         # CBH bins × layer
#         for cbh_name, lo, hi in CBH_BINS:
#             sub_cbh = pairs[pairs["cbh_bin"] == cbh_name]
#             if sub_cbh.empty:
#                 continue
#             for layer_name in LAYER_DEF.keys():
#                 sub = sub_cbh[sub_cbh["layer"] == layer_name]
#                 mm = compute_metrics(sub["ref"].values, sub["mod"].values)
#                 metrics_all_cbh.append({
#                     "cbh_bin": cbh_name,
#                     "variable": var,
#                     "dataset": dname,
#                     "layer": layer_name,
#                     **mm
#                 })

#         # RAIN bins × layer
#         for rain_name, lo, hi in RAIN_BINS:
#             sub_rain = pairs[pairs["rain_bin"] == rain_name]
#             if sub_rain.empty:
#                 continue
#             for layer_name in LAYER_DEF.keys():
#                 sub = sub_rain[sub_rain["layer"] == layer_name]
#                 mm = compute_metrics(sub["ref"].values, sub["mod"].values)
#                 metrics_all_rain.append({
#                     "rain_bin": rain_name,
#                     "variable": var,
#                     "dataset": dname,
#                     "layer": layer_name,
#                     **mm
#                 })

# paired_df = pd.concat(paired_all, ignore_index=True) if paired_all else pd.DataFrame()
# metrics_df = pd.DataFrame(metrics_all)
# metrics_cbh_df = pd.DataFrame(metrics_all_cbh)
# metrics_rain_df = pd.DataFrame(metrics_all_rain)

# print("[OK] RH/CBH/RAIN-binned pairing & metrics done.")
# # =========================================================
# # Add FY4B_all ONLY for RAIN-binned Temperature Taylor plots
# # (Keep original 3 rings in everything else)
# # =========================================================
# metrics_rain_df_taylor = metrics_rain_df.copy()

# if (not paired_df.empty):
#     fy = paired_df[(paired_df["variable"] == "T") &
#                    (paired_df["dataset"].astype(str).str.startswith("FY4B"))].copy()

#     rows_add = []
#     if not fy.empty:
#         # 仅对你定义的雨分箱做 FY4B_all
#         for rain_name, _, _ in RAIN_BINS:
#             for layer_name in LAYER_DEF.keys():
#                 sub = fy[(fy["rain_bin"] == rain_name) &
#                          (fy["layer"] == layer_name)]
#                 if sub.empty:
#                     continue

#                 mm = compute_metrics(sub["ref"].values, sub["mod"].values)  # 用你现有 MIN_PAIRS
#                 rows_add.append({
#                     "rain_bin": rain_name,
#                     "variable": "T",
#                     "dataset": "FY4B_all",
#                     "layer": layer_name,
#                     **mm
#                 })

#     if rows_add:
#         metrics_rain_df_taylor = pd.concat(
#             [metrics_rain_df_taylor, pd.DataFrame(rows_add)],
#             ignore_index=True
#         )


# # =========================
# # 5) TAYLOR DIAGRAMS
# # =========================
# def plot_taylor_for(rh_bin_name, var, layer_name):
#     subm = metrics_df[(metrics_df["rh_bin"] == rh_bin_name) &
#                       (metrics_df["variable"] == var) &
#                       (metrics_df["layer"] == layer_name)].copy()
#     subm = subm.dropna(subset=["std_ref", "std_mod", "r"])
#     if subm.empty:
#         return

#     subm["std_ratio"] = subm["std_mod"] / subm["std_ref"].replace(0, np.nan)

#     fig = plt.figure(figsize=(8.5, 8.5))
#     td = TaylorDiagram(ref_std=1.0, fig=fig, rect=111, label=REF_LABEL, srange=(0, 1.8))

#     for _, row in subm.iterrows():
#         label = f"{row['dataset']}, T={row['T']:.3f}"
#         dataset_name = str(row['dataset'])

#         marker_map = {
#             'fy4B': '^',
#             'radiosonde': '*',
#             'era5': '.',
#             'cra': '.',
#             'mwr': 's',
#             'sonde': '*'
#         }
#         marker = "o"
#         for key, mark in marker_map.items():
#             if key in dataset_name:
#                 marker = mark
#                 break

#         td.add_sample(row["std_ratio"], row["r"], label=label, marker=marker)

#     title_var = "Temperature (K)" if var == "T" else "Specific humidity (g/kg)"
#     plt.title(f"{rh_bin_name} | {layer_name} hPa {title_var} (Ref={REF_LABEL})",
#               fontweight="bold", pad=22)
#     plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0.72), frameon=False)

#     out_png = os.path.join(OUTPUT_DIR, f"Taylor_{rh_bin_name}_{var}_{layer_name}.png")
#     plt.savefig(out_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)

# for rh_name, _, _ in RH_BINS:
#     for var in ["T", "q"]:
#         for layer in LAYER_DEF.keys():
#             plot_taylor_for(rh_name, var, layer)
# print("[OK] RH-binned Taylor diagrams saved.")


# def plot_taylor_for_cbh(cbh_bin_name, var, layer_name):
#     subm = metrics_cbh_df[(metrics_cbh_df["cbh_bin"] == cbh_bin_name) &
#                           (metrics_cbh_df["variable"] == var) &
#                           (metrics_cbh_df["layer"] == layer_name)].copy()
#     subm = subm.dropna(subset=["std_ref", "std_mod", "r"])
#     if subm.empty:
#         return

#     subm["std_ratio"] = subm["std_mod"] / subm["std_ref"].replace(0, np.nan)

#     fig = plt.figure(figsize=(8.5, 8.5))
#     td = TaylorDiagram(ref_std=1.0, fig=fig, rect=111, label=REF_LABEL, srange=(0, 1.8))

#     for _, row in subm.iterrows():
#         label = f"{row['dataset']}, T={row['T']:.3f}"
#         dataset_name = str(row['dataset'])

#         marker_map = {
#             'fy4B': '^',
#             'radiosonde': '*',
#             'era5': '.',
#             'cra': '.',
#             'mwr': 's',
#             'sonde': '*'
#         }
#         marker = "o"
#         for key, mark in marker_map.items():
#             if key in dataset_name:
#                 marker = mark
#                 break

#         td.add_sample(row["std_ratio"], row["r"], label=label, marker=marker)

#     title_var = "Temperature (K)" if var == "T" else "Specific humidity (g/kg)"
#     plt.title(f"{cbh_bin_name} | {layer_name} hPa {title_var} (Ref={REF_LABEL})",
#               fontweight="bold", pad=22)
#     plt.legend(loc="lower right", bbox_to_anchor=(1.15, 0.72), frameon=False)

#     out_png = os.path.join(OUTPUT_DIR, f"Taylor_CBH_{cbh_bin_name}_{var}_{layer_name}.png")
#     plt.savefig(out_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)

# for cbh_name, _, _ in CBH_BINS:
#     for var in ["T", "q"]:
#         for layer in LAYER_DEF.keys():
#             plot_taylor_for_cbh(cbh_name, var, layer)
# print("[OK] CBH-binned Taylor diagrams saved.")


# def plot_taylor_for_rain(rain_bin_name, var, layer_name):
#     subm = metrics_rain_df_taylor[(metrics_rain_df_taylor["rain_bin"] == rain_bin_name) &
#                                   (metrics_rain_df_taylor["variable"] == var) &
#                                   (metrics_rain_df_taylor["layer"] == layer_name)].copy()

#     subm = subm.dropna(subset=["std_ref", "std_mod", "r"])
#     if subm.empty:
#         return

#     subm["std_ratio"] = subm["std_mod"] / subm["std_ref"].replace(0, np.nan)
#     subm = sort_by_fixed_order(subm, col="dataset")
#     fig = plt.figure(figsize=(8.5, 8.5))
#     td = TaylorDiagram(ref_std=1.0, fig=fig, rect=111, label=REF_LABEL, srange=(0, 1.8))
#     palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

#     # 你希望固定的顺序（按你想在不同图里保持一致的优先级排列）
#     fixed_order = [
#         "FY4B_all", "FY4B_<=6km", "FY4B_20-25km", "FY4B_30-35km",
#         "Radiosonde", "ERA5_near", "ERA5_4pt", "CRA_near", "CRA_4pt", "MWR"
#     ]
#     color_map = {ds: palette[i % len(palette)] for i, ds in enumerate(fixed_order)}

#     # for _, row in subm.iterrows():
#     #     label = f"{row['dataset']}"#, T={row['T']:.3f}
#     #     dataset_name = str(row["dataset"]).lower()

#     #     marker_map = {
#     #         'fy4B': '^',
#     #         'radiosonde': '*',
#     #         'era5': '.',
#     #         'cra': '.',
#     #         'mwr': 's',
#     #         'sonde': '*'
#     #     }
#     #     marker = "o"
#     #     for key, mark in marker_map.items():
#     #         if key in dataset_name:
#     #             marker = mark
#     #             break

#     #     td.add_sample(row["std_ratio"], row["r"], label=label, marker=marker)
#     for _, row in subm.iterrows():
#         ds = str(row["dataset"])
#         marker = TAYLOR_MARKER_MAP.get(ds, "o")
#         color  = TAYLOR_COLOR_MAP.get(ds, None)
    
#         td.add_sample(row["std_ratio"], row["r"], label=ds, marker=marker, color=color)
#     title_var = "Temperature (K)" if var == "T" else "Specific humidity (g/kg)"
#     unit_tag = "mm/h" if RAIN_BIN_MODE == "rate" else "mm"
#     plt.title(f"{rain_bin_name} ({unit_tag}) | {layer_name} hPa {title_var} (Ref={REF_LABEL})",
#               fontweight="bold", pad=22)
#     plt.legend(loc="lower right", bbox_to_anchor=(0.55, 1.52), frameon=False)

#     out_png = os.path.join(OUTPUT_DIR, f"Taylor_RAIN_{rain_bin_name}_{var}_{layer_name}.png")
#     plt.savefig(out_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)

# for rain_name, _, _ in RAIN_BINS:
#     for var in ["T", "q"]:
#         for layer in LAYER_DEF.keys():
#             plot_taylor_for_rain(rain_name, var, layer)
# print("[OK] RAIN-binned Taylor diagrams saved.")


# # =========================
# # 6) BIAS PROFILE UNCERTAINTY PLOT (bootstrap 95% CI)
# # =========================
# def plot_bias_uncertainty(rh_bin_name, var):
#     x = paired_df[(paired_df["rh_bin"] == rh_bin_name) &
#                   (paired_df["variable"] == var)].copy()
#     if x.empty:
#         return

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111)
#     linestyle_map = {
#         "FY4B_<=6km": (0, (5, 1)),
#         "FY4B_20-25km": (0, (5, 1)),
#         "FY4B_30-35km": (0, (5, 1)),
#         "CRA_near": "dotted",
#         "CRA_4pt": "dotted",
#         "ERA5_near": "dotted",
#         "ERA5_4pt": "dotted",
#         "Radiosonde": "solid",
#         "MWR": "dashdot",
#     }

#     for dname in sorted(x["dataset"].unique()):
#         xd = x[x["dataset"] == dname]
#         means, los, his, ps = [], [], [], []

#         for p in sorted(mwr_pressures, reverse=True):
#             bp = xd[xd["pressure_hPa"] == p]["bias"].values
#             m, lo, hi = bootstrap_mean_ci(bp)
#             if np.isfinite(m):
#                 ps.append(p); means.append(m); los.append(lo); his.append(hi)

#         if len(ps) < 3:
#             continue

#         ps = np.array(ps)
#         means = np.array(means)
#         los = np.array(los)
#         his = np.array(his)

#         linewidth = 2.5 if dname == "Radiosonde" else 1.5
#         ax.plot(means, ps, linewidth=linewidth, label=dname,
#                 linestyle=linestyle_map.get(dname, "-"))
#         ax.fill_betweenx(ps, los, his, alpha=0.15)

#     ax.axvline(0, linewidth=2, color="black", linestyle=(0, (5, 10)))
#     ax.invert_yaxis()
#     ax.set_ylabel("Pressure (hPa)", fontweight="bold")

#     if var == "T":
#         ax.set_xlabel(f"Bias = (dataset - {REF_LABEL}) (K)", fontweight="bold")
#         title_var = "Temperature"
#     else:
#         ax.set_xlabel(f"Bias = (dataset - {REF_LABEL}) (g/kg)", fontweight="bold")
#         title_var = "Specific humidity"

#     ax.set_title(f"{rh_bin_name} {title_var} Bias Profile (Ref={REF_LABEL})",
#                  fontweight="bold", pad=14)
#     ax.legend(loc="upper right", frameon=False)
#     for spine in ax.spines.values():
#         spine.set_linewidth(2)
#     ax.tick_params(axis='both', which='major', width=2, labelsize=14)

#     out_png = os.path.join(OUTPUT_DIR, f"BiasUncertainty_{rh_bin_name}_{var}.png")
#     plt.savefig(out_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)

# for rh_name, _, _ in RH_BINS:
#     plot_bias_uncertainty(rh_name, "T")
#     plot_bias_uncertainty(rh_name, "q")
# print("[OK] RH-binned bias uncertainty plots saved.")


# def plot_bias_uncertainty_cbh(cbh_bin_name, var):
#     x = paired_df[(paired_df["cbh_bin"] == cbh_bin_name) &
#                   (paired_df["variable"] == var)].copy()
#     if x.empty:
#         return

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111)

#     linestyle_map = {
#         "FY4B_<=6km": (0, (5, 1)),
#         "FY4B_20-25km": (0, (5, 1)),
#         "FY4B_30-35km": (0, (5, 1)),
#         "CRA_near": "dotted",
#         "CRA_4pt": "dotted",
#         "ERA5_near": "dotted",
#         "ERA5_4pt": "dotted",
#         "Radiosonde": "solid",
#         "MWR": "dashdot",
#     }

#     for dname in sorted(x["dataset"].unique()):
#         xd = x[x["dataset"] == dname]
#         means, los, his, ps = [], [], [], []

#         for p in sorted(mwr_pressures, reverse=True):
#             bp = xd[xd["pressure_hPa"] == p]["bias"].values
#             m, lo, hi = bootstrap_mean_ci(bp)
#             if np.isfinite(m):
#                 ps.append(p); means.append(m); los.append(lo); his.append(hi)

#         if len(ps) < 3:
#             continue

#         ps = np.array(ps)
#         means = np.array(means)
#         los = np.array(los)
#         his = np.array(his)

#         linewidth = 2.5 if dname == "Radiosonde" else 1.5
#         ax.plot(means, ps, linewidth=linewidth, label=dname,
#                 linestyle=linestyle_map.get(dname, "-"))
#         ax.fill_betweenx(ps, los, his, alpha=0.15)

#     ax.axvline(0, linewidth=2, color="black", linestyle=(0, (5, 10)))
#     ax.invert_yaxis()
#     ax.set_ylabel("Pressure (hPa)", fontweight="bold")

#     if var == "T":
#         ax.set_xlabel(f"Bias = (dataset - {REF_LABEL}) (K)", fontweight="bold")
#         title_var = "Temperature"
#     else:
#         ax.set_xlabel(f"Bias = (dataset - {REF_LABEL}) (g/kg)", fontweight="bold")
#         title_var = "Specific humidity"

#     ax.set_title(f"{cbh_bin_name} {title_var} Bias Profile (Ref={REF_LABEL})",
#                  fontweight="bold", pad=14)
#     ax.legend(loc="upper right", frameon=False)

#     for spine in ax.spines.values():
#         spine.set_linewidth(2)
#     ax.tick_params(axis='both', which='major', width=2, labelsize=14)

#     out_png = os.path.join(OUTPUT_DIR, f"BiasUncertainty_CBH_{cbh_bin_name}_{var}.png")
#     plt.savefig(out_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)

# for cbh_name, _, _ in CBH_BINS:
#     plot_bias_uncertainty_cbh(cbh_name, "T")
#     plot_bias_uncertainty_cbh(cbh_name, "q")
# print("[OK] CBH-binned bias uncertainty plots saved.")


# def plot_bias_uncertainty_rain(rain_bin_name, var):
#     x = paired_df[(paired_df["rain_bin"] == rain_bin_name) &
#                   (paired_df["variable"] == var)].copy()
#     if x.empty:
#         return
# # --- 合并 FY4B 三个环带为一个整体（仅用于雨分箱偏差图） ---
#     is_fy4B = x["dataset"].astype(str).str.startswith("FY4B")
#     x.loc[is_fy4B, "dataset"] = "FY4B_all"
#     # if (var == "T") and (str(rain_bin_name) != "RAIN_0"):
#     #     keep_fy4B = {"FY4B_<=6km", "FY4B_20-25km"}
#     #     is_fy4B = x["dataset"].astype(str).str.startswith("FY4B")
#     #     x = x[~is_fy4B | x["dataset"].isin(keep_fy4B)].copy()
#     fig = plt.figure(figsize=(7, 9))
#     ax = fig.add_subplot(111)

#     linestyle_map = {
#         "FY4B_<=6km": (0, (5, 1)),
#         "FY4B_20-25km": (0, (5, 1)),
#         "FY4B_30-35km": (0, (5, 1)),
#         "FY4B_all": (0, (5, 1)),
#         "CRA_near": "dotted",
#         "CRA_4pt": "dotted",
#         "ERA5_near": "dotted",
#         "ERA5_4pt": "dotted",
#         "Radiosonde": "solid",
#         "MWR": "dashdot",
#     }

#     for dname in sorted(x["dataset"].unique()):
#         xd = x[x["dataset"] == dname]
#         means, los, his, ps = [], [], [], []

#         for p in sorted(mwr_pressures, reverse=True):
#             bp = xd[xd["pressure_hPa"] == p]["bias"].values
#             m, lo, hi = bootstrap_mean_ci(bp)
#             if np.isfinite(m):
#                 ps.append(p); means.append(m); los.append(lo); his.append(hi)

#         if len(ps) < 3:
#             continue

#         ps = np.array(ps)
#         means = np.array(means)
#         los = np.array(los)
#         his = np.array(his)

#         linewidth = 2.5 if dname == "Radiosonde" else 1.5
#         ax.plot(means, ps, linewidth=linewidth, label=dname,
#                 linestyle=linestyle_map.get(dname, "-"))
#         ax.fill_betweenx(ps, los, his, alpha=0.15)

#     ax.axvline(0, linewidth=2, color="black", linestyle=(0, (5, 10)))
#     ax.invert_yaxis()
#     ax.set_ylabel("Pressure (hPa)", fontweight="bold")

#     if var == "T":
#         ax.set_xlabel(f"Bias = (dataset - {REF_LABEL}) (K)", fontweight="bold")
#         title_var = "Temperature"
#     else:
#         ax.set_xlabel(f"Bias = (dataset - {REF_LABEL}) (g/kg)", fontweight="bold")
#         title_var = "Specific humidity"

#     unit_tag = "mm/h" if RAIN_BIN_MODE == "rate" else "mm"
#     ax.set_title(f"{rain_bin_name} ({unit_tag}) {title_var} Bias Profile (Ref={REF_LABEL})",
#                  fontweight="bold", pad=14)
#     ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(0, -0.8))

#     for spine in ax.spines.values():
#         spine.set_linewidth(2)
#     ax.tick_params(axis='both', which='major', width=2, labelsize=18)

#     out_png = os.path.join(OUTPUT_DIR, f"BiasUncertainty_RAIN_{rain_bin_name}_{var}.png")
#     plt.savefig(out_png, dpi=300, bbox_inches="tight")
#     plt.close(fig)

# for rain_name, _, _ in RAIN_BINS:
#     plot_bias_uncertainty_rain(rain_name, "T")
    
#     plot_bias_uncertainty_rain(rain_name, "q")
# print("[OK] RAIN-binned bias uncertainty plots saved.")








# # =========================
# # 7) EXPORT PAIRED DATA: ONE EXCEL PER (BIN × LAYER), with T & q in SAME workbook
# # =========================
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# group_map = {
#     'FY4B_<=6km': 'Satellite',
#     'FY4B_20-25km': 'Satellite',
#     'FY4B_30-35km': 'Satellite',
#     'Radiosonde': 'Radiosonde',
#     'CRA_near': 'Reanalysis',
#     'CRA_4pt': 'Reanalysis',
#     'ERA5_near': 'Reanalysis',
#     'ERA5_4pt': 'Reanalysis',
#     'MWR': 'MWR',   # 新增：当 ref!=MWR 时，MWR 会作为 dataset 出现
# }

# paired_df = paired_df.copy()
# paired_df["group"] = paired_df["dataset"].map(group_map).fillna("Other")

# export_cols = [
#     "dataset", "group", "layer",
#     "BJT_aligned", "pressure_hPa",
#     "RH", "rh_bin",
#     "CBH", "cbh_bin",
#     "ref", "mod", "bias",
#     "RainRate_mmh", "RainAmt_mm", "rain_bin",
# ]
# export_cols = [c for c in export_cols if c in paired_df.columns]

# EXCEL_MAX_ROWS = 1048576

# def _sanitize_filename(s: str) -> str:
#     s = str(s)
#     s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
#     s = s.replace(" ", "")
#     return s[:180]

# def _safe_sheet(s: str) -> str:
#     s = str(s)
#     s = re.sub(r"[\[\]\*:/\\\?]+", "_", s)
#     return s[:31]

# def _write_df_chunked(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_base: str):
#     if df.empty:
#         return
#     n = len(df)
#     if n <= EXCEL_MAX_ROWS:
#         df.to_excel(writer, sheet_name=_safe_sheet(sheet_base), index=False)
#         return
#     n_parts = int(np.ceil(n / EXCEL_MAX_ROWS))
#     for i in range(n_parts):
#         a = i * EXCEL_MAX_ROWS
#         b = min((i + 1) * EXCEL_MAX_ROWS, n)
#         part = df.iloc[a:b].copy()
#         sheet_name = f"{sheet_base}_{i+1}"
#         part.to_excel(writer, sheet_name=_safe_sheet(sheet_name), index=False)

# def _export_one_book(df_sub: pd.DataFrame,
#                      metrics_sub: pd.DataFrame,
#                      out_path: str,
#                      title_note: str = ""):
#     if df_sub.empty:
#         return

#     sort_cols = [c for c in ["variable", "group", "dataset", "BJT_aligned", "pressure_hPa"] if c in df_sub.columns]
#     if sort_cols:
#         df_sub = df_sub.sort_values(sort_cols)

#     group_order = ["Satellite", "Radiosonde", "Reanalysis", "MWR", "Other"]
#     groups = [g for g in group_order if g in df_sub["group"].unique()] + \
#              [g for g in sorted(df_sub["group"].unique()) if g not in group_order]

#     with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#         info = pd.DataFrame({
#             "key": ["note", "reference_dataset"],
#             "value": [title_note, REF_LABEL]
#         })
#         info.to_excel(writer, sheet_name="Info", index=False)

#         for grp in groups:
#             for var in ["T", "q"]:
#                 d = df_sub[(df_sub["group"] == grp) & (df_sub["variable"] == var)].copy()
#                 if d.empty:
#                     continue
#                 cols = export_cols.copy()
#                 if "variable" in df_sub.columns and "variable" not in cols:
#                     cols = ["variable"] + cols
#                 cols = [c for c in cols if c in d.columns]
#                 d = d[cols]
#                 sheet = f"{grp[:3]}_{var}"
#                 _write_df_chunked(writer, d, sheet)

#         if metrics_sub is not None and (not metrics_sub.empty):
#             _write_df_chunked(writer, metrics_sub, "Metrics")


# # A) RH-binned export
# rh_out_dir = os.path.join(OUTPUT_DIR, "Export_RH_bins")
# os.makedirs(rh_out_dir, exist_ok=True)

# for rh_name, _, _ in RH_BINS:
#     for layer_name in LAYER_DEF.keys():
#         df_sub = paired_df[(paired_df.get("rh_bin") == rh_name) &
#                            (paired_df.get("layer") == layer_name)].copy()
#         if df_sub.empty:
#             continue

#         metrics_sub = metrics_df[(metrics_df["rh_bin"] == rh_name) &
#                                  (metrics_df["layer"] == layer_name)].copy() if (not metrics_df.empty) else pd.DataFrame()

#         out_fn = _sanitize_filename(f"Paired_RH_{rh_name}_{layer_name}.xlsx")
#         out_path = os.path.join(rh_out_dir, out_fn)
#         note = f"RH bin={rh_name}; layer={layer_name}. T & q in this workbook. Ref={REF_LABEL}."
#         _export_one_book(df_sub, metrics_sub, out_path, title_note=note)

# print(f"[OK] RH-binned export done: {rh_out_dir}")

# # B) CBH-binned export
# cbh_out_dir = os.path.join(OUTPUT_DIR, "Export_CBH_bins")
# os.makedirs(cbh_out_dir, exist_ok=True)

# for cbh_name, _, _ in CBH_BINS:
#     for layer_name in LAYER_DEF.keys():
#         df_sub = paired_df[(paired_df.get("cbh_bin") == cbh_name) &
#                            (paired_df.get("layer") == layer_name)].copy()
#         if df_sub.empty:
#             continue

#         metrics_sub = metrics_cbh_df[(metrics_cbh_df["cbh_bin"] == cbh_name) &
#                                      (metrics_cbh_df["layer"] == layer_name)].copy() if (not metrics_cbh_df.empty) else pd.DataFrame()

#         out_fn = _sanitize_filename(f"Paired_CBH_{cbh_name}_{layer_name}.xlsx")
#         out_path = os.path.join(cbh_out_dir, out_fn)
#         note = f"CBH bin={cbh_name}; layer={layer_name}. T & q in this workbook. Ref={REF_LABEL}."
#         _export_one_book(df_sub, metrics_sub, out_path, title_note=note)

# print(f"[OK] CBH-binned export done: {cbh_out_dir}")

# # C) RAIN-binned export
# rain_out_dir = os.path.join(OUTPUT_DIR, "Export_RAIN_bins")
# os.makedirs(rain_out_dir, exist_ok=True)

# for rain_name, _, _ in RAIN_BINS:
#     for layer_name in LAYER_DEF.keys():
#         df_sub = paired_df[(paired_df.get("rain_bin") == rain_name) &
#                            (paired_df.get("layer") == layer_name)].copy()
#         if df_sub.empty:
#             continue

#         metrics_sub = metrics_rain_df[(metrics_rain_df["rain_bin"] == rain_name) &
#                                       (metrics_rain_df["layer"] == layer_name)].copy() if (not metrics_rain_df.empty) else pd.DataFrame()

#         out_fn = _sanitize_filename(f"Paired_RAIN_{rain_name}_{layer_name}.xlsx")
#         out_path = os.path.join(rain_out_dir, out_fn)
#         note = f"RAIN bin={rain_name}; layer={layer_name}. RAIN_BIN_MODE={RAIN_BIN_MODE}. Ref={REF_LABEL}."
#         _export_one_book(df_sub, metrics_sub, out_path, title_note=note)

# print(f"[OK] RAIN-binned export done: {rain_out_dir}")
# print(f"[DONE] Reference dataset used: {REF_LABEL}")







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator

# =========================================================
# 0) OUTPUT
# =========================================================
OUT_DIR = Path(r"E:\Beijing2024") / "raw data profile in rain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 0.5) PRESSURE FILTER (hPa): keep 300–850 only
# =========================================================
P_MIN, P_MAX = 300.0, 850.0

# =========================================================
# 1) PRECIP EVENTS (3 events)  —— minute & hour windows
# =========================================================
EVENTS = [
    dict(
        name="Event1_20240809_1246_20240810_0003",
        minute=(pd.Timestamp("2024-08-09 12:46"), pd.Timestamp("2024-08-10 00:03")),
        hour=(pd.Timestamp("2024-08-09 12:00"), pd.Timestamp("2024-08-10 01:00")),
    ),
    dict(
        name="Event2_20240607_0916_1500",
        minute=(pd.Timestamp("2024-06-07 09:16"), pd.Timestamp("2024-06-07 15:00")),
        hour=(pd.Timestamp("2024-06-07 09:00"), pd.Timestamp("2024-06-07 15:00")),
    ),
    dict(
        name="Event3_20240810_1620_1730",
        minute=(pd.Timestamp("2024-08-10 16:20"), pd.Timestamp("2024-08-10 17:30")),
        hour=(pd.Timestamp("2024-08-10 16:00"), pd.Timestamp("2024-08-10 18:00")),
    ),
]

# =========================================================
# 2) DATAFRAME HANDLES
# =========================================================
# MWR temperature wide table: prefer temp_p_df if exists else temp
try:
    mwr_temp_wide = temp_p_df
except NameError:
    mwr_temp_wide = temp  # per your description

# FY4B temperature long table
fy4B_T_df = fy4B_T

# Reanalysis long tables
era5_4pt = era5_tq_4pt_filtered
era5_near = era5_tq_near_filtered
cra_4pt = cra_tq_4pt_filtered
cra_near = cra_tq_near_filtered

# Radiosonde long table
rad_df = radiosonde

# =========================================================
# 3) HELPERS
# =========================================================
def _to_naive_dt(s):
    t = pd.to_datetime(s)
    if hasattr(t, "dt"):
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert(None)
    else:
        if getattr(t, "tz", None) is not None:
            t = t.tz_convert(None)
    return t

def build_time_mask(times, start, end):
    t = _to_naive_dt(times)
    st = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    return (t >= st) & (t <= ed)

def get_numeric_pressure_cols(df, exclude_cols=None, pmin=P_MIN, pmax=P_MAX):
    """
    From a wide table, find pressure columns (numeric names), and filter to pmin–pmax.
    Return (pressure_values_desc, pressure_colnames_desc).
    """
    exclude_cols = set(exclude_cols or [])
    cols = [c for c in df.columns if c not in exclude_cols]

    pres = []
    keep = []
    for c in cols:
        try:
            p = float(str(c).strip())
            if (p >= pmin) and (p <= pmax):
                pres.append(p)
                keep.append(c)
        except Exception:
            continue

    if len(keep) == 0:
        raise ValueError(f"未找到 300–850 hPa 范围内的 pressure 列（宽表列名应为压力数值）。")

    order = np.argsort(pres)[::-1]  # descending pressure
    pres_sorted = np.array([pres[i] for i in order], dtype=float)
    cols_sorted = [keep[i] for i in order]
    return pres_sorted, cols_sorted

def ensure_time_index_for_wide(df, time_col_candidates=("BJT", "BJT_aligned", "time", "datetime")):
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = _to_naive_dt(out.index)
        return out.sort_index()
    time_col = None
    for c in time_col_candidates:
        if c in out.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("宽表没有DatetimeIndex，也找不到时间列（BJT/BJT_aligned/time/datetime）。")
    out[time_col] = _to_naive_dt(out[time_col])
    out = out.set_index(time_col).sort_index()
    return out

def envelope_from_wide(df_wide, start, end, pmin=P_MIN, pmax=P_MAX):
    """
    Wide: index=time, columns=pressure(hPa numeric string/int/float).
    Filter time by event and pressure by 300–850 hPa, then compute min/max/mean across time.
    """
    df = ensure_time_index_for_wide(df_wide)
    mask = build_time_mask(df.index, start, end)
    sub = df.loc[mask]
    if sub.empty:
        return None

    pres_vals, pres_cols = get_numeric_pressure_cols(sub, pmin=pmin, pmax=pmax)

    block = sub[pres_cols].astype(float)

    lo = block.min(axis=0, skipna=True).values.astype(float)
    hi = block.max(axis=0, skipna=True).values.astype(float)
    mean = block.mean(axis=0, skipna=True).values.astype(float)

    y = pres_vals
    return y, lo, hi, mean

def envelope_from_long(df_long, start, end, time_col, p_col, v_col, extra_mask=None, pmin=P_MIN, pmax=P_MAX):
    """
    Long: rows (time, pressure, value). Filter time by event and pressure by 300–850 hPa.
    If multiple points at same time/pressure, first average them, then compute envelope across time.
    """
    df = df_long.copy()
    df[time_col] = _to_naive_dt(df[time_col])

    mask = build_time_mask(df[time_col], start, end)

    # pressure filter (300–850)
    mask = mask & (df[p_col].astype(float) >= pmin) & (df[p_col].astype(float) <= pmax)

    if extra_mask is not None:
        mask = mask & extra_mask(df)

    sub = df.loc[mask, [time_col, p_col, v_col]].copy()
    if sub.empty:
        return None

    gp = sub.groupby([time_col, p_col])[v_col].mean().reset_index()
    piv = gp.pivot(index=p_col, columns=time_col, values=v_col)

    # keep index within range (double safety) and sort
    piv = piv.loc[(piv.index.astype(float) >= pmin) & (piv.index.astype(float) <= pmax)]
    piv = piv.sort_index(ascending=False)

    if piv.empty:
        return None

    y = piv.index.values.astype(float)
    lo = piv.min(axis=1, skipna=True).values.astype(float)
    hi = piv.max(axis=1, skipna=True).values.astype(float)
    mean = piv.mean(axis=1, skipna=True).values.astype(float)

    return y, lo, hi, mean

def plot_band_and_mean(ax, y, lo, hi, mean, color, label,
                       alpha_fill=0.25, lw_mean=2.2,
                       hatch=None, hatch_lw=0.8, hatch_alpha=0.9,
                       z_fill=1, z_line=10):
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    mean = np.asarray(mean, dtype=float)

    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(mean)
    if ok.sum() < 2:
        return

    # (1) 底层：保持原来的填充颜色不变
    ax.fill_betweenx(
        y[ok], lo[ok], hi[ok],
        color=color, alpha=alpha_fill, linewidth=0,
        zorder=z_fill
    )

    # (2) 叠加层：只加纹理，不改变底色
    if hatch is not None:
        poly_h = ax.fill_betweenx(
            y[ok], lo[ok], hi[ok],
            facecolor="none",          # 关键：不盖住底色
            edgecolor=color,           # 纹理颜色跟随数据颜色（也可改成 "k" 更醒目）
            linewidth=hatch_lw,
            alpha=hatch_alpha,
            zorder=z_fill + 0.2
        )
        poly_h.set_hatch(hatch)

    # 均值线：最上层
    ax.plot(mean[ok], y[ok], color=color, linestyle="-", lw=lw_mean, label=label, zorder=z_line)

def style_axes(ax):

    ax.tick_params(axis="both", labelsize=22,width=2, length=4)
    ax.grid(True, alpha=0.25)
    ax.invert_yaxis()
    # range（你前面要求的 x=240–300；y=300–850 且反向）
    ax.set_xlim(230, 300)
    ax.set_ylim(850, 300)
    
    # tick spacing
    ax.xaxis.set_major_locator(MultipleLocator(10))     # x every 5 K
    ax.yaxis.set_major_locator(MultipleLocator(50))    # y every 25 hPa

   # grid follows major ticks
    ax.grid(True, which="major", alpha=0.25)
    # spine linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(2)

# =========================================================
# 4) FIXED STYLES (colors unchanged from your earlier spec)
# =========================================================
MWR_COLOR = "red"

FY_STYLE = {
    "≤6 km": "darkgreen",
    "20–25 km": "lime",
    "30–35 km": "chartreuse",
}
def fy_cat_mask(df, cat):
    d = df["distance_km"].astype(float)
    if cat == "≤6 km":
        return d <= 6
    if cat == "20–25 km":
        return (d >= 20) & (d <= 25)
    if cat == "30–35 km":
        return (d >= 30) & (d <= 35)
    return pd.Series(False, index=df.index)

REAN_STYLE = {
    "ERA5_4pt":  ("dodgerblue", era5_4pt,  "pressure_level"),
    "ERA5_near": ("blue",         era5_near, "pressure_level"),
    "CRA_4pt":   ("magenta",  cra_4pt,   "Level (hPa)"),
    "CRA_near":  ("violet",    cra_near,  "Level (hPa)"),
}

RAD_COLOR = "gold"


# =========================================================
# 5) MAIN: one figure per event (all datasets overlaid), Temperature only
# =========================================================
for ev in EVENTS:
    ev_name = ev["name"]
    m_st, m_ed = ev["minute"]
    h_st, h_ed = ev["hour"]

    fig, ax = plt.subplots(figsize=(7.8, 9.4))

    # ---------- MWR (1-min wide) ----------
    try:
        env = envelope_from_wide(mwr_temp_wide, m_st, m_ed)
        if env is not None:
            y, lo, hi, mean = env

            plot_band_and_mean(ax, y, lo, hi, mean, MWR_COLOR, "MWR",
                   alpha_fill=0.25, hatch="/", z_fill=1, z_line=20)
        else:
            print(f"[WARN] {ev_name}: MWR event window has no data.")
    except Exception as e:
        print(f"[WARN] {ev_name}: MWR failed -> {e}")

    # ---------- FY4B (2-hour long): three distance categories ----------
    try:
        for cat, col in FY_STYLE.items():
            env = envelope_from_long(
                fy4B_T_df,
                h_st, h_ed,
                time_col="BJT",
                p_col="pressure_hPa",
                v_col="T_K",
                extra_mask=lambda df, c=cat: fy_cat_mask(df, c),
            )
            if env is None:
                print(f"[WARN] {ev_name}: FY4B {cat} has no data in 300–850 hPa or event window.")
                continue
            y, lo, hi, mean = env
            plot_band_and_mean(ax, y, lo, hi, mean, col, f"FY4B {cat}")
    except Exception as e:
        print(f"[WARN] {ev_name}: FY4B failed -> {e}")

    # ---------- Reanalysis (hourly long): ERA5/CRA ----------
    try:
        for label, (col, df0, pcol) in REAN_STYLE.items():
            env = envelope_from_long(
                df0,
                h_st, h_ed,
                time_col="BJT",
                p_col=pcol,
                v_col="Temp (K)",
            )
            if env is None:
                print(f"[WARN] {ev_name}: {label} has no data in 300–850 hPa or event window.")
                continue
            y, lo, hi, mean = env
            plot_band_and_mean(ax, y, lo, hi, mean, col, f"{label}")
    except Exception as e:
        print(f"[WARN] {ev_name}: Reanalysis failed -> {e}")

    # ---------- Radiosonde (long): temperature_C -> K ----------
    try:
        rad = rad_df.copy()
        rad["time"] = _to_naive_dt(rad["time"])
        rad["T_K"] = rad["temperature_C"].astype(float) + 273.15

        env = envelope_from_long(
            rad,
            m_st, m_ed,
            time_col="time",
            p_col="pressure_hPa",
            v_col="T_K",
        )
        if env is None:
            print(f"[WARN] {ev_name}: Radiosonde has no data in 300–850 hPa or event window.")
        else:
            y, lo, hi, mean = env

            plot_band_and_mean(ax, y, lo, hi, mean, RAD_COLOR, "Radiosonde",
                   alpha_fill=0.25, hatch="-", z_fill=2, z_line=25)
    except Exception as e:
        print(f"[WARN] {ev_name}: Radiosonde failed -> {e}")

    # ---------- cosmetics ----------
    style_axes(ax)
    ax.set_title(
        #f"Temperature Profile Range (filled, α=0.250) & Mean (solid)\n"
        f"{m_st:%Y-%m-%d %H:%M} to {m_ed:%Y-%m-%d %H:%M}  |  300–850 hPa",
        fontsize=16, pad = 30
    )
    ax.grid(True)
        
    # ---------- legend (custom order + hatch symbols) ----------
    LEGEND_ORDER = [
        "MWR",
        "Radiosonde",
        "FY4B ≤6 km",
        "FY4B 20–25 km",
        "FY4B 30–35 km",
        "ERA5_4pt",
        "ERA5_near",
        "CRA_4pt",
        "CRA_near",
    ]
    
    handles, labels = ax.get_legend_handles_labels()
    
    # 去重（保留第一次出现的顺序）
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    
    hmap = {l: h for h, l in uniq}

    def _styled_handle(label):
    # 关键：如果该label没有真实画出来，就不要生成legend项
        if label not in hmap:
            return None
    
        if label == "MWR":
            return Patch(facecolor=MWR_COLOR, edgecolor=MWR_COLOR,
                         alpha=0.25, hatch="//", linewidth=0.8)
    
        if label == "Radiosonde":
            return Line2D([0], [0], color=RAD_COLOR, lw=2.2)
    
        return hmap.get(label, None)

    new_handles, new_labels = [], []
    for lab in LEGEND_ORDER:
        hh = _styled_handle(lab)
        if hh is not None:
            new_handles.append(hh)
            new_labels.append(lab)
    
    # 可选：把没写进 LEGEND_ORDER 的其它条目追加到最后
    for lab in hmap.keys():
        if lab not in set(LEGEND_ORDER):
            hh = _styled_handle(lab)
            if hh is not None:
                new_handles.append(hh)
                new_labels.append(lab)
    
    leg = ax.legend(
        new_handles, new_labels,
        frameon=True, fontsize=16, loc="lower left",
        handler_map={tuple: HandlerTuple(ndivide=None)}
    )
    
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("white")
    frame.set_linewidth(1.0)
    frame.set_alpha(1.0)

    plt.tight_layout()

    # ---------- save ----------
    png_path = OUT_DIR / f"{ev_name}_TEMP_band_mean_alltypes_P300_850.png"
    #tif_path = OUT_DIR / f"{ev_name}_TEMP_band_mean_alltypes_P300_850.tif"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    # fig.savefig(tif_path, dpi=300, bbox_inches="tight", transparent=True,
    #             pil_kwargs={"compression": "tiff_adobe_deflate"})
    plt.close(fig)

print(f"[OK] Saved 3 event figures to: {OUT_DIR}")



###############"raw data profile not in rain"##################

# =========================================================
# 0) OUTPUT
# =========================================================
OUT_DIR = Path(r"E:\Beijing2024") / "raw data profile not in rain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1) PERIODS (plot windows)
# =========================================================
periods = {
    "P1_20240606_20240609": ("2024-06-06", "2024-06-09"),
    "P2_20240805_20240810": ("2024-08-05", "2024-08-10"),
}

# =========================================================
# 2) RAIN EVENTS TO EXCLUDE (3 events)
#   - minute windows for MWR + Radiosonde
#   - hour windows for FY4B + Reanalysis
# =========================================================
RAIN_EVENTS_MINUTE = [
    (pd.Timestamp("2024-08-09 12:46"), pd.Timestamp("2024-08-10 00:03")),
    (pd.Timestamp("2024-06-07 09:16"), pd.Timestamp("2024-06-07 15:00")),
    (pd.Timestamp("2024-08-10 16:20"), pd.Timestamp("2024-08-10 17:30")),
]

RAIN_EVENTS_HOUR = [
    (pd.Timestamp("2024-08-09 12:00"), pd.Timestamp("2024-08-10 01:00")),
    (pd.Timestamp("2024-06-07 09:00"), pd.Timestamp("2024-06-07 15:00")),
    (pd.Timestamp("2024-08-10 16:00"), pd.Timestamp("2024-08-10 18:00")),
]

# =========================================================
# 3) PRESSURE FILTER (hPa): keep 300–850 only
# =========================================================
P_MIN, P_MAX = 300.0, 850.0

# =========================================================
# 4) DATAFRAME HANDLES
# =========================================================
try:
    mwr_temp_wide = temp_p_df
except NameError:
    mwr_temp_wide = temp  # if your MWR temp wide table is named "temp"

fy4B_T_df = fy4B_T

era5_4pt  = era5_tq_4pt_filtered
era5_near = era5_tq_near_filtered
cra_4pt   = cra_tq_4pt_filtered
cra_near  = cra_tq_near_filtered

rad_df = radiosonde

# =========================================================
# 5) HELPERS
# =========================================================
def _to_naive_dt(s):
    t = pd.to_datetime(s)
    if hasattr(t, "dt"):
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert(None)
    else:
        if getattr(t, "tz", None) is not None:
            t = t.tz_convert(None)
    return t

def mask_in_window(times, start, end):
    t = _to_naive_dt(times)
    st = pd.Timestamp(start)
    ed = pd.Timestamp(end)
    return (t >= st) & (t <= ed)

def mask_in_any_events(times, events):
    t = _to_naive_dt(times)
    m = np.zeros(len(t), dtype=bool)
    for (st, ed) in events:
        st = pd.Timestamp(st)
        ed = pd.Timestamp(ed)
        m |= (t >= st) & (t <= ed)
    return m

def get_numeric_pressure_cols(df, exclude_cols=None, pmin=P_MIN, pmax=P_MAX):
    exclude_cols = set(exclude_cols or [])
    cols = [c for c in df.columns if c not in exclude_cols]
    pres, keep = [], []
    for c in cols:
        try:
            p = float(str(c).strip())
            if (p >= pmin) and (p <= pmax):
                pres.append(p)
                keep.append(c)
        except Exception:
            continue
    if len(keep) == 0:
        raise ValueError(f"未找到 {pmin:.0f}–{pmax:.0f} hPa 范围内的 pressure 列（宽表列名应为压力数值）。")
    order = np.argsort(pres)[::-1]  # descending pressure
    pres_sorted = np.array([pres[i] for i in order], dtype=float)
    cols_sorted = [keep[i] for i in order]
    return pres_sorted, cols_sorted

def ensure_time_index_for_wide(df, time_col_candidates=("BJT", "BJT_aligned", "time", "datetime")):
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = _to_naive_dt(out.index)
        return out.sort_index()
    time_col = None
    for c in time_col_candidates:
        if c in out.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("宽表没有DatetimeIndex，也找不到时间列（BJT/BJT_aligned/time/datetime）。")
    out[time_col] = _to_naive_dt(out[time_col])
    return out.set_index(time_col).sort_index()

def envelope_from_wide_excluding_events(df_wide, start, end, exclude_events, pmin=P_MIN, pmax=P_MAX):
    df = ensure_time_index_for_wide(df_wide)

    m_period = mask_in_window(df.index, start, end)
    m_rain   = mask_in_any_events(df.index, exclude_events)
    sub = df.loc[m_period & (~m_rain)]
    if sub.empty:
        return None

    pres_vals, pres_cols = get_numeric_pressure_cols(sub, pmin=pmin, pmax=pmax)
    block = sub[pres_cols].astype(float)

    lo = block.min(axis=0, skipna=True).values.astype(float)
    hi = block.max(axis=0, skipna=True).values.astype(float)
    mean = block.mean(axis=0, skipna=True).values.astype(float)
    return pres_vals, lo, hi, mean

def envelope_from_long_excluding_events(df_long, start, end, time_col, p_col, v_col,
                                       exclude_events, extra_mask=None, pmin=P_MIN, pmax=P_MAX):
    df = df_long.copy()
    df[time_col] = _to_naive_dt(df[time_col])

    m_period = mask_in_window(df[time_col], start, end)
    m_rain   = mask_in_any_events(df[time_col], exclude_events)
    m_press  = (df[p_col].astype(float) >= pmin) & (df[p_col].astype(float) <= pmax)

    mask = m_period & (~m_rain) & m_press
    if extra_mask is not None:
        mask = mask & extra_mask(df)

    sub = df.loc[mask, [time_col, p_col, v_col]].copy()
    if sub.empty:
        return None

    gp = sub.groupby([time_col, p_col])[v_col].mean().reset_index()
    piv = gp.pivot(index=p_col, columns=time_col, values=v_col)
    piv = piv.sort_index(ascending=False)

    y = piv.index.values.astype(float)
    lo = piv.min(axis=1, skipna=True).values.astype(float)
    hi = piv.max(axis=1, skipna=True).values.astype(float)
    mean = piv.mean(axis=1, skipna=True).values.astype(float)
    return y, lo, hi, mean


#下面这个能画底纹hatch
def plot_band_and_mean(ax, y, lo, hi, mean, color, label,
                       alpha_fill=0.25, lw_mean=2.2,
                       hatch=None, hatch_lw=0.8, hatch_alpha=0.9,
                       z_fill=1, z_line=10):
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    mean = np.asarray(mean, dtype=float)

    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(mean)
    if ok.sum() < 2:
        return

    # (1) 底层：保持原来的填充颜色不变
    ax.fill_betweenx(
        y[ok], lo[ok], hi[ok],
        color=color, alpha=alpha_fill, linewidth=0,
        zorder=z_fill
    )

    # (2) 叠加层：只加纹理，不改变底色
    if hatch is not None:
        poly_h = ax.fill_betweenx(
            y[ok], lo[ok], hi[ok],
            facecolor="none",          # 关键：不盖住底色
            edgecolor=color,           # 纹理颜色跟随数据颜色（也可改成 "k" 更醒目）
            linewidth=hatch_lw,
            alpha=hatch_alpha,
            zorder=z_fill + 0.2
        )
        poly_h.set_hatch(hatch)

    # 均值线：最上层
    ax.plot(mean[ok], y[ok], color=color, linestyle="-", lw=lw_mean, label=label, zorder=z_line)

def style_axes(ax):
    ax.set_xlabel("Temperature (K)", fontsize=22)
    ax.set_ylabel("Pressure (hPa)", fontsize=22)
    ax.tick_params(axis="both", labelsize=22,width=2, length=4)

    ax.set_xlim(205, 305)
    ax.set_ylim(850, 300)

    ax.xaxis.set_major_locator(MultipleLocator(10))     # x every 5 K
    ax.yaxis.set_major_locator(MultipleLocator(50))    # y every 25 hPa
    ax.grid(True, which="major", alpha=0.25)
    # spine linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(2)

# =========================================================
# 6) FIXED STYLES (unchanged)
# =========================================================
MWR_COLOR = "red"

FY_STYLE = {
    "≤6 km": "darkgreen",
    "20–25 km": "lime",
    "30–35 km": "chartreuse",
}
def fy_cat_mask(df, cat):
    d = df["distance_km"].astype(float)
    if cat == "≤6 km":
        return d <= 6
    if cat == "20–25 km":
        return (d >= 20) & (d <= 25)
    if cat == "30–35 km":
        return (d >= 30) & (d <= 35)
    return pd.Series(False, index=df.index)

REAN_STYLE = {
    "ERA5_4pt":  ("dodgerblue", era5_4pt,  "pressure_level"),
    "ERA5_near": ("blue",         era5_near, "pressure_level"),
    "CRA_4pt":   ("magenta",  cra_4pt,   "Level (hPa)"),
    "CRA_near":  ("violet",    cra_near,  "Level (hPa)"),
}

RAD_COLOR = "gold"

# =========================================================
# 7) MAIN: 2 figures (P1/P2), exclude rain events, Temperature only
# =========================================================
for pname, (pstart_date, pend_date) in periods.items():
    # inclusive full-day window
    st = pd.Timestamp(pstart_date).normalize()
    ed = pd.Timestamp(pend_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    fig, ax = plt.subplots(figsize=(7.8, 9.4))

    # ---------- MWR (1-min wide): exclude minute rain events ----------
    try:
        env = envelope_from_wide_excluding_events(mwr_temp_wide, st, ed, exclude_events=RAIN_EVENTS_MINUTE)
        if env is not None:
            y, lo, hi, mean = env
            #plot_band_and_mean(ax, y, lo, hi, mean, MWR_COLOR, "MWR")
            plot_band_and_mean(ax, y, lo, hi, mean, MWR_COLOR, "MWR",
                   alpha_fill=0.25, hatch="/", z_fill=1, z_line=20)

        else:
            print(f"[WARN] {pname}: MWR non-rain window has no data (after excluding events).")
    except Exception as e:
        print(f"[WARN] {pname}: MWR failed -> {e}")

    # ---------- FY4B (2-hour long): exclude hour rain events ----------
    try:
        for cat, col in FY_STYLE.items():
            env = envelope_from_long_excluding_events(
                fy4B_T_df, st, ed,
                time_col="BJT",
                p_col="pressure_hPa",
                v_col="T_K",
                exclude_events=RAIN_EVENTS_HOUR,
                extra_mask=lambda df, c=cat: fy_cat_mask(df, c),
            )
            if env is None:
                print(f"[WARN] {pname}: FY4B {cat} has no data (non-rain, 300–850).")
                continue
            y, lo, hi, mean = env
            plot_band_and_mean(ax, y, lo, hi, mean, col, f"FY4B {cat}")
    except Exception as e:
        print(f"[WARN] {pname}: FY4B failed -> {e}")

    # ---------- Reanalysis (hourly long): exclude hour rain events ----------
    try:
        for label, (col, df0, pcol) in REAN_STYLE.items():
            env = envelope_from_long_excluding_events(
                df0, st, ed,
                time_col="BJT",
                p_col=pcol,
                v_col="Temp (K)",
                exclude_events=RAIN_EVENTS_HOUR,
            )
            if env is None:
                print(f"[WARN] {pname}: {label} has no data (non-rain, 300–850).")
                continue
            y, lo, hi, mean = env
            plot_band_and_mean(ax, y, lo, hi, mean, col, f"{label}")
    except Exception as e:
        print(f"[WARN] {pname}: Reanalysis failed -> {e}")

    # ---------- Radiosonde (long): exclude minute rain events ----------
    try:
        rad = rad_df.copy()
        rad["time"] = _to_naive_dt(rad["time"])
        rad["T_K"] = rad["temperature_C"].astype(float) + 273.15

        env = envelope_from_long_excluding_events(
            rad, st, ed,
            time_col="time",
            p_col="pressure_hPa",
            v_col="T_K",
            exclude_events=RAIN_EVENTS_MINUTE,
        )
        if env is None:
            print(f"[WARN] {pname}: Radiosonde has no data (non-rain, 300–850).")
        else:
            y, lo, hi, mean = env
            #plot_band_and_mean(ax, y, lo, hi, mean, RAD_COLOR, "Radiosonde")
            plot_band_and_mean(ax, y, lo, hi, mean, RAD_COLOR, "Radiosonde",
                   alpha_fill=0.25, hatch="-", z_fill=2, z_line=25)

            
    except Exception as e:
        print(f"[WARN] {pname}: Radiosonde failed -> {e}")

    # ---------- cosmetics ----------
    style_axes(ax)
    ax.set_title(
        #f"Temperature Profile Range (filled, α=0.250) & Mean (solid)\n"
        f"{st:%Y-%m-%d %H:%M} to {ed:%Y-%m-%d %H:%M}  |  300–850 hPa  |  Rain events excluded",
        fontsize=20,pad=30
        )
        # ---------- legend (custom order + hatch symbols) ----------
    LEGEND_ORDER = [
        "MWR",
        "Radiosonde",
        "FY4B ≤6 km",
        "FY4B 20–25 km",
        "FY4B 30–35 km",
        "ERA5_4pt",
        "ERA5_near",
        "CRA_4pt",
        "CRA_near",
    ]
    
    handles, labels = ax.get_legend_handles_labels()
    
    # 去重（保留第一次出现的顺序）
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    
    hmap = {l: h for h, l in uniq}
    
    def _styled_handle(label):
        if label == "MWR":
            patch = Patch(facecolor=MWR_COLOR, edgecolor=MWR_COLOR, alpha=0.25, hatch="//", linewidth=0.8)
            #line  = Line2D([0], [0], color=MWR_COLOR, lw=2.2)
            return (patch)#, line
        if label == "Radiosonde":
            patch = Patch(facecolor=RAD_COLOR, edgecolor=RAD_COLOR, alpha=0.25, hatch="--", linewidth=0.8)
            #line  = Line2D([0], [0], color=RAD_COLOR, lw=2.2)
            return (patch)#, line
        return hmap.get(label, None)
    
    new_handles, new_labels = [], []
    for lab in LEGEND_ORDER:
        hh = _styled_handle(lab)
        if hh is not None:
            new_handles.append(hh)
            new_labels.append(lab)
    
    # 可选：把没写进 LEGEND_ORDER 的其它条目追加到最后
    for lab in hmap.keys():
        if lab not in set(LEGEND_ORDER):
            hh = _styled_handle(lab)
            if hh is not None:
                new_handles.append(hh)
                new_labels.append(lab)
    
    leg = ax.legend(
        new_handles, new_labels,
        frameon=True, fontsize=16, loc="lower left",
        handler_map={tuple: HandlerTuple(ndivide=None)}
    )
    
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("white")
    frame.set_linewidth(1.0)
    frame.set_alpha(1.0)


    # ---------- save ----------
    png_path = OUT_DIR / f"{pname}_TEMP_band_mean_alltypes_nonrain_P300_850.png"
    #tif_path = OUT_DIR / f"{pname}_TEMP_band_mean_alltypes_nonrain_P300_850.tif"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    #fig.savefig(tif_path, dpi=300, bbox_inches="tight", transparent=True,
    #            pil_kwargs={"compression": "tiff_adobe_deflate"})
    plt.close(fig)

print(f"[OK] Saved 2 non-rain period figures to: {OUT_DIR}")

