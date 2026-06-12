import numpy as np
import pandas as pd

def extract_micp_features(df):
    """
    df需要包含：
    - 'Radius' (孔喉半径, μm 或 logR)
    - 'Swanson' or 'Saturation' (累计汞饱和度)
    """

    features = {}

    radius = df["Radius"].values
    sat = df["Saturation"].values

    # --- 插值函数 ---
    def get_radius_at_percent(p):
        return np.interp(p, sat, radius)

    # ✅ 关键孔喉参数
    features["R10"] = get_radius_at_percent(10)
    features["R20"] = get_radius_at_percent(20)
    features["R35"] = get_radius_at_percent(35)
    features["R50"] = get_radius_at_percent(50)
    features["R80"] = get_radius_at_percent(80)

    # ✅ 分布特征
    features["radius_mean"] = np.mean(radius)
    features["radius_std"] = np.std(radius)
    features["radius_skew"] = pd.Series(radius).skew()
    features["radius_kurt"] = pd.Series(radius).kurt()

    # ✅ 曲线斜率（简单proxy）
    d_sat = np.gradient(sat)
    d_rad = np.gradient(radius)
    slope = np.divide(d_sat, d_rad, out=np.zeros_like(d_sat), where=d_rad!=0)

    features["slope_mean"] = np.mean(slope)
    features["slope_max"] = np.max(slope)

    # ✅ 峰值特征（简单方法）
    peak_index = np.argmax(slope)
    features["peak_radius"] = radius[peak_index]

    return pd.DataFrame([features])