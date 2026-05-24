import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# -----------------------------
# 1. 构造 25 个点的 FDR 数据
# 单位：%
# 要求：
# - 相邻 FDR 绝不相等
# - 包含上升、下降、局部波动
# - 整体趋势随 PSM 排名位置增加而升高
# -----------------------------

x = np.arange(1, 26)

fdr = np.array([
    0.10, 0.14, 0.12, 0.18, 0.23,
    0.20, 0.29, 0.35, 0.31, 0.42,
    0.50, 0.46, 0.60, 0.69, 0.64,
    0.78, 0.89, 0.83, 1.02, 1.14,
    1.08, 1.28, 1.43, 1.36, 1.55
])

# 检查：相邻 FDR 不允许相等
if np.any(np.diff(fdr) == 0):
    raise ValueError("FDR 数据存在相邻值相等的情况，请检查数据。")

# -----------------------------
# 2. 正确计算 q-value
# q_i = min_{j >= i} FDR_j
# 从右向左取累计最小值
# -----------------------------

q_value = np.minimum.accumulate(fdr[::-1])[::-1]

# -----------------------------
# 3. 设置中文字体
# -----------------------------

candidate_fonts = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS"
]

available_fonts = {font.name for font in font_manager.fontManager.ttflist}

for font in candidate_fonts:
    if font in available_fonts:
        plt.rcParams["font.sans-serif"] = [font]
        break

plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# 4. 绘图
# 横纵比 3:4
# 不使用 marker，因此不会显示圆点
# 字体整体放大
# -----------------------------

plt.figure(figsize=(6, 8), dpi=150)

plt.plot(
    x,
    fdr,
    linewidth=3.0,
    label="累计 FDR"
)

plt.step(
    x,
    q_value,
    where="post",
    linewidth=3.4,
    label="q-value（从右向左累计最小 FDR）"
)

plt.xlabel("PSM 排名位置（按得分从高到低）", fontsize=16)
plt.ylabel("FDR / q-value（%）", fontsize=16)

plt.xlim(1, 25)
plt.ylim(0, max(fdr.max(), q_value.max()) * 1.15)

plt.xticks(np.arange(1, 26, 2), fontsize=14)
plt.yticks(np.arange(0, 1.9, 0.2), fontsize=14)

plt.grid(True, linestyle="--", alpha=0.45)
plt.legend(loc="upper left", fontsize=13)

note = (
    "说明：蓝线为按得分阈值累计计算的 FDR，\n"
    "相邻 PSM 位置的 FDR 均发生变化；\n"
    "橙线为 q-value，按 q_i = min_{j>=i} FDR_j 计算，\n"
    "因此单调不下降且不高于对应位置的 FDR。"
)

plt.figtext(
    0.10,
    0.01,
    note,
    ha="left",
    fontsize=12
)

plt.tight_layout(rect=(0, 0.16, 1, 1))
plt.show()