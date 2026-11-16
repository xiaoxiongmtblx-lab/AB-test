# -*- coding: utf-8 -*-
"""AB 实验统计分析引擎。

方法论红线（本文件所有函数遵守）：
1. 只对数据做统计推断，结论文案由模板 + 计算数字生成，不做任何主观修饰；
2. p 值只用于「拒绝/不拒绝原假设」的判断，不表述为「H0 为真的概率」；
3. 事后功效（post-hoc power）仅作参考展示，并明确标注其局限；
4. 蒙特卡洛模拟仅用于统计方法论演示（如功效曲线），与真实业务数据严格区分。

统计实现：
- 双比例检验：statsmodels proportions_ztest（合并方差 Wald z 检验）
- 置信区间：每组转化率用 Wilson 区间（小样本性质好于 Wald），
  差值的 CI 用非合并方差的正态近似（教科书标准做法，注释中说明局限）
- SRM（样本比例失衡）检验：卡方拟合优度
- 功效/样本量：基于 Cohen's h 效应量的正态近似（statsmodels NormalIndPower）
- 多重比较：Bonferroni 与 Holm（step-down）校正
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


# ---------------- 数据清洗（聚合优先架构） ----------------

@dataclass
class CleaningReport:
    """AB 数据清洗报告：每一步涉及多少行、为什么（可审计）。

    设计说明：不物理删除行，而是添加 valid/first_user 标志列，
    统计一律由 groupby 聚合得出（生产管道同风格，也避免大表反复物化）。"""
    raw_rows: int = 0
    dropped_mismatch: int = 0     # 组与页面不匹配（分流故障）
    dropped_duplicate: int = 0    # 重复用户（保留首次出现）
    dropped_missing: int = 0      # 关键字段缺失
    final_rows: int = 0

    def to_dict(self):
        return {
            "原始行数": self.raw_rows,
            "剔除·组与页面不匹配": self.dropped_mismatch,
            "剔除·重复用户（保留首次）": self.dropped_duplicate,
            "剔除·关键字段缺失": self.dropped_missing,
            "有效分析行数": self.final_rows,
        }


def prepare_ab_data(ab: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """营销页 AB 数据准备：加标志列而非物理删行。

    1. valid：组与落地页匹配（control↔old_page / treatment↔new_page）才有效；
    2. first_user：按时间排序后同一用户的首次出现（重复分流身份不稳定）；
    3. clean = valid & first_user，后续所有统计用它过滤。
    """
    rep = CleaningReport(raw_rows=len(ab))
    df = ab.dropna(subset=["user_id", "group", "landing_page", "converted"]).copy()
    rep.dropped_missing = len(ab) - len(df)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["user_rank"] = df.groupby("user_id").cumcount()
    df["first_user"] = df["user_rank"] == 0
    df["valid"] = (
        ((df["group"] == "control") & (df["landing_page"] == "old_page"))
        | ((df["group"] == "treatment") & (df["landing_page"] == "new_page"))
    )
    df["clean"] = df["valid"] & df["first_user"]

    # 清洗报告数字由聚合得出（不物化中间表）
    rep.dropped_mismatch = int((~df["valid"]).sum())
    rep.dropped_duplicate = int((df["first_user"] & ~df["valid"]).sum() + (~df["first_user"]).sum() - int((~df["first_user"] & ~df["valid"]).sum()))
    # 更直观口径：重复用户中有效页面匹配的那部分 + 不匹配且非首行的交叉避免重复计数，
    # 采用标准口径：重复用户行数 - 其中组页不匹配行数（与 mismatch 分别计）
    rep.dropped_duplicate = int((~df["first_user"] & df["valid"]).sum())
    rep.final_rows = int(df["clean"].sum())
    return df, rep


def ab_group_stats(df: pd.DataFrame) -> dict:
    """从标志列帧中提取清洗后两组的 (转化数, 样本量)——纯聚合。"""
    g = df.groupby(["group", "clean"])["converted"].agg(["sum", "count"])
    cell = g.reset_index()
    c = cell[(cell["group"] == "control") & (cell["clean"])]
    t = cell[(cell["group"] == "treatment") & (cell["clean"])]
    return {
        "x1": int(c["sum"].iloc[0]) if len(c) else 0,
        "n1": int(c["count"].iloc[0]) if len(c) else 0,
        "x2": int(t["sum"].iloc[0]) if len(t) else 0,
        "n2": int(t["count"].iloc[0]) if len(t) else 0,
        "raw_n1": int((df["group"] == "control").sum()),
        "raw_n2": int((df["group"] == "treatment").sum()),
    }


def clean_cookie_cats(cc: pd.DataFrame) -> pd.DataFrame:
    """清洗 Cookie Cats 数据：布尔列防御式转换（兼容 TRUE/FALSE 字符串或原生布尔）。"""
    df = cc.copy()
    for col in ["retention_1", "retention_7"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False, "1": True, "0": False})
    df = df.dropna(subset=["userid", "version"])
    return df


# ---------------- SRM 检验 ----------------

@dataclass
class SRMResult:
    n_control: int
    n_treatment: int
    expected_ratio: float
    chi2: float
    p_value: float
    passed: bool

    def summary(self) -> str:
        if self.passed:
            return (
                f"SRM 检验通过（卡方 p={self.p_value:.3f} > 0.001）："
                f"两组样本量 {self.n_control:,} vs {self.n_treatment:,}，"
                f"与预期 {self.expected_ratio:.0%}:{1 - self.expected_ratio:.0%} 分流比例相容。"
            )
        return (
            f"SRM 检验未通过（卡方 p={self.p_value:.2e} ≤ 0.001）：样本比例显著偏离 "
            f"{self.expected_ratio:.0%}:{1 - self.expected_ratio:.0%}，实验分流或数据管道存在故障，"
            f"后续一切检验结果都不可信，应先排查工程问题。"
        )


def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5, alpha: float = 0.001) -> SRMResult:
    """样本比例失衡检验（卡方拟合优度）。

    实践惯例用较严格的 α=0.001：因为 SRM 影响的是实验有效性本身，
    宁可错杀（重跑实验）也不放过。
    """
    total = n_control + n_treatment
    observed = np.array([n_control, n_treatment], dtype=float)
    expected = np.array([expected_ratio, 1 - expected_ratio]) * total
    chi2, p = sps.chisquare(observed, expected)
    return SRMResult(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_ratio=expected_ratio,
        chi2=float(chi2),
        p_value=float(p),
        passed=bool(p > alpha),
    )


# ---------------- 双比例检验 ----------------

@dataclass
class PropTestResult:
    x1: int; n1: int  # 对照组：转化数 / 样本量
    x2: int; n2: int  # 实验组
    p1: float; p2: float
    diff: float          # p2 - p1（实验组 - 对照组）
    rel_lift: float      # 相对提升 (p2-p1)/p1
    z: float
    p_value: float       # 双侧
    ci_low: float        # 差值 CI
    ci_high: float
    ci1: tuple           # 对照组 Wilson CI
    ci2: tuple           # 实验组 Wilson CI
    cohens_h: float
    alpha: float
    significant: bool

    def verdict(self) -> str:
        """由数字直接生成的结论文案（无主观修饰）。"""
        lift_pct = self.rel_lift * 100
        if self.significant:
            direction = "高于" if self.diff > 0 else "低于"
            return (
                f"在 α={self.alpha} 的双侧检验下，实验组转化率（{self.p2:.2%}）显著{direction}对照组（{self.p1:.2%}），"
                f"绝对差异 {self.diff*100:+.2f}pp（95% CI [{self.ci_low*100:.2f}, {self.ci_high*100:.2f}]pp），"
                f"相对提升 {lift_pct:+.1f}%。CI 不包含 0，效应方向可信；幅度以 CI 区间为准，而非点估计。"
            )
        return (
            f"在 α={self.alpha} 的双侧检验下未能拒绝原假设（p={self.p_value:.3f}）："
            f"实验组 {self.p2:.2%} vs 对照组 {self.p1:.2%}，差异 {self.diff*100:+.2f}pp "
            f"（95% CI [{self.ci_low*100:.2f}, {self.ci_high*100:.2f}]pp，包含 0）。"
            f"注意：这不等于「证明无差异」——只是本样本量下证据不足；"
            f"若要确认可检测到的最小差异，请看「实验设计器」页的 MDE 计算。"
        )


def two_proportion_test(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> PropTestResult:
    """双比例 z 检验（合并方差）+ Wilson CI + Cohen's h。

    约定：1=对照组，2=实验组；diff = p2 - p1。
    """
    p1, p2 = x1 / n1, x2 / n2
    z, pval = proportions_ztest([x1, x2], [n1, n2], alternative="two-sided")
    # 差值 CI：非合并方差正态近似（Wald）。样本量 > 数千时与精确方法差异可忽略。
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = sps.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    ci_low, ci_high = diff - z_crit * se_diff, diff + z_crit * se_diff
    ci1 = proportion_confint(x1, n1, alpha=alpha, method="wilson")
    ci2 = proportion_confint(x2, n2, alpha=alpha, method="wilson")
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    return PropTestResult(
        x1=x1, n1=n1, x2=x2, n2=n2, p1=p1, p2=p2,
        diff=diff, rel_lift=diff / p1 if p1 else 0.0,
        z=float(z), p_value=float(pval),
        ci_low=float(ci_low), ci_high=float(ci_high),
        ci1=(float(ci1[0]), float(ci1[1])), ci2=(float(ci2[0]), float(ci2[1])),
        cohens_h=float(h), alpha=alpha, significant=bool(pval < alpha),
    )


# ---------------- 功效与样本量 ----------------

def post_hoc_power(result: PropTestResult, alpha: float = 0.05) -> float:
    """事后（观测）功效——仅作参考。

    局限说明：基于观测效应量的事后功效与 p 值一一对应，
    信息量有限；真正有决策价值的是【事前】功效分析（见 required_sample_size）。
    """
    return float(
        NormalIndPower().power(
            effect_size=abs(result.cohens_h),
            nobs1=result.n1,
            ratio=result.n2 / result.n1,
            alpha=alpha,
        )
    )


def required_sample_size(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0) -> int:
    """事前功效分析：给定基线转化率与最小可检测差异，计算每组所需样本量。

    效应量用 Cohen's h（比例数据的标准效应量）。
    """
    p_alt = p_base + mde_abs
    h = abs(2 * np.arcsin(np.sqrt(p_alt)) - 2 * np.arcsin(np.sqrt(p_base)))
    n = NormalIndPower().solve_power(effect_size=h, power=power, alpha=alpha, ratio=ratio, alternative="two-sided")
    return int(np.ceil(n))


def mde_given_n(n_per_group: int, p_base: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """反向问题：现有样本量下，能以给定功效检测到的最小差异（MDE）。

    先解出可检测的 Cohen's h，再换算回比例差。
    """
    h = NormalIndPower().solve_power(effect_size=None, nobs1=n_per_group, power=power, alpha=alpha, ratio=1.0, alternative="two-sided")
    p2 = np.sin(np.arcsin(np.sqrt(p_base)) + h / 2) ** 2
    return float(p2 - p_base)


# ---------------- 多重比较校正 ----------------

def bonferroni_holm(p_values: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Bonferroni 与 Holm(step-down) 多重比较校正。

    返回 DataFrame：各假设的原始 p、两种校正后的 p 与显著性判断。
    Holm 步骤：p 升序排列，第 i 个校正 p = max_{j<=i} (m-j+1)*p_j（截断到 1，保持单调）。
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    rows = []
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = (m - i) * p
        running = max(running, adj)
        holm = min(1.0, running)
        bonf = min(1.0, m * p)
        rows.append({
            "指标": name,
            "原始 p 值": p,
            "Bonferroni 校正 p": bonf,
            "Holm 校正 p": holm,
            "Bonferroni 显著": bool(bonf < alpha),
            "Holm 显著": bool(holm < alpha),
        })
    return pd.DataFrame(rows)


# ---------------- Peek（偷看）轨迹 ----------------

def peeking_trajectory(df: pd.DataFrame, checkpoints: int = 30) -> pd.DataFrame:
    """按时间累积样本，计算每个检查点的 z 检验 p 值（真实数据，向量化实现）。

    用途：可视化「反复看 p 值再决定停不停」如何虚增假阳性
    （nominal α=0.05 下，若在每个检查点都用 0.05 判停，整体假阳性率远超 5%）。
    实现：排序后用 cumsum 计算两组累积转化数/样本量，在等距检查点采样。
    """
    is_c = (df["group"] == "control").to_numpy()
    is_t = ~is_c
    clean = df["clean"].to_numpy()
    conv = df["converted"].to_numpy().astype(np.int64)

    # int64 权重（乘法提升类型，避免布尔 cumsum 内核）
    one = np.ones(len(df), dtype=np.int64)
    w_c = one * (is_c & clean)
    w_t = one * (is_t & clean)

    n_c = w_c.cumsum()
    n_t = w_t.cumsum()
    x_c = (w_c * conv).cumsum()
    x_t = (w_t * conv).cumsum()

    total = len(df)
    step = max(1, total // checkpoints)
    rows = []
    for k in range(step, total + 1, step):
        if n_c[k - 1] < 100 or n_t[k - 1] < 100:
            continue
        _, pval = proportions_ztest([int(x_c[k - 1]), int(x_t[k - 1])], [int(n_c[k - 1]), int(n_t[k - 1])])
        rows.append({
            "累积样本": int(n_c[k - 1] + n_t[k - 1]),
            "累积对照转化率": x_c[k - 1] / n_c[k - 1],
            "累积实验转化率": x_t[k - 1] / n_t[k - 1],
            "p 值": float(pval),
        })
    return pd.DataFrame(rows)


# ---------------- 蒙特卡洛功效模拟（统计教学用）----------------

def monte_carlo_power(p_base: float, true_lift: float, n_per_group: int, n_sim: int = 2000, alpha: float = 0.05, seed: int = 42) -> dict:
    """蒙特卡洛模拟：真实提升为 true_lift 时，检验的拒绝率（=经验功效）。

    ⚠️ 本函数模拟的是「统计方法的行为」（二项抽样 + z 检验），
    不涉及任何真实业务数据，仅用于方法论教学：
    - 展示功效随样本量/真实效应的收敛关系
    - 与解析功效公式交叉验证（两者应接近）
    """
    rng = np.random.default_rng(seed)
    p_alt = p_base + true_lift
    x1 = rng.binomial(n_per_group, p_base, size=n_sim)
    x2 = rng.binomial(n_per_group, p_alt, size=n_sim)

    # 向量化的合并方差 z 检验（与 proportions_ztest 同口径，避免 2000 次 Python 循环）
    p1 = x1 / n_per_group
    p2 = x2 / n_per_group
    pp = (x1 + x2) / (2 * n_per_group)
    se = np.sqrt(pp * (1 - pp) * 2 / n_per_group)
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = np.where(se > 0, (p2 - p1) / se, 0.0)
    pvals = 2 * (1 - sps.norm.cdf(np.abs(zstat)))

    # 解析功效（Cohen's h）作交叉验证
    h = abs(2 * np.arcsin(np.sqrt(p_alt)) - 2 * np.arcsin(np.sqrt(p_base)))
    analytic = float(NormalIndPower().power(effect_size=h, nobs1=n_per_group, ratio=1.0, alpha=alpha))

    return {
        "p_base": p_base,
        "true_lift": true_lift,
        "n_per_group": n_per_group,
        "n_sim": n_sim,
        "alpha": alpha,
        "empirical_power": float((pvals < alpha).mean()),
        "analytic_power": analytic,
        "p_value_mean": float(np.mean(pvals)),
    }
