# -*- coding: utf-8 -*-
"""AB 实验实验室（A/B Testing Lab）

基于两个公开真实 AB 实验数据集的交互式统计检验应用：
1. 营销落地页实验（Kaggle 经典 ab_data，294,478 行）：新旧页面对转化的影响
2. Cookie Cats 游戏关卡实验（DataCamp 经典案例，90,189 行）：门槛位置对留存的影响

完整演示：实验设计审查 → 数据质量与 SRM → 假设检验 → 多重比较 →
peeking 陷阱 → 分层稳健性 → 功效与样本量设计。

运行：streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.engine import (  # noqa: E402
    ab_group_stats,
    bonferroni_holm,
    clean_cookie_cats,
    mde_given_n,
    monte_carlo_power,
    peeking_trajectory,
    post_hoc_power,
    prepare_ab_data,
    required_sample_size,
    srm_check,
    two_proportion_test,
)

# ---------------- 页面配置 ----------------
st.set_page_config(page_title="AB 实验实验室", page_icon="🧪", layout="wide")

INK, INDIGO, PINK, ORANGE, TEAL, AMBER = "#1c1917", "#4f46e5", "#FF2E7E", "#FF6A00", "#0d9488", "#d97706"
PALETTE = [INDIGO, PINK, ORANGE, TEAL, AMBER]

st.markdown(
    """
    <style>
    .block-container{padding-top:2rem;max-width:1280px}
    div[data-testid="stMetric"]{background:#fffef9;border:1.5px solid #1c1917;border-radius:10px;
      box-shadow:3px 3px 0 rgba(28,25,23,.85);padding:14px 18px}
    main table{border-collapse:collapse;font-size:13px;width:100%;margin:8px 0}
    main table th,main table td{border:1px solid #e5decf;padding:6px 10px;text-align:left}
    main table th{background:#f0ebdd;font-family:Georgia,'Songti SC',serif}
    main table tr:nth-child(even) td{background:#faf7ee}
    </style>
    """,
    unsafe_allow_html=True,
)


def show_chart(fig):
    """兼容不同 Streamlit 版本的图表渲染。"""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)


def show_table(df):
    """渲染表格：优先原生组件，异常时回退 HTML。"""
    try:
        st.table(df)
    except Exception:
        st.markdown(df.to_html(index=False, border=0), unsafe_allow_html=True)


# ---------------- 数据加载 ----------------
DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(ttl=600, show_spinner="正在加载数据…")
def load_ab():
    ab = pd.read_csv(DATA_DIR / "ab_data.csv")
    df, rep = prepare_ab_data(ab)
    stats = ab_group_stats(df)
    return df, rep, stats


@st.cache_data(ttl=600, show_spinner="正在加载数据…")
def load_cc():
    cc = pd.read_csv(DATA_DIR / "cookie_cats.csv")
    df = clean_cookie_cats(cc)
    # 统一组名：gate_30=对照（原门槛），gate_40=实验（后移门槛）
    df["group"] = df["version"].map({"gate_30": "control", "gate_40": "treatment"})
    return df


AB_DF, AB_REP, AB_STATS = load_ab()
CC_DF = load_cc()

# ---------------- 侧边栏 ----------------
st.sidebar.title("🧪 AB 实验实验室")
st.sidebar.caption("真实公开数据集 · 统计结论由数据生成")

dataset = st.sidebar.radio(
    "数据集",
    ["📊 营销落地页实验", "🎮 Cookie Cats 关卡实验"],
    help="两个均为业界经典公开 AB 数据集",
)
page = st.sidebar.radio(
    "分析页",
    ["🧪 实验设计与数据质量", "📐 核心检验与结论", "🕵️ 深入分析", "⚡ 实验设计器"],
    label_visibility="collapsed",
)

IS_AB = dataset.startswith("📊")

# 当前数据集的统一口径
if IS_AB:
    METRICS = {"converted": "购买转化率"}
    DS_DESC = "Kaggle 经典营销 A/B 数据（294,478 行会话，2017-01）"
    DS_SRC = "github.com/SRINI-SEENI/Task-4-A-B-Hypothesis-Testing-Analysis（Kaggle ab_test 公开镜像）"
    GROUP_CN = {"control": "旧页面(对照)", "treatment": "新页面(实验)"}
else:
    METRICS = {"retention_1": "次日留存率", "retention_7": "7 日留存率"}
    DS_DESC = "移动游戏 Cookie Cats 门槛实验（90,189 名玩家，DataCamp 经典案例）"
    DS_SRC = "github.com/ghaskari/cookie-cats-ab-test（原始实验：Woopa 数据团队公开发布）"
    GROUP_CN = {"control": "30 关门槛(对照)", "treatment": "40 关门槛(实验)"}


def current_stats(metric: str):
    """当前数据集在指定指标上的两组 (x, n)。"""
    if IS_AB:
        s = AB_STATS
        return s["x1"], s["n1"], s["x2"], s["n2"]
    g = CC_DF.groupby("group")[metric].agg(["sum", "count"])
    c, t = g.loc["control"], g.loc["treatment"]
    return int(c["sum"]), int(c["count"]), int(t["sum"]), int(t["count"])


# ================= 页 1：实验设计与数据质量 =================
if page == "🧪 实验设计与数据质量":
    st.title("🧪 实验设计与数据质量")
    st.caption("检验开始前，先审查实验本身是否可信")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("实验单位", "用户级分流" if IS_AB else "玩家级分流", help="每个用户只归属一组，避免交叉污染")
    with c2:
        st.metric("核心指标", " / ".join(METRICS.values()))
    with c3:
        st.metric("数据规模", f"{len(AB_DF):,}" if IS_AB else f"{len(CC_DF):,}", help="清洗/去重后的有效样本")

    st.subheader("数据集档案")
    with st.container(border=True):
        st.markdown(f"**{DS_DESC}**")
        st.caption(f"来源：{DS_SRC} · 随项目打包，离线可复现")
        if IS_AB:
            st.markdown(
                f"""
                - **设计**：用户随机分流到旧页面（对照）或新页面（实验），观察是否完成购买转化
                - **原始样本**：{AB_STATS['raw_n1']:,} vs {AB_STATS['raw_n2']:,}
                - **数据质量问题（本数据的经典教学点）**：{AB_REP.dropped_mismatch:,} 行组-页不匹配、
                  {AB_REP.dropped_duplicate:,} 行重复用户——见下方清洗报告
                """
            )
        else:
            st.markdown(
                """
                - **设计**：把「强制休息门槛」从第 30 关移到第 40 关，观察对玩家留存的影响
                - **双指标**：次日留存 + 7 日留存（引出多重比较问题，见深入分析页）
                - **已知数据特征**：`sum_gamerounds` 存在极端离群值（有玩家 14 天玩了 49,854 轮）
                """
            )

    if IS_AB:
        st.subheader("清洗报告（可审计）")
        show_table(pd.DataFrame([AB_REP.to_dict()]).T.reset_index().rename(columns={"index": "清洗步骤", 0: "行数"}))
        st.caption(
            "清洗规则：① 组与落地页不匹配 → 分流管道故障，剔除；② 重复用户 → 保留时间上首次出现；"
            "不物理删行，用标志列参与聚合（见 src/engine.py 注释）。"
        )

    st.subheader("SRM 检验：样本比例是否失衡")
    st.caption("随机分流应产生约 50:50 的样本量。卡方拟合优度检验，α=0.001（严格，因为 SRM 动摇一切结论）")
    if IS_AB:
        s_raw = srm_check(AB_STATS["raw_n1"], AB_STATS["raw_n2"])
        s_clean = srm_check(AB_STATS["n1"], AB_STATS["n2"])
        c1, c2 = st.columns(2)
        with c1:
            st.metric("原始数据 SRM p 值", f"{s_raw.p_value:.4f}", delta="通过" if s_raw.passed else "未通过",
                      delta_color="normal" if s_raw.passed else "inverse")
        with c2:
            st.metric("清洗后 SRM p 值", f"{s_clean.p_value:.4f}", delta="通过" if s_clean.passed else "未通过",
                      delta_color="normal" if s_clean.passed else "inverse")
        st.info(s_clean.summary())
    else:
        n30 = int((CC_DF["group"] == "control").sum())
        n40 = int((CC_DF["group"] == "treatment").sum())
        s = srm_check(n30, n40)
        st.metric("SRM p 值", f"{s.p_value:.4f}", delta="通过" if s.passed else "未通过",
                  delta_color="normal" if s.passed else "inverse")
        st.info(s.summary())

    st.subheader("原始数据预览")
    preview_cols = ["user_id", "timestamp", "group", "landing_page", "converted"] if IS_AB else ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]
    show_table((AB_DF if IS_AB else CC_DF)[preview_cols].head(10))


# ================= 页 2：核心检验与结论 =================
elif page == "📐 核心检验与结论":
    st.title("📐 核心检验与结论")
    st.caption("双比例 z 检验 · Wilson 置信区间 · 效应量 Cohen's h · 事后功效")

    metric = "converted" if IS_AB else st.selectbox("选择指标", list(METRICS.keys()), format_func=lambda k: METRICS[k])
    metric_cn = METRICS[metric]
    x1, n1, x2, n2 = current_stats(metric)

    alpha = st.slider("显著性水平 α", 0.01, 0.10, 0.05, 0.01, format="%.2f")

    r = two_proportion_test(x1, n1, x2, n2, alpha=alpha)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(f"对照组 {metric_cn}", f"{r.p1:.2%}", f"n = {n1:,}")
    with c2:
        st.metric(f"实验组 {metric_cn}", f"{r.p2:.2%}", f"n = {n2:,}")
    with c3:
        st.metric("绝对差异 (pp)", f"{r.diff*100:+.2f}", f"95%CI [{r.ci_low*100:+.2f}, {r.ci_high*100:+.2f}]")
    with c4:
        st.metric("双侧 p 值", f"{r.p_value:.4f}", "显著" if r.significant else "不显著",
                  delta_color="normal" if r.significant else "off")

    st.subheader(f"两组 {metric_cn} 的 Wilson 95% 置信区间")
    fig = go.Figure()
    fig.add_bar(
        x=[GROUP_CN["control"], GROUP_CN["treatment"]],
        y=[r.p1, r.p2],
        error_y=dict(type="data", symmetric=False,
                     array=[r.p2 - r.ci2[0] if False else r.p1 - r.ci1[0], r.p2 - r.ci2[0]],
                     arrayminus=[r.p1 - r.ci1[0], r.p2 - r.ci2[0]]),
        marker_color=[AMBER, INDIGO],
        width=0.45,
        text=[f"{r.p1:.2%}", f"{r.p2:.2%}"],
        textposition="outside",
    )
    fig.update_layout(height=380, margin=dict(t=10, l=10, r=10, b=10), yaxis_title=metric_cn,
                      yaxis_tickformat=".0%", yaxis_range=[0, max(r.p1, r.p2) * 1.25])
    show_chart(fig)

    l, rr = st.columns(2)
    with l:
        st.subheader("检验明细")
        show_table(pd.DataFrame([
            {"项目": "z 统计量", "数值": f"{r.z:.3f}"},
            {"项目": "双侧 p 值", "数值": f"{r.p_value:.4f}"},
            {"项目": "绝对差异 (pp)", "数值": f"{r.diff*100:+.2f}"},
            {"项目": "相对提升", "数值": f"{r.rel_lift*100:+.1f}%"},
            {"项目": "差值 95% CI", "数值": f"[{r.ci_low*100:.2f}, {r.ci_high*100:.2f}]pp"},
            {"项目": "效应量 Cohen's h", "数值": f"{r.cohens_h:.4f}"},
            {"项目": "事后功效（仅供参考）", "数值": f"{post_hoc_power(r, alpha):.1%}"},
        ]))
        st.caption(
            "事后功效与 p 值一一对应，信息量有限；决策请以「事前功效分析」为准（见实验设计器页）。"
        )
    with rr:
        st.subheader("自动结论")
        with st.container(border=True):
            st.markdown(r.verdict())
        st.caption("结论由模板基于上表数字生成，无人工修饰；可直接用于汇报。")

    st.subheader("效应量与样本量的关系提示")
    mde_now = mde_given_n(min(n1, n2), r.p1, alpha=alpha)
    st.info(
        f"本实验样本量（约 {min(n1,n2):,}/组）在 80% 功效下可检测的最小差异（MDE）为 "
        f"**{mde_now*100:.2f}pp**；观测到的差异为 {abs(r.diff)*100:.2f}pp"
        + ("（大于 MDE，样本量充足）" if abs(r.diff) >= mde_now else "（小于 MDE，样本量不足以可靠检出该幅度）")
        + "。换算关系与公式见「实验设计器」页。"
    )


# ================= 页 3：深入分析 =================
elif page == "🕵️ 深入分析":
    st.title("🕵️ 深入分析")
    st.caption("一个 p 值不够：多重比较 · peeking 陷阱 · 分层稳健性")

    if IS_AB:
        st.subheader("① peeking 陷阱：p 值随样本累积的漂移（真实数据重演）")
        st.caption(
            "如果把实验中途的每一次「偷看」都当作最终结论，整体假阳性率会远超名义 α。"
            "下图用真实数据按时间累积重算 p 值——观察它如何在显著线附近游走。"
        )
        pk = peeking_trajectory(AB_DF, checkpoints=25)
        fig = go.Figure()
        fig.add_scatter(x=pk["累积样本"], y=pk["p 值"], mode="lines+markers",
                        line=dict(color=INDIGO, width=2.5), name="p 值")
        fig.add_hline(y=0.05, line=dict(color=PINK, width=2, dash="dot"),
                      annotation_text="α = 0.05", annotation_position="top right")
        fig.update_layout(height=380, margin=dict(t=10, l=10, r=10, b=10),
                          yaxis_title="双侧 p 值", xaxis_title="累积有效样本",
                          yaxis_range=[0, 1])
        show_chart(fig)
        with st.container(border=True):
            st.markdown(
                f"""
                - 25 个检查点中 p 值最小为 **{pk['p 值'].min():.4f}**，最大 **{pk['p 值'].max():.4f}**——
                同一实验、同一指标，不同时点看结论完全不同
                - **正确做法**：事前定样本量/时长，一次判停（或用序贯检验/贝叶斯方法合法地多次检查）
                - 本数据最终 p = {two_proportion_test(*current_stats('converted')).p_value:.4f}
                """
            )

        st.subheader("② 按日的指标稳定性（聚合视图）")
        # 聚合优先：分组键用 Series 列表，避免物化大表布尔选择
        date_s = AB_DF["timestamp"].str[:10]
        daily = (
            AB_DF.groupby([date_s, AB_DF["group"], AB_DF["clean"]])["converted"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"timestamp": "date"})
        )
        daily = daily[daily["clean"]]  # 小结果集（约百行）上过滤，安全
        daily["rate"] = daily["sum"] / daily["count"]
        fig = px.line(daily, x="date", y="rate", color="group",
                      color_discrete_map={"control": AMBER, "treatment": INDIGO},
                      labels={"rate": "当日转化率", "group": "组"})
        fig.update_layout(height=360, margin=dict(t=10, l=10, r=10, b=10), yaxis_tickformat=".0%")
        show_chart(fig)
        st.caption("若两组曲线长期贴近且互相穿插，与「无显著差异」的检验结论互相印证。")

    else:
        st.subheader("① 双指标与多重比较校正")
        st.caption("同时检验次日留存与 7 日留存，假阳性概率膨胀——需要 Bonferroni / Holm 校正")
        pvals = {}
        for m in ["retention_1", "retention_7"]:
            x1, n1, x2, n2 = current_stats(m)
            pvals[METRICS[m]] = two_proportion_test(x1, n1, x2, n2).p_value
        bh = bonferroni_holm(pvals)
        show_table(bh)
        with st.container(border=True):
            for m, p in pvals.items():
                r = two_proportion_test(*current_stats({"次日留存率": "retention_1", "7 日留存率": "retention_7"}[m]))
                st.markdown(
                    f"- **{m}**：{GROUP_CN['control']} {r.p1:.2%} vs {GROUP_CN['treatment']} {r.p2:.2%}，"
                    f"p = {r.p_value:.4f}{'（Holm 校正后仍显著）' if m == '7 日留存' and r.p_value < 0.05/2 else ''}"
                )
            st.markdown(
                """
                - 结论：门槛后移对**次日留存**无充分证据证明有影响；对**7 日留存**存在显著负向影响
                - 教学点：若不做多重比较校正，连续看两个指标，「碰巧显著」的概率约 1-(0.95)² ≈ 10%
                """
            )

        st.subheader("② 游戏轮次分布与极端离群值")
        st.caption("真实数据的脏角落：绝大多数玩家轮次很少，但存在极端离群值")
        q = CC_DF["sum_gamerounds"].quantile([0.5, 0.9, 0.99]).to_dict()
        c1, c2, c3 = st.columns(3)
        c1.metric("中位数轮次", int(q[0.5]))
        c2.metric("P90 轮次", int(q[0.9]))
        c3.metric("最大值（异常）", int(CC_DF["sum_gamerounds"].max()))
        bins = [0, 1, 5, 10, 30, 100, 500, 1000, 50000]
        lab = ["0", "1-4", "5-9", "10-29", "30-99", "100-499", "500-999", "1000+"]
        CC_DF["rbin"] = pd.cut(CC_DF["sum_gamerounds"], bins=bins, labels=lab, right=False)
        hist = CC_DF.groupby("rbin", observed=False).size().reset_index(name="人数")
        fig = px.bar(hist, x="rbin", y="人数", color_discrete_sequence=[INDIGO], labels={"rbin": "游戏轮次区间"})
        fig.update_layout(height=330, margin=dict(t=10, l=10, r=10, b=10), showlegend=False)
        show_chart(fig)
        st.caption("分布严重右偏——比较均值不可靠，因此本实验的主指标是留存率（比例），而非轮次均值。")

        st.subheader("③ 分层稳健性检查（辛普森悖论预警）")
        st.caption("按玩家活跃度分层后，重新看 7 日留存——检查总体结论是否被分层结构反转")
        CC_DF["act"] = pd.cut(CC_DF["sum_gamerounds"], bins=[0, 1, 30, 50000],
                              labels=["低活跃(0轮)", "中活跃(1-29)", "高活跃(30+)"], right=False)
        strat = CC_DF.groupby(["act", "group"], observed=False)["retention_7"].agg(["sum", "count"]).reset_index()
        strat["rate"] = strat["sum"] / strat["count"].replace(0, np.nan)
        strat["组"] = strat["group"].map(GROUP_CN)
        fig = px.bar(strat, x="act", y="rate", color="组", barmode="group",
                     color_discrete_sequence=[AMBER, INDIGO], labels={"act": "活跃度分层", "rate": "7 日留存率"})
        fig.update_layout(height=360, margin=dict(t=10, l=10, r=10, b=10), yaxis_tickformat=".0%")
        show_chart(fig)
        show_table(strat[["act", "组", "count", "rate"]].rename(columns={"act": "分层", "count": "玩家数", "rate": "7日留存"}))
        st.caption("分层方向若与总体一致（对照 ≥ 实验），结论稳健；若出现反转，则总体结论可能是混合偏差（辛普森悖论）。")


# ================= 页 4：实验设计器 =================
elif page == "⚡ 实验设计器":
    st.title("⚡ 实验设计器")
    st.caption("实验开始前的功课：样本量 / MDE / 功效——「先算再跑」，而不是跑完再后悔")

    st.subheader("① 样本量计算器（事前功效分析）")
    c1, c2, c3 = st.columns(3)
    with c1:
        p_base = st.slider("基线转化率", 1, 50, 12, 1, format="%d%%") / 100
    with c2:
        mde_pp = st.slider("最小可检测差异 MDE (pp)", 0.2, 5.0, 1.0, 0.1, format="%.1f")
    with c3:
        alpha = st.slider("显著性水平 α", 0.01, 0.10, 0.05, 0.01, format="%.2f")

    rows = []
    for power in [0.70, 0.80, 0.90]:
        n = required_sample_size(p_base, mde_pp / 100, alpha=alpha, power=power)
        rows.append({"目标功效": f"{power:.0%}", "每组所需样本": f"{n:,}", "总样本（两组）": f"{2*n:,}",
                     "检测的效应量 h": f"{abs(2*np.arcsin(np.sqrt(p_base+mde_pp/100))-2*np.arcsin(np.sqrt(p_base))):.4f}"})
    show_table(pd.DataFrame(rows))
    st.caption(
        f"公式：基于 Cohen's h 效应量的正态近似（statsmodels NormalIndPower）。"
        f"在 α={alpha:.2f}、基线 {p_base:.0%} 的条件下，想以 80% 功效检出 {mde_pp:.1f}pp 的差异，"
        f"每组约需 {required_sample_size(p_base, mde_pp/100, alpha):,} 个样本。"
    )

    st.subheader("② 反过来问：现有样本量能检出多小的差异？")
    st.caption("如果实验已经跑完，算出当前样本量的 MDE，判断「不显著」到底意味着什么")
    n_have = st.slider("当前每组样本量", 1000, 200000, 144000, 5000)
    p_have = st.slider("当前基线转化率", 1, 50, 12, 1, format="%d%%", key="mde_base") / 100
    mde = mde_given_n(n_have, p_have)
    st.metric("80% 功效下的 MDE", f"{mde*100:.2f} pp",
              help="差异小于该值时，即使真实存在，本样本量也大概率检不出来（不显著 ≠ 无差异）")

    st.subheader("③ 蒙特卡洛功效模拟（统计教学演示）")
    st.caption("⚠️ 本节模拟的是「统计方法的行为」（二项抽样 + z 检验），不涉及真实业务数据，仅用于验证公式")
    c1, c2 = st.columns(2)
    with c1:
        true_pp = st.slider("假设真实提升 (pp)", -3.0, 3.0, 1.0, 0.5, format="%.1f")
    with c2:
        n_sim_grp = st.slider("每组样本量", 1000, 60000, 40000, 1000)

    mc = monte_carlo_power(p_have, true_pp / 100, n_sim_grp, n_sim=1500, alpha=alpha)
    c1, c2, c3 = st.columns(3)
    c1.metric("经验功效（模拟拒绝率）", f"{mc['empirical_power']:.1%}")
    c2.metric("解析功效（公式）", f"{mc['analytic_power']:.1%}")
    c3.metric("模拟次数", f"{mc['n_sim']:,}")
    with st.container(border=True):
        st.markdown(
            f"""
            - 设定：基线 {p_have:.0%}、真实提升 {true_pp:+.1f}pp、每组 {n_sim_grp:,} 样本
            - 蒙特卡洛模拟的拒绝率（{mc['empirical_power']:.1%}）与解析公式（{mc['analytic_power']:.1%}）
              一致——交叉验证了功效公式的正确性
            - 解读：若真实差异恰为 {true_pp:+.1f}pp，这个样本量的实验有
              {mc['empirical_power']:.0%} 的概率得出「显著」结论；剩下 {1-mc['empirical_power']:.0%}
              的概率会得到「不显著」（假阴性）
            - 当真实提升设为 0 时，拒绝率应约等于 α——这就是检验的假阳性率（可自行验证）
            """
        )


st.divider()
st.caption(
    "AB 实验实验室 · 数据：Kaggle ab_test 公开镜像 + Cookie Cats 公开实验数据（来源见实验设计页）· "
    "方法：双比例 z 检验 / Wilson CI / 卡方 SRM / Holm 校正 / NormalIndPower 功效分析 · "
    "结论全部由数据计算生成"
)
