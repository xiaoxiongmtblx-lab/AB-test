# AB 实验实验室 A/B Testing Lab

基于**两个真实公开 AB 实验数据集**的交互式统计分析应用（Streamlit），
完整演示从实验设计审查到统计决策的规范流程，配套单文件方法论讲解。

## 数据来源（全部公开，可自由迁移）

| 数据集 | 内容 | 规模 | 来源 |
|---|---|---|---|
| 营销落地页实验 | 新旧页面对购买转化的影响 | 294,478 行（2017-01） | Kaggle 经典 ab_test 数据（GitHub 公开镜像） |
| Cookie Cats 关卡实验 | 游戏门槛 30→40 关对留存的影响 | 90,189 名玩家 | Woopa 数据团队公开发布（DataCamp 经典案例，GitHub 镜像） |

数据已随项目打包在 `data/` 目录，**离线可运行**；运行结果可完全复现。

## 运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 功能（4 页）

| 页 | 内容 | 方法要点 |
|---|---|---|
| 🧪 实验设计与数据质量 | 数据集档案、清洗报告、SRM 检验 | 卡方拟合优度（α=0.001） |
| 📐 核心检验与结论 | 转化率对比、z 检验、Wilson CI、效应量、MDE、自动结论 | 双比例 z 检验 + Cohen's h |
| 🕵️ 深入分析 | peeking 轨迹 / 按日稳定性 / 双指标 Holm 校正 / 分层辛普森检查 | 多重比较、分层稳健性 |
| ⚡ 实验设计器 | 样本量计算器、MDE 反推、蒙特卡洛功效模拟 | NormalIndPower 事前功效分析 |

配套 **`讲解.html`**（浏览器直接打开）：假设检验框架、p 值正确读法、SRM、
peeking、多重比较、辛普森悖论、常见作死清单、两个实验的完整结论。

## 两个实验的结论（由数据计算生成，可在应用中复算）

1. **新落地页**：12.04% vs 11.87%，p=0.16，95%CI 跨 0——样本量支撑 MDE≈0.34pp，
   无充分证据支持新页面上线
2. **Cookie Cats**：7 日留存显著下降（p≈0.0016，Holm 校正后仍显著），次日留存无显著差异
   ——建议维持 30 关门槛

## 目录结构

```
ab-testing-lab/
├── app.py            # Streamlit 应用（4 页 + 双数据集切换）
├── 讲解.html          # 单文件方法论讲解（离线可读）
├── requirements.txt
├── data/
│   ├── ab_data.csv        # 营销 AB 实验（294,478 行）
│   └── cookie_cats.csv    # 游戏关卡实验（90,189 行）
└── src/
    └── engine.py     # 统计引擎（检验/CI/功效/多重比较/peeking/模拟，注释含方法论）
```

## 方法论红线（本项目严格遵守）

- ❌ 不编造任何数据与结论；结论文案由模板 + 计算数字生成
- ⚠️ 蒙特卡洛模拟仅用于统计方法教学（验证功效公式），与真实业务数据严格区分并标注
- ✅ 事后功效标注局限；决策依据是事前功效分析
- ✅ 清洗规则可审计（标志列 + 聚合统计，不物理删行）

## 推上 GitHub

```bash
git init
git add .
git commit -m "feat: AB 实验实验室 v1.0"
git branch -M main
git remote add origin https://github.com/<你的用户名>/ab-testing-lab.git
git push -u origin main
```

## License

MIT
