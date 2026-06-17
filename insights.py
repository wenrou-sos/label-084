"""
垃圾分类数据自然语言解读生成模块
根据聚合数据自动生成图表下方的文字解读
"""
import pandas as pd
from typing import Dict


def _format_pct(value: float) -> str:
    """格式化百分比为 1 位小数"""
    return f"{value:.1f}%"


def _format_kg(value: float) -> str:
    """格式化重量，大于1000kg自动转吨"""
    if value >= 1000:
        return f"{value/1000:,.1f} 吨"
    return f"{value:,.0f} kg"


def _format_change(delta: float, unit: str = "个百分点") -> str:
    """格式化变化量，带正负号"""
    sign = "↑" if delta >= 0 else "↓"
    abs_delta = abs(delta)
    return f"{sign} {abs_delta:.1f} {unit}"


# ============================================================
# 街道排名解读
# ============================================================

def generate_ranking_insights(
    subdistrict_accuracy: pd.DataFrame,
    subdistrict_participation: pd.DataFrame,
    waste_totals: pd.DataFrame
) -> Dict[str, str]:
    """
    生成街道排名相关的文字解读

    Args:
        subdistrict_accuracy: 各街道准确率数据，需包含 街道、分类准确率(百分比值)
        subdistrict_participation: 各街道参与率数据，需包含 街道、参与率(百分比值)
        waste_totals: 各类垃圾总量数据

    Returns:
        {
            'accuracy': 准确率排名解读,
            'participation': 参与率排名解读,
            'waste_distribution': 垃圾分布解读
        }
    """
    insights = {}

    # --- 准确率排名解读 ---
    acc_sorted = subdistrict_accuracy.sort_values('分类准确率', ascending=False).reset_index(drop=True)
    top_acc = acc_sorted.iloc[0]
    bottom_acc = acc_sorted.iloc[-1]
    acc_gap = top_acc['分类准确率'] - bottom_acc['分类准确率']

    if top_acc['分类准确率'] >= 85:
        acc_desc = "表现优秀"
    elif top_acc['分类准确率'] >= 75:
        acc_desc = "表现良好"
    else:
        acc_desc = "有待提升"

    insights['accuracy'] = (
        f"📊 **准确率解读：**"
        f"{top_acc['街道']}{acc_desc}，准确率达 {_format_pct(top_acc['分类准确率'])}，"
        f"排名第一；"
        f"{bottom_acc['街道']}准确率相对落后，为 {_format_pct(bottom_acc['分类准确率'])}；"
        f"最高与最低相差 {acc_gap:.1f} 个百分点。"
    )

    # --- 参与率排名解读 ---
    par_sorted = subdistrict_participation.sort_values('参与率', ascending=False).reset_index(drop=True)
    top_par = par_sorted.iloc[0]
    bottom_par = par_sorted.iloc[-1]
    par_gap = top_par['参与率'] - bottom_par['参与率']
    avg_par = par_sorted['参与率'].mean()

    if avg_par >= 70:
        par_desc = "居民参与度较高"
    elif avg_par >= 60:
        par_desc = "居民参与度中等"
    else:
        par_desc = "居民参与度有待提升"

    insights['participation'] = (
        f"👥 **参与率解读：**"
        f"{top_par['街道']}参与率最高，达 {_format_pct(top_par['参与率'])}；"
        f"整体{par_desc}，平均参与率 {_format_pct(avg_par)}；"
        f"各街道间差距 {par_gap:.1f} 个百分点。"
    )

    # --- 垃圾分布解读 ---
    total_waste = waste_totals['重量(kg)'].sum()
    kitchen_pct = waste_totals.loc[waste_totals['垃圾类型'] == '厨余垃圾', '重量(kg)'].values[0] / total_waste * 100
    recyclable_pct = waste_totals.loc[waste_totals['垃圾类型'] == '可回收物', '重量(kg)'].values[0] / total_waste * 100
    other_pct = waste_totals.loc[waste_totals['垃圾类型'] == '其他垃圾', '重量(kg)'].values[0] / total_waste * 100

    insights['waste_distribution'] = (
        f"🗑️ **垃圾构成解读：**"
        f"期间总垃圾量约 {_format_kg(total_waste)}，"
        f"其中厨余垃圾占比最大（{_format_pct(kitchen_pct)}），"
        f"可回收物占 {_format_pct(recyclable_pct)}，"
        f"其他垃圾占 {_format_pct(other_pct)}。"
    )

    return insights


# ============================================================
# 月度趋势解读
# ============================================================

def generate_trend_insights(monthly_trend: pd.DataFrame) -> Dict[str, str]:
    """
    生成月度趋势相关的文字解读

    Args:
        monthly_trend: 月度趋势数据，需包含 年月、分类准确率(%)、参与率(%)、混投总量(kg)

    Returns:
        {
            'accuracy_trend': 准确率趋势解读,
            'recent_change': 环比变化解读,
            'mixed_trend': 混投量趋势解读
        }
    """
    insights = {}

    if len(monthly_trend) < 2:
        insights['accuracy_trend'] = "📈 **趋势解读：** 数据点不足，无法分析趋势。"
        insights['recent_change'] = "📊 **环比变化：** 数据不足。"
        insights['mixed_trend'] = "📉 数据不足。"
        return insights

    first_acc = monthly_trend['分类准确率'].iloc[0]
    last_acc = monthly_trend['分类准确率'].iloc[-1]
    acc_total_change = last_acc - first_acc

    is_upward = acc_total_change > 0
    is_steady = abs(acc_total_change) < 2

    if is_steady:
        trend_desc = "整体保持平稳"
    elif is_upward:
        trend_desc = "整体呈上升趋势"
    else:
        trend_desc = "整体呈下降趋势"

    insights['accuracy_trend'] = (
        f"📈 **准确率趋势：**"
        f"从 {monthly_trend['年月'].iloc[0]} 到 {monthly_trend['年月'].iloc[-1]}，"
        f"分类准确率{trend_desc}，"
        f"由 {_format_pct(first_acc)} {_format_change(acc_total_change, '个百分点')} 至 {_format_pct(last_acc)}。"
    )

    # --- 近三个月环比变化 ---
    if len(monthly_trend) >= 3:
        last_month_acc = monthly_trend['分类准确率'].iloc[-1]
        prev_month_acc = monthly_trend['分类准确率'].iloc[-2]
        mom_change = last_month_acc - prev_month_acc

        insights['recent_change'] = (
            f"📊 **近期变化：**"
            f"{monthly_trend['年月'].iloc[-1]} 分类准确率为 {_format_pct(last_month_acc)}，"
            f"较上月{_format_change(mom_change, '个百分点')}。"
        )
    else:
        insights['recent_change'] = "📊 **近期变化：** 数据不足，无法计算环比。"

    # --- 混投量趋势 ---
    first_mixed = monthly_trend['混投总量(kg)'].iloc[0]
    last_mixed = monthly_trend['混投总量(kg)'].iloc[-1]
    mixed_change_pct = (last_mixed - first_mixed) / first_mixed * 100 if first_mixed > 0 else 0

    if mixed_change_pct < -5:
        mixed_desc = "明显下降，分类规范性持续改善"
    elif mixed_change_pct > 5:
        mixed_desc = "有所上升，需加强管理"
    else:
        mixed_desc = "基本持平"

    insights['mixed_trend'] = (
        f"🚫 **混投量趋势：**"
        f"期间混投量{mixed_desc}，"
        f"{_format_change(mixed_change_pct, '%')}。"
    )

    return insights


# ============================================================
# 混投高发小区解读
# ============================================================

def generate_high_risk_insights(
    community_stats: pd.DataFrame,
    mixed_threshold: float
) -> Dict[str, str]:
    """
    生成混投高发小区的文字解读

    Args:
        community_stats: 各社区统计数据，含 街道、社区、分类准确率(%)、混投总量(kg)
        mixed_threshold: 高风险阈值（百分比）

    Returns:
        {
            'high_risk_count': 高风险数量解读,
            'worst_communities': 最差社区解读,
            'subdistrict_distribution': 街道分布解读
        }
    """
    insights = {}

    high_risk = community_stats[community_stats['分类准确率'] < mixed_threshold]
    high_risk_count = len(high_risk)
    total_count = len(community_stats)
    high_risk_ratio = high_risk_count / total_count * 100

    if high_risk_count == 0:
        risk_level = "情况良好，无高风险小区"
    elif high_risk_ratio < 20:
        risk_level = "整体可控"
    elif high_risk_ratio < 50:
        risk_level = "问题较多"
    else:
        risk_level = "形势严峻"

    insights['high_risk_count'] = (
        f"⚠️ **高风险小区：**"
        f"共 {high_risk_count} 个小区分类准确率低于 {mixed_threshold:.0f}%（占总数 {_format_pct(high_risk_ratio)}），"
        f"{risk_level}。"
    )

    # --- 最差社区 Top3 ---
    worst3 = community_stats.sort_values('分类准确率').head(3)
    worst_names = "、".join(worst3['社区'].tolist())

    insights['worst_communities'] = (
        f"🏚️ **问题最突出的社区：**"
        f"{worst_names} 等社区分类准确率最低，"
        f"仅 {_format_pct(worst3['分类准确率'].iloc[0])}。"
    )

    # --- 街道分布 ---
    if high_risk_count > 0:
        by_sub = high_risk.groupby('街道').size().sort_values(ascending=False)
        worst_sub = by_sub.index[0]
        worst_sub_count = by_sub.iloc[0]
        insights['subdistrict_distribution'] = (
            f"📍 **分布特点：**"
            f"高风险小区主要集中在{worst_sub}（{worst_sub_count} 个），"
            f"建议重点加强该区域的宣传督导。"
        )
    else:
        insights['subdistrict_distribution'] = (
            "🎉 **分布特点：** 所有社区均达标，继续保持！"
        )

    return insights


# ============================================================
# 混投品类分析解读
# ============================================================

def generate_mixed_category_insights(
    kitchen_mixed: float,
    plastic_in_kitchen: float,
    paper_in_kitchen: float,
    glass_in_kitchen: float,
    other_mixed: float
) -> Dict[str, str]:
    """
    生成混投品类分析的文字解读

    Args:
        kitchen_mixed: 厨余袋中混投总量
        plastic_in_kitchen: 厨余袋中塑料瓶混投量
        paper_in_kitchen: 厨余袋中纸张混投量
        glass_in_kitchen: 厨余袋中玻璃混投量
        other_mixed: 其他混投量

    Returns:
        {
            'main_category': 主要混投品类解读,
            'kitchen_mixed_ratio': 厨余袋混投解读,
            'improvement_suggestion': 改进建议
        }
    """
    insights = {}

    total_mixed = kitchen_mixed + other_mixed
    kitchen_ratio = kitchen_mixed / total_mixed * 100 if total_mixed > 0 else 0

    # 找出厨余袋内占比最高的品类
    other_in_kitchen = kitchen_mixed - plastic_in_kitchen - paper_in_kitchen - glass_in_kitchen
    categories = {
        '塑料瓶': plastic_in_kitchen,
        '纸张': paper_in_kitchen,
        '玻璃': glass_in_kitchen,
        '其他': other_in_kitchen
    }
    top_category = max(categories, key=categories.get)
    top_pct = categories[top_category] / kitchen_mixed * 100 if kitchen_mixed > 0 else 0

    insights['main_category'] = (
        f"🥡 **主要混投品类：**"
        f"厨余袋内混投中，"
        f"**{top_category}**占比最高，达 {_format_pct(top_pct)}，"
        f"是最常见的混投物品。"
    )

    insights['kitchen_mixed_ratio'] = (
        f"🍳 **厨余袋混投占比：**"
        f"厨余袋内混投约占总混投量的 {_format_pct(kitchen_ratio)}，"
        f"总量 {_format_kg(kitchen_mixed)}。"
    )

    # 根据主要混投品类对应的改进建议
    suggestions = {
        '塑料瓶': "建议在投放前将塑料瓶沥干后单独投放至可回收物桶，或在厨余桶旁增设塑料回收装置。",
        '纸张': "建议加强宣传纸张单独收集，投放到可回收物桶，避免混入厨余。",
        '玻璃': "建议玻璃制品单独投放至可回收物，注意避免破碎。",
        '其他': "建议加强分类知识宣传，提升居民分类意识。"
    }

    insights['improvement_suggestion'] = (
        f"💡 **改进建议：**"
        f"针对{top_category}混投问题，"
        f"{suggestions.get(top_category, suggestions['其他'])}"
    )

    return insights


# ============================================================
# 综合评分解读
# ============================================================

def generate_score_insights(
    score_df: pd.DataFrame,
    weights_all_zero: bool,
    w_accuracy_norm: float,
    w_participation_norm: float,
    w_mixed_control_norm: float,
    w_reduction_norm: float
) -> Dict[str, str]:
    """
    生成综合评分的文字解读

    Args:
        score_df: 评分结果DataFrame，含 街道、综合得分、排名、各维度得分
        weights_all_zero: 权重是否全为0
        w_accuracy_norm: 归一化准确率权重
        w_participation_norm: 归一化参与率权重
        w_mixed_control_norm: 归一化混投控制权重
        w_reduction_norm: 归一化垃圾减量权重

    Returns:
        {
            'champion': 冠军街道解读,
            'dimension_analysis': 维度分析解读,
            'overall_assessment': 整体评估
        }
    """
    insights = {}

    if weights_all_zero:
        insights['champion'] = "⚠️ **注意：** 所有评分权重均为0，综合得分全部为0分。请设置权重后查看排名。"
        insights['dimension_analysis'] = "请至少为一个维度设置大于0的权重。"
        insights['overall_assessment'] = ""
        return insights

    champion = score_df.iloc[0]
    last_place = score_df.iloc[-1]
    score_gap = champion['综合得分'] - last_place['综合得分']

    # 冠军分析
    champion_strengths = []
    if champion['分类准确率得分'] >= 80:
        champion_strengths.append("分类准确率高")
    if champion['参与率得分'] >= 80:
        champion_strengths.append("居民参与积极")
    if champion['混投控制得分'] >= 80:
        champion_strengths.append("混投控制好")
    if champion['垃圾减量得分'] >= 80:
        champion_strengths.append("减量效果好")

    strength_desc = "、".join(champion_strengths) if champion_strengths else "综合实力均衡"

    insights['champion'] = (
        f"🏆 **冠军街道：**"
        f"**{champion['街道']}** 以 {champion['综合得分']:.1f} 分位列第一，"
        f"主要优势是{strength_desc}。"
    )

    # 找出表现最强和最弱的维度
    dim_scores = {
        '分类准确率': score_df['分类准确率得分'].mean(),
        '参与率': score_df['参与率得分'].mean(),
        '混投控制': score_df['混投控制得分'].mean(),
        '垃圾减量': score_df['垃圾减量得分'].mean()
    }
    strongest_dim = max(dim_scores, key=dim_scores.get)
    weakest_dim = min(dim_scores, key=dim_scores.get)

    insights['dimension_analysis'] = (
        f"📊 **维度分析：**"
        f"整体来看，"
        f"**{strongest_dim}**表现最好（平均 {dim_scores[strongest_dim]:.1f} 分），"
        f"**{weakest_dim}**相对薄弱（平均 {dim_scores[weakest_dim]:.1f} 分），"
        f"是重点提升方向。"
    )

    # 整体评估
    avg_score = score_df['综合得分'].mean()

    if avg_score >= 80:
        overall_desc = "整体表现优秀"
    elif avg_score >= 60:
        overall_desc = "整体表现良好"
    elif avg_score >= 40:
        overall_desc = "整体表现中等"
    else:
        overall_desc = "整体有待提升"

    gap_desc = "差距较大" if score_gap > 30 else "差距不大"

    insights['overall_assessment'] = (
        f"📈 **整体评估：**"
        f"{overall_desc}，"
        f"平均综合得分 {avg_score:.1f} 分，"
        f"首尾相差 {score_gap:.1f} 分，"
        f"各街道间发展{gap_desc}。"
    )

    return insights
