"""
垃圾分类综合评分模块
包含权重归一化、各维度打分、综合评分计算等可测试函数
"""
import pandas as pd
from typing import Dict, Tuple, List


def normalize_weights(
    w_accuracy: float,
    w_participation: float,
    w_mixed_control: float,
    w_reduction: float
) -> Tuple[float, float, float, float, bool]:
    """
    归一化权重
    
    Args:
        w_accuracy: 分类准确率权重（原始值，未归一化）
        w_participation: 参与率权重（原始值，未归一化）
        w_mixed_control: 混投控制权重（原始值，未归一化）
        w_reduction: 垃圾减量权重（原始值，未归一化）
    
    Returns:
        (归一化准确率权重, 归一化参与率权重, 归一化混投控制权重, 归一化垃圾减量权重, 权重是否全为0)
    """
    weight_sum = w_accuracy + w_participation + w_mixed_control + w_reduction
    
    if weight_sum == 0:
        return 0.0, 0.0, 0.0, 0.0, True
    
    return (
        w_accuracy / weight_sum,
        w_participation / weight_sum,
        w_mixed_control / weight_sum,
        w_reduction / weight_sum,
        False
    )


def min_max_normalize(
    values: pd.Series,
    reverse: bool = False
) -> pd.Series:
    """
    Min-Max归一化到 0-100 分
    
    Args:
        values: 待归一化的数值序列
        reverse: 是否反向归一化（值越小分越高，用于混投率、其他垃圾占比等负向指标）
    
    Returns:
        归一化后的分数序列（0-100）
    """
    vmin, vmax = values.min(), values.max()
    
    if vmax == vmin:
        return pd.Series([50.0] * len(values), index=values.index)
    
    if reverse:
        return (vmax - values) / (vmax - vmin) * 100
    else:
        return (values - vmin) / (vmax - vmin) * 100


def calculate_subdistrict_scores(
    df: pd.DataFrame,
    w_accuracy_norm: float,
    w_participation_norm: float,
    w_mixed_control_norm: float,
    w_reduction_norm: float
) -> pd.DataFrame:
    """
    计算各街道的综合评分
    
    Args:
        df: 筛选后的垃圾分类数据，需包含列：
            街道, 分类准确率, 参与率, 混投总量(kg), 
            厨余垃圾量(kg), 可回收物量(kg), 有害垃圾量(kg), 其他垃圾量(kg)
        w_accuracy_norm: 归一化后的分类准确率权重
        w_participation_norm: 归一化后的参与率权重
        w_mixed_control_norm: 归一化后的混投控制权重
        w_reduction_norm: 归一化后的垃圾减量权重
    
    Returns:
        各街道评分DataFrame，包含列：
        街道, 分类准确率, 参与率, 混投率, 其他垃圾占比,
        分类准确率得分, 参与率得分, 混投控制得分, 垃圾减量得分,
        综合得分, 排名
    """
    sub_score = df.groupby('街道').agg({
        '分类准确率': 'mean',
        '参与率': 'mean',
        '混投总量(kg)': 'sum',
        '厨余垃圾量(kg)': 'sum',
        '可回收物量(kg)': 'sum',
        '有害垃圾量(kg)': 'sum',
        '其他垃圾量(kg)': 'sum'
    }).reset_index()
    
    sub_score['总垃圾量(kg)'] = (
        sub_score['厨余垃圾量(kg)'] + 
        sub_score['可回收物量(kg)'] + 
        sub_score['有害垃圾量(kg)'] + 
        sub_score['其他垃圾量(kg)']
    )
    sub_score['混投率'] = sub_score['混投总量(kg)'] / sub_score['总垃圾量(kg)']
    sub_score['其他垃圾占比'] = sub_score['其他垃圾量(kg)'] / sub_score['总垃圾量(kg)']
    
    sub_score['分类准确率得分'] = min_max_normalize(sub_score['分类准确率'], reverse=False)
    sub_score['参与率得分'] = min_max_normalize(sub_score['参与率'], reverse=False)
    sub_score['混投控制得分'] = min_max_normalize(sub_score['混投率'], reverse=True)
    sub_score['垃圾减量得分'] = min_max_normalize(sub_score['其他垃圾占比'], reverse=True)
    
    sub_score['综合得分'] = (
        sub_score['分类准确率得分'] * w_accuracy_norm +
        sub_score['参与率得分'] * w_participation_norm +
        sub_score['混投控制得分'] * w_mixed_control_norm +
        sub_score['垃圾减量得分'] * w_reduction_norm
    )
    
    sub_score = sub_score.sort_values('综合得分', ascending=False).reset_index(drop=True)
    sub_score['排名'] = range(1, len(sub_score) + 1)
    
    return sub_score


def get_radar_chart_data(score_df: pd.DataFrame) -> Dict[str, List]:
    """
    生成雷达图所需的数据格式
    
    Args:
        score_df: calculate_subdistrict_scores 的输出结果
    
    Returns:
        {'categories': 维度列表, 'subdistricts': [{name, values}] }
    """
    categories = ['分类准确率', '参与率', '混投控制', '垃圾减量']
    subdistricts = []
    
    for _, row in score_df.iterrows():
        values = [
            row['分类准确率得分'],
            row['参与率得分'],
            row['混投控制得分'],
            row['垃圾减量得分']
        ]
        subdistricts.append({
            'name': row['街道'],
            'values': values
        })
    
    return {
        'categories': categories,
        'subdistricts': subdistricts
    }
