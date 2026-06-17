"""
垃圾分类综合评分模块单元测试
运行方式：python -m pytest test_scoring.py -v
"""
import pytest
import pandas as pd
import numpy as np
from scoring import (
    normalize_weights,
    min_max_normalize,
    calculate_subdistrict_scores,
    get_radar_chart_data,
)


np.random.seed(42)


def create_test_dataframe(n_rows=30, subdistricts=None):
    """创建测试用的模拟DataFrame"""
    if subdistricts is None:
        subdistricts = ['和平街道', '解放街道', '建设街道']
    
    dates = pd.date_range('2025-01-01', periods=n_rows, freq='D')
    records = []
    
    for date in dates:
        for sd in subdistricts:
            records.append({
                '街道': sd,
                '日期': date,
                '社区': f'{sd}测试社区',
                '分类准确率': np.random.uniform(0.6, 0.95),
                '参与率': np.random.uniform(0.4, 0.85),
                '混投总量(kg)': np.random.uniform(100, 500),
                '厨余垃圾量(kg)': np.random.uniform(500, 2000),
                '可回收物量(kg)': np.random.uniform(200, 800),
                '有害垃圾量(kg)': np.random.uniform(5, 30),
                '其他垃圾量(kg)': np.random.uniform(300, 1200),
            })
    
    return pd.DataFrame(records)


class TestNormalizeWeights:
    """权重归一化函数测试"""
    
    def test_all_weights_zero(self):
        """测试所有权重为0的情况 - 修复的核心bug"""
        wa, wp, wm, wr, all_zero = normalize_weights(0, 0, 0, 0)
        assert all_zero is True, "所有权重为0时应返回all_zero=True"
        assert wa == 0.0, "权重全0时归一化权重应为0"
        assert wp == 0.0, "权重全0时归一化权重应为0"
        assert wm == 0.0, "权重全0时归一化权重应为0"
        assert wr == 0.0, "权重全0时归一化权重应为0"
        # 关键断言：不能是之前偷偷恢复的默认值 (0.4, 0.25, 0.2, 0.15)
        assert wa != 0.4, "权重全0时不应恢复为默认值0.4"
        assert wp != 0.25, "权重全0时不应恢复为默认值0.25"
        assert wm != 0.2, "权重全0时不应恢复为默认值0.2"
        assert wr != 0.15, "权重全0时不应恢复为默认值0.15"
    
    def test_normal_weights(self):
        """测试正常权重 - 40/25/20/15"""
        wa, wp, wm, wr, all_zero = normalize_weights(40, 25, 20, 15)
        assert all_zero is False
        assert abs(wa - 0.40) < 1e-9
        assert abs(wp - 0.25) < 1e-9
        assert abs(wm - 0.20) < 1e-9
        assert abs(wr - 0.15) < 1e-9
        assert abs((wa + wp + wm + wr) - 1.0) < 1e-9
    
    def test_equal_weights(self):
        """测试等权重 - 25/25/25/25"""
        wa, wp, wm, wr, all_zero = normalize_weights(25, 25, 25, 25)
        assert all_zero is False
        assert abs(wa - 0.25) < 1e-9
        assert abs(wp - 0.25) < 1e-9
        assert abs(wm - 0.25) < 1e-9
        assert abs(wr - 0.25) < 1e-9
    
    def test_single_weight_nonzero(self):
        """测试单个权重非零 - 100/0/0/0"""
        wa, wp, wm, wr, all_zero = normalize_weights(100, 0, 0, 0)
        assert all_zero is False
        assert abs(wa - 1.0) < 1e-9
        assert wp == 0.0
        assert wm == 0.0
        assert wr == 0.0
    
    def test_three_weights_zero(self):
        """测试3个权重为0 - 0/0/50/0"""
        wa, wp, wm, wr, all_zero = normalize_weights(0, 0, 50, 0)
        assert all_zero is False
        assert wa == 0.0
        assert wp == 0.0
        assert abs(wm - 1.0) < 1e-9
        assert wr == 0.0
    
    def test_weights_not_normalized_values(self):
        """测试任意比例值 - 10/20/30/40 = 10%/20%/30%/40%"""
        wa, wp, wm, wr, all_zero = normalize_weights(10, 20, 30, 40)
        assert all_zero is False
        assert abs(wa - 0.10) < 1e-9
        assert abs(wp - 0.20) < 1e-9
        assert abs(wm - 0.30) < 1e-9
        assert abs(wr - 0.40) < 1e-9
    
    def test_sum_to_one_after_normalization(self):
        """测试任意权重归一化后总和为1"""
        for weights in [
            (1, 1, 1, 1),
            (3, 0, 7, 0),
            (100, 200, 300, 400),
            (99, 99, 99, 99),
        ]:
            wa, wp, wm, wr, all_zero = normalize_weights(*weights)
            assert all_zero is False
            assert abs((wa + wp + wm + wr) - 1.0) < 1e-9


class TestMinMaxNormalize:
    """Min-Max归一化函数测试"""
    
    def test_basic_forward_normalization(self):
        """测试正向归一化 [1, 2, 3, 4, 5] -> [0, 25, 50, 75, 100]"""
        values = pd.Series([1, 2, 3, 4, 5])
        result = min_max_normalize(values, reverse=False)
        assert len(result) == 5
        assert abs(result.iloc[0] - 0.0) < 1e-9
        assert abs(result.iloc[2] - 50.0) < 1e-9
        assert abs(result.iloc[4] - 100.0) < 1e-9
    
    def test_reverse_normalization(self):
        """测试反向归一化 [1, 2, 3, 4, 5] -> [100, 75, 50, 25, 0]"""
        values = pd.Series([1, 2, 3, 4, 5])
        result = min_max_normalize(values, reverse=True)
        assert abs(result.iloc[0] - 100.0) < 1e-9
        assert abs(result.iloc[2] - 50.0) < 1e-9
        assert abs(result.iloc[4] - 0.0) < 1e-9
    
    def test_all_same_values(self):
        """测试所有值相同的边界情况"""
        values = pd.Series([7, 7, 7, 7])
        result_forward = min_max_normalize(values, reverse=False)
        result_reverse = min_max_normalize(values, reverse=True)
        assert abs(result_forward.iloc[0] - 50.0) < 1e-9
        assert abs(result_reverse.iloc[0] - 50.0) < 1e-9
        for r in result_forward:
            assert abs(r - 50.0) < 1e-9
    
    def test_single_value(self):
        """测试单个值的边界情况"""
        values = pd.Series([42])
        result = min_max_normalize(values, reverse=False)
        assert abs(result.iloc[0] - 50.0) < 1e-9
    
    def test_two_values(self):
        """测试两个值"""
        values = pd.Series([10, 30])
        result = min_max_normalize(values, reverse=False)
        assert abs(result.iloc[0] - 0.0) < 1e-9
        assert abs(result.iloc[1] - 100.0) < 1e-9
    
    def test_negative_values(self):
        """测试负值（虽然业务上不太可能）"""
        values = pd.Series([-10, 0, 10])
        result = min_max_normalize(values, reverse=False)
        assert abs(result.iloc[0] - 0.0) < 1e-9
        assert abs(result.iloc[1] - 50.0) < 1e-9
        assert abs(result.iloc[2] - 100.0) < 1e-9
    
    def test_output_range_0_to_100(self):
        """测试输出始终在[0, 100]范围内"""
        values = pd.Series(np.random.randn(100) * 100)
        for reverse in [False, True]:
            result = min_max_normalize(values, reverse=reverse)
            assert result.min() >= -1e-9
            assert result.max() <= 100.0 + 1e-9


class TestCalculateSubdistrictScores:
    """街道综合评分计算函数测试"""
    
    def test_basic_calculation(self):
        """测试基本计算流程能正常运行"""
        df = create_test_dataframe(n_rows=10)
        result = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        
        assert isinstance(result, pd.DataFrame)
        assert '街道' in result.columns
        assert '分类准确率得分' in result.columns
        assert '参与率得分' in result.columns
        assert '混投控制得分' in result.columns
        assert '垃圾减量得分' in result.columns
        assert '综合得分' in result.columns
        assert '排名' in result.columns
    
    def test_zero_weights_all_scores_are_zero(self):
        """测试权重全0时，综合得分全部为0 - 核心bug修复验证"""
        df = create_test_dataframe(n_rows=10)
        result = calculate_subdistrict_scores(df, 0.0, 0.0, 0.0, 0.0)
        
        # 各维度得分可以不是0（归一化结果），但综合得分必须全部为0
        for idx, row in result.iterrows():
            expected_total = (
                row['分类准确率得分'] * 0.0 +
                row['参与率得分'] * 0.0 +
                row['混投控制得分'] * 0.0 +
                row['垃圾减量得分'] * 0.0
            )
            assert abs(row['综合得分'] - expected_total) < 1e-9
            assert abs(row['综合得分'] - 0.0) < 1e-9, f"综合得分应为0，但得到{row['综合得分']}"
    
    def test_score_columns_in_0_100_range(self):
        """测试各维度得分都在0-100范围内"""
        df = create_test_dataframe(n_rows=30)
        result = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        
        score_cols = ['分类准确率得分', '参与率得分', '混投控制得分', '垃圾减量得分']
        for col in score_cols:
            assert result[col].min() >= -1e-9, f"{col} 最小值 < 0"
            assert result[col].max() <= 100.0 + 1e-9, f"{col} 最大值 > 100"
    
    def test_total_score_bounded_by_weights(self):
        """测试综合得分范围：单个维度最高得100，受权重限制"""
        df = create_test_dataframe(n_rows=30)
        wa, wp, wm, wr = 0.4, 0.25, 0.2, 0.15
        result = calculate_subdistrict_scores(df, wa, wp, wm, wr)
        
        # 理论最高综合得分 = 100*wa + 100*wp + 100*wm + 100*wr = 100
        # 理论最低综合得分 = 0
        for score in result['综合得分']:
            assert score >= -1e-9, f"综合得分 {score} < 0"
            assert score <= 100.0 + 1e-9, f"综合得分 {score} > 100"
    
    def test_rank_starts_from_1_and_unique(self):
        """测试排名从1开始且不重复"""
        df = create_test_dataframe(n_rows=30)
        result = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        
        ranks = sorted(result['排名'].tolist())
        expected_ranks = list(range(1, len(result) + 1))
        assert ranks == expected_ranks, f"排名错误: {ranks} vs {expected_ranks}"
    
    def test_sorted_by_total_score_descending(self):
        """测试结果按综合得分降序排列"""
        df = create_test_dataframe(n_rows=30)
        result = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        
        scores = result['综合得分'].tolist()
        assert scores == sorted(scores, reverse=True), "结果未按综合得分降序排列"
    
    def test_single_weight_full_influence(self):
        """测试单个权重为1时，综合得分等于该维度得分"""
        df = create_test_dataframe(n_rows=30)
        # 只有分类准确率权重为1
        result = calculate_subdistrict_scores(df, 1.0, 0.0, 0.0, 0.0)
        for idx, row in result.iterrows():
            assert abs(row['综合得分'] - row['分类准确率得分']) < 1e-9
        
        # 只有参与率权重为1
        result2 = calculate_subdistrict_scores(df, 0.0, 1.0, 0.0, 0.0)
        for idx, row in result2.iterrows():
            assert abs(row['综合得分'] - row['参与率得分']) < 1e-9
    
    def test_output_columns(self):
        """测试输出包含所有期望的列"""
        df = create_test_dataframe(n_rows=5)
        result = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        
        expected_columns = [
            '街道', '分类准确率', '参与率', '混投率', '其他垃圾占比',
            '分类准确率得分', '参与率得分', '混投控制得分', '垃圾减量得分',
            '综合得分', '排名'
        ]
        for col in expected_columns:
            assert col in result.columns, f"缺少列: {col}"
    
    def test_mixed_control_reverse_scoring(self):
        """测试混投控制：混投率越低分越高（反向归一化）"""
        df = create_test_dataframe(n_rows=30)
        result = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        
        # 混投率与混投控制得分负相关
        corr = result['混投率'].corr(result['混投控制得分'])
        assert corr <= 0.0 + 1e-9, f"混投率与混投控制得分应为负相关，实际相关系数: {corr}"
    
    def test_reduction_reverse_scoring(self):
        """测试垃圾减量：其他垃圾占比越低分越高（反向归一化）"""
        df = create_test_dataframe(n_rows=30)
        result = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        
        corr = result['其他垃圾占比'].corr(result['垃圾减量得分'])
        assert corr <= 0.0 + 1e-9, f"其他垃圾占比与垃圾减量得分应为负相关，实际相关系数: {corr}"
    
    def test_accuracy_forward_scoring(self):
        """测试分类准确率：准确率越高分越高（正向归一化）"""
        df = create_test_dataframe(n_rows=30)
        result = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        
        corr = result['分类准确率'].corr(result['分类准确率得分'])
        assert corr >= 0.0 - 1e-9, f"分类准确率与得分应为正相关，实际相关系数: {corr}"


class TestGetRadarChartData:
    """雷达图数据生成函数测试"""
    
    def test_output_structure(self):
        """测试输出结构符合预期"""
        df = create_test_dataframe(n_rows=10, subdistricts=['A', 'B', 'C'])
        scores = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        result = get_radar_chart_data(scores)
        
        assert 'categories' in result
        assert 'subdistricts' in result
        assert len(result['categories']) == 4
        assert len(result['subdistricts']) == 3
    
    def test_categories_names(self):
        """测试维度名称正确"""
        df = create_test_dataframe(n_rows=10)
        scores = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        result = get_radar_chart_data(scores)
        
        assert result['categories'] == ['分类准确率', '参与率', '混投控制', '垃圾减量']
    
    def test_subdistrict_values_length(self):
        """测试每个街道的values数量与categories一致"""
        df = create_test_dataframe(n_rows=10)
        scores = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        result = get_radar_chart_data(scores)
        
        for sd in result['subdistricts']:
            assert 'name' in sd
            assert 'values' in sd
            assert len(sd['values']) == len(result['categories'])
    
    def test_zero_weights_radar_values_unchanged(self):
        """测试权重全0时，雷达图结构正常（维度得分独立于权重，排名可能不同）"""
        df = create_test_dataframe(n_rows=10, subdistricts=['A', 'B'])
        scores_zero_w = calculate_subdistrict_scores(df, 0.0, 0.0, 0.0, 0.0)
        scores_normal_w = calculate_subdistrict_scores(df, 0.4, 0.25, 0.2, 0.15)
        
        radar_zero = get_radar_chart_data(scores_zero_w)
        radar_normal = get_radar_chart_data(scores_normal_w)
        
        # 雷达图categories应该相同
        assert radar_zero['categories'] == radar_normal['categories']
        
        # 街道总数应该相同
        assert len(radar_zero['subdistricts']) == len(radar_normal['subdistricts'])
        
        # 每个街道都有name和values
        for sd_list in [radar_zero['subdistricts'], radar_normal['subdistricts']]:
            for sd in sd_list:
                assert 'name' in sd
                assert 'values' in sd
                assert len(sd['values']) == 4
                # 每个维度值都在0-100之间
                for v in sd['values']:
                    assert 0.0 - 1e-9 <= v <= 100.0 + 1e-9


class TestIntegrationEdgeCases:
    """综合边界情况测试"""
    
    def test_full_pipeline_zero_weights(self):
        """端到端测试：权重全0，综合得分全为0"""
        df = create_test_dataframe(n_rows=30, subdistricts=['和平', '解放', '建设'])
        
        wa, wp, wm, wr, all_zero = normalize_weights(0, 0, 0, 0)
        assert all_zero is True
        assert wa == 0.0 and wp == 0.0 and wm == 0.0 and wr == 0.0
        
        scores = calculate_subdistrict_scores(df, wa, wp, wm, wr)
        
        for _, row in scores.iterrows():
            assert abs(row['综合得分']) < 1e-9, (
                f"权重全0时 {row['街道']} 综合得分应为0，"
                f"但计算得到: {row['综合得分']}。\n"
                f"各维度得分: 准确率={row['分类准确率得分']:.2f}, "
                f"参与率={row['参与率得分']:.2f}, "
                f"混投控制={row['混投控制得分']:.2f}, "
                f"垃圾减量={row['垃圾减量得分']:.2f}"
            )
    
    def test_full_pipeline_normal_weights(self):
        """端到端测试：正常权重，综合得分合理"""
        df = create_test_dataframe(n_rows=30)
        
        wa, wp, wm, wr, all_zero = normalize_weights(40, 25, 20, 15)
        assert all_zero is False
        
        scores = calculate_subdistrict_scores(df, wa, wp, wm, wr)
        radar = get_radar_chart_data(scores)
        
        # 验证基本结构
        assert len(scores) == 3
        assert len(radar['subdistricts']) == 3
        
        # 验证综合得分范围
        assert scores['综合得分'].max() <= 100.0
        assert scores['综合得分'].min() >= 0.0
        
        # 最高综合得分对应排名1
        assert scores.iloc[0]['排名'] == 1
    
    def test_one_dimension_dominates(self):
        """测试极端权重：某维度100%权重，排名由该维度决定"""
        df = create_test_dataframe(n_rows=30, subdistricts=['A', 'B', 'C', 'D'])
        
        # 100%权重给分类准确率
        wa, wp, wm, wr, all_zero = normalize_weights(100, 0, 0, 0)
        scores = calculate_subdistrict_scores(df, wa, wp, wm, wr)
        
        # 排名顺序应与分类准确率得分顺序一致
        accuracy_ranking = scores.sort_values('分类准确率得分', ascending=False)['街道'].tolist()
        total_ranking = scores['街道'].tolist()
        assert accuracy_ranking == total_ranking, (
            "当分类准确率权重为100%时，综合排名应与分类准确率得分排名完全一致\n"
            f"综合排名: {total_ranking}\n"
            f"准确率排名: {accuracy_ranking}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
