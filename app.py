import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scoring import (
    normalize_weights,
    calculate_subdistrict_scores,
    get_radar_chart_data,
    min_max_normalize
)

st.set_page_config(
    page_title="城市垃圾分类成效分析看板",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #a8e063, #56ab2f);
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .high-risk {
        background-color: #ffe0e0 !important;
        color: #c0392b !important;
        font-weight: bold;
    }
    .stDataFrame {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('data/waste_data.csv', encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    return df

df = load_data()

st.markdown('<div class="main-header">🗑️ 城市垃圾分类成效分析看板</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🔍 筛选条件")
    
    date_min = df['日期'].min().date()
    date_max = df['日期'].max().date()
    
    start_date = st.date_input("开始日期", date_min, min_value=date_min, max_value=date_max)
    end_date = st.date_input("结束日期", date_max, min_value=date_min, max_value=date_max)
    
    selected_subdistricts = st.multiselect(
        "选择街道",
        options=sorted(df['街道'].unique()),
        default=sorted(df['街道'].unique())
    )
    
    mixed_threshold = st.slider(
        "混投高风险阈值(准确率%)",
        min_value=60,
        max_value=85,
        value=70,
        help="低于该准确率的小区将被标记为高风险"
    )
    
    st.markdown("---")
    st.header("⚖️ 综合评分权重")
    w_accuracy = st.slider("分类准确率权重", 0, 100, 40, help="分类准确率在综合评分中的权重占比")
    w_participation = st.slider("参与率权重", 0, 100, 25, help="居民参与率在综合评分中的权重占比")
    w_mixed_control = st.slider("混投控制权重", 0, 100, 20, help="混投控制表现（混投率越低分越高）在综合评分中的权重占比")
    w_reduction = st.slider("垃圾减量权重", 0, 100, 15, help="其他垃圾占比越低分越高，在综合评分中的权重占比")
    
    w_accuracy_norm, w_participation_norm, w_mixed_control_norm, w_reduction_norm, weights_all_zero = normalize_weights(
        w_accuracy, w_participation, w_mixed_control, w_reduction
    )

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_df = df[
    (df['日期'] >= start_date) & 
    (df['日期'] <= end_date) & 
    (df['街道'].isin(selected_subdistricts))
]

if len(filtered_df) == 0:
    st.warning("⚠️ 当前筛选条件下无数据，请调整日期范围或选择更多街道后重试。")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_kitchen = filtered_df['厨余垃圾量(kg)'].sum()
    st.markdown(f"""
    <div class="metric-card metric-card-green">
        <div style="font-size: 0.9rem; opacity: 0.9;">🍳 厨余垃圾总量</div>
        <div style="font-size: 1.8rem; font-weight: bold;">{total_kitchen:,.0f} kg</div>
        <div style="font-size: 0.8rem; opacity: 0.8;">{total_kitchen/1000:,.1f} 吨</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_recyclable = filtered_df['可回收物量(kg)'].sum()
    st.markdown(f"""
    <div class="metric-card metric-card-blue">
        <div style="font-size: 0.9rem; opacity: 0.9;">♻️ 可回收物总量</div>
        <div style="font-size: 1.8rem; font-weight: bold;">{total_recyclable:,.0f} kg</div>
        <div style="font-size: 0.8rem; opacity: 0.8;">{total_recyclable/1000:,.1f} 吨</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_accuracy = filtered_df['分类准确率'].mean() * 100
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; opacity: 0.9;">🎯 平均分类准确率</div>
        <div style="font-size: 1.8rem; font-weight: bold;">{avg_accuracy:.1f}%</div>
        <div style="font-size: 0.8rem; opacity: 0.8;">
            {'↑ 上升趋势' if avg_accuracy > 75 else '↓ 需要改进'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_participation = filtered_df['参与率'].mean() * 100
    st.markdown(f"""
    <div class="metric-card metric-card-orange">
        <div style="font-size: 0.9rem; opacity: 0.9;">👥 居民参与率</div>
        <div style="font-size: 1.8rem; font-weight: bold;">{avg_participation:.1f}%</div>
        <div style="font-size: 0.8rem; opacity: 0.8;">
            {filtered_df['参与户数'].sum():,.0f} 户参与
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 街道成效排名", "📈 月度趋势分析", "🏘️ 混投高发小区", "🥡 混投品类分析", "🏆 综合评分"])

with tab1:
    st.subheader("📊 各街道分类成效排名")
    
    col_bar1, col_bar2 = st.columns([1, 1])
    
    with col_bar1:
        st.markdown("#### 按分类准确率排名")
        subdistrict_accuracy = filtered_df.groupby('街道').agg({
            '分类准确率': 'mean',
            '厨余垃圾量(kg)': 'sum',
            '可回收物量(kg)': 'sum',
            '有害垃圾量(kg)': 'sum',
            '其他垃圾量(kg)': 'sum'
        }).reset_index()
        subdistrict_accuracy['分类准确率'] = subdistrict_accuracy['分类准确率'] * 100
        subdistrict_accuracy = subdistrict_accuracy.sort_values('分类准确率', ascending=True)
        
        colors = ['#2ecc71' if x >= 80 else '#f39c12' if x >= 70 else '#e74c3c' for x in subdistrict_accuracy['分类准确率']]
        
        fig1 = go.Figure(go.Bar(
            x=subdistrict_accuracy['分类准确率'],
            y=subdistrict_accuracy['街道'],
            orientation='h',
            marker=dict(color=colors),
            text=subdistrict_accuracy['分类准确率'].round(1).astype(str) + '%',
            textposition='auto',
        ))
        fig1.update_layout(
            title='各街道分类准确率排名',
            xaxis_title='分类准确率 (%)',
            yaxis_title='街道',
            height=500,
            xaxis=dict(range=[60, 95]),
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_bar2:
        st.markdown("#### 按参与率排名")
        subdistrict_participation = filtered_df.groupby('街道').agg({
            '参与率': 'mean',
            '参与户数': 'sum',
            '总户数': 'sum'
        }).reset_index()
        subdistrict_participation['参与率'] = subdistrict_participation['参与率'] * 100
        subdistrict_participation = subdistrict_participation.sort_values('参与率', ascending=True)
        
        colors2 = ['#3498db' if x >= 70 else '#9b59b6' if x >= 60 else '#e67e22' for x in subdistrict_participation['参与率']]
        
        fig2 = go.Figure(go.Bar(
            x=subdistrict_participation['参与率'],
            y=subdistrict_participation['街道'],
            orientation='h',
            marker=dict(color=colors2),
            text=subdistrict_participation['参与率'].round(1).astype(str) + '%',
            textposition='auto',
        ))
        fig2.update_layout(
            title='各街道居民参与率排名',
            xaxis_title='参与率 (%)',
            yaxis_title='街道',
            height=500,
            xaxis=dict(range=[40, 90]),
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("#### 各类垃圾总量分布")
    waste_totals = pd.DataFrame({
        '垃圾类型': ['厨余垃圾', '可回收物', '有害垃圾', '其他垃圾'],
        '重量(kg)': [
            filtered_df['厨余垃圾量(kg)'].sum(),
            filtered_df['可回收物量(kg)'].sum(),
            filtered_df['有害垃圾量(kg)'].sum(),
            filtered_df['其他垃圾量(kg)'].sum()
        ]
    })
    colors_pie = ['#27ae60', '#3498db', '#e74c3c', '#95a5a6']
    
    fig_pie = px.pie(
        waste_totals,
        values='重量(kg)',
        names='垃圾类型',
        color_discrete_sequence=colors_pie,
        title='各类垃圾总量占比',
        hole=0.4
    )
    fig_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("📈 月度趋势分析")
    
    monthly_trend = filtered_df.assign(年月=filtered_df['日期'].dt.to_period('M').astype(str))
    monthly_trend = monthly_trend.groupby('年月').agg({
        '分类准确率': 'mean',
        '参与率': 'mean',
        '厨余垃圾量(kg)': 'sum',
        '可回收物量(kg)': 'sum',
        '混投总量(kg)': 'sum'
    }).reset_index()
    monthly_trend['分类准确率'] = monthly_trend['分类准确率'] * 100
    monthly_trend['参与率'] = monthly_trend['参与率'] * 100
    
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig3.add_trace(
        go.Scatter(
            x=monthly_trend['年月'],
            y=monthly_trend['分类准确率'],
            name='分类准确率',
            line=dict(color='#27ae60', width=3),
            mode='lines+markers',
            marker=dict(size=8)
        ),
        secondary_y=False,
    )
    
    fig3.add_trace(
        go.Scatter(
            x=monthly_trend['年月'],
            y=monthly_trend['参与率'],
            name='参与率',
            line=dict(color='#3498db', width=3, dash='dash'),
            mode='lines+markers',
            marker=dict(size=8)
        ),
        secondary_y=True,
    )
    
    fig3.update_layout(
        title='分类准确率与参与率月度趋势',
        xaxis_title='月份',
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.1)
    )
    fig3.update_yaxes(title_text='分类准确率 (%)', secondary_y=False, range=[60, 95])
    fig3.update_yaxes(title_text='参与率 (%)', secondary_y=True, range=[40, 90])
    
    st.plotly_chart(fig3, use_container_width=True)
    
    col_trend1, col_trend2 = st.columns(2)
    
    with col_trend1:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=monthly_trend['年月'],
            y=monthly_trend['厨余垃圾量(kg)'],
            name='厨余垃圾',
            marker_color='#27ae60'
        ))
        fig4.add_trace(go.Bar(
            x=monthly_trend['年月'],
            y=monthly_trend['可回收物量(kg)'],
            name='可回收物',
            marker_color='#3498db'
        ))
        fig4.update_layout(
            title='每月分类垃圾分出量',
            barmode='group',
            yaxis_title='重量 (kg)',
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with col_trend2:
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=monthly_trend['年月'],
            y=monthly_trend['混投总量(kg)'],
            name='混投总量',
            marker_color='#e74c3c'
        ))
        fig5.update_layout(
            title='每月混投总量趋势',
            yaxis_title='混投重量 (kg)',
            height=400
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("#### 趋势分析总结")
    if len(monthly_trend) >= 2:
        first_accuracy = monthly_trend['分类准确率'].iloc[0]
        last_accuracy = monthly_trend['分类准确率'].iloc[-1]
        accuracy_change = last_accuracy - first_accuracy
    elif len(monthly_trend) == 1:
        first_accuracy = monthly_trend['分类准确率'].iloc[0]
        last_accuracy = first_accuracy
        accuracy_change = 0
    else:
        first_accuracy = 0
        last_accuracy = 0
        accuracy_change = 0
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric(
            "准确率变化",
            f"{last_accuracy:.1f}%",
            f"{accuracy_change:+.1f}%",
            delta_color="normal"
        )
    with col_sum2:
        total_waste = monthly_trend['厨余垃圾量(kg)'].sum() + monthly_trend['可回收物量(kg)'].sum()
        st.metric(
            "累计分出量",
            f"{total_waste/1000:,.1f} 吨"
        )
    with col_sum3:
        avg_mixed = monthly_trend['混投总量(kg)'].mean()
        st.metric(
            "月均混投量",
            f"{avg_mixed:,.0f} kg"
        )

with tab3:
    st.subheader("🏘️ 混投高发小区预警")
    st.info(f"🔴 准确率低于 {mixed_threshold}% 的小区将被标红警告")
    
    community_stats = filtered_df.groupby(['街道', '社区']).agg({
        '分类准确率': 'mean',
        '参与率': 'mean',
        '混投总量(kg)': 'sum',
        '厨余袋中混投量(kg)': 'sum',
        '参与户数': 'sum'
    }).reset_index()
    community_stats['分类准确率'] = community_stats['分类准确率'] * 100
    community_stats['参与率'] = community_stats['参与率'] * 100
    community_stats = community_stats.sort_values('分类准确率', ascending=True)
    
    high_risk_mask = community_stats['分类准确率'] < mixed_threshold
    high_risk_count = high_risk_mask.sum()
    
    st.warning(f"⚠️ 共发现 {high_risk_count} 个混投高发小区")
    
    def highlight_high_risk(s):
        return ['background-color: #ffe0e0; color: #c0392b; font-weight: bold' if v < mixed_threshold else '' for v in s]
    
    display_cols = ['街道', '社区', '分类准确率', '参与率', '混投总量(kg)', '厨余袋中混投量(kg)']
    display_df = community_stats[display_cols].round(2)
    display_df.columns = ['街道', '社区', '分类准确率(%)', '参与率(%)', '混投总量(kg)', '厨余袋混投(kg)']
    
    st.dataframe(
        display_df.style.apply(highlight_high_risk, subset=['分类准确率(%)']),
        use_container_width=True,
        height=500,
        hide_index=True
    )
    
    st.markdown("#### 高风险小区分布")
    high_risk_df = community_stats[high_risk_mask].copy()
    if len(high_risk_df) > 0:
        high_risk_by_sub = high_risk_df.groupby('街道').size().reset_index(name='高风险小区数')
        
        fig_risk = px.bar(
            high_risk_by_sub,
            x='街道',
            y='高风险小区数',
            color='高风险小区数',
            color_continuous_scale='Reds',
            title='各街道高风险小区数量分布',
            text='高风险小区数'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.success("🎉 太棒了！当前筛选条件下没有高风险小区！")

with tab4:
    st.subheader("🥡 混投品类分析")
    
    kitchen_mixed = filtered_df['厨余袋中混投量(kg)'].sum()
    other_mixed = filtered_df['其他混投量(kg)'].sum()
    plastic_in_kitchen = filtered_df['混投塑料瓶(kg)'].sum()
    paper_in_kitchen = filtered_df['混投纸张(kg)'].sum()
    glass_in_kitchen = filtered_df['混投玻璃(kg)'].sum()
    other_in_kitchen = kitchen_mixed - plastic_in_kitchen - paper_in_kitchen - glass_in_kitchen
    total_mixed = kitchen_mixed + other_mixed
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("塑料瓶混投", f"{plastic_in_kitchen:,.0f} kg")
    with col_m2:
        st.metric("纸张混投", f"{paper_in_kitchen:,.0f} kg")
    with col_m3:
        st.metric("玻璃混投", f"{glass_in_kitchen:,.0f} kg")
    with col_m4:
        st.metric("总混投量", f"{total_mixed:,.0f} kg")
    
    st.markdown("#### 混投主要品类分布")
    
    col_ana1, col_ana2 = st.columns(2)
    
    with col_ana1:
        all_mixed_detail = pd.DataFrame({
            '品类': ['厨余袋-塑料瓶', '厨余袋-纸张', '厨余袋-玻璃', '厨余袋-其他', '非厨余袋混投'],
            '重量(kg)': [
                plastic_in_kitchen,
                paper_in_kitchen,
                glass_in_kitchen,
                max(other_in_kitchen, 0),
                other_mixed
            ]
        })
        
        fig_mix_overall = px.pie(
            all_mixed_detail,
            values='重量(kg)',
            names='品类',
            color_discrete_sequence=['#e74c3c', '#f39c12', '#3498db', '#95a5a6', '#8e44ad'],
            title='全部混投品类分布（互斥）',
            hole=0.4
        )
        fig_mix_overall.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_mix_overall, use_container_width=True)
    
    with col_ana2:
        kitchen_mixed_detail = pd.DataFrame({
            '品类': ['塑料瓶', '纸张', '玻璃', '其他'],
            '重量(kg)': [
                plastic_in_kitchen,
                paper_in_kitchen,
                glass_in_kitchen,
                max(other_in_kitchen, 0)
            ]
        })
        
        fig_mix1 = px.pie(
            kitchen_mixed_detail,
            values='重量(kg)',
            names='品类',
            color_discrete_sequence=['#e74c3c', '#f39c12', '#3498db', '#95a5a6'],
            title='厨余袋内混投品类分布',
            hole=0.4
        )
        fig_mix1.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_mix1, use_container_width=True)
    
    st.markdown("#### 各街道混投品类对比")
    subdistrict_mixed = filtered_df.groupby('街道').agg({
        '混投塑料瓶(kg)': 'sum',
        '混投纸张(kg)': 'sum',
        '混投玻璃(kg)': 'sum'
    }).reset_index()
    subdistrict_mixed.columns = ['街道', '塑料瓶', '纸张', '玻璃']
    
    fig_mix2 = go.Figure()
    fig_mix2.add_trace(go.Bar(
        x=subdistrict_mixed['街道'],
        y=subdistrict_mixed['塑料瓶'],
        name='塑料瓶',
        marker_color='#e74c3c'
    ))
    fig_mix2.add_trace(go.Bar(
        x=subdistrict_mixed['街道'],
        y=subdistrict_mixed['纸张'],
        name='纸张',
        marker_color='#f39c12'
    ))
    fig_mix2.add_trace(go.Bar(
        x=subdistrict_mixed['街道'],
        y=subdistrict_mixed['玻璃'],
        name='玻璃',
        marker_color='#3498db'
    ))
    fig_mix2.update_layout(
        title='各街道混投品类对比（堆叠）',
        barmode='stack',
        yaxis_title='混投重量 (kg)',
        height=450
    )
    st.plotly_chart(fig_mix2, use_container_width=True)
    
    st.markdown("#### 混投品类月度趋势")
    monthly_mixed = filtered_df.assign(年月=filtered_df['日期'].dt.to_period('M').astype(str))
    monthly_mixed = monthly_mixed.groupby('年月').agg({
        '混投塑料瓶(kg)': 'sum',
        '混投纸张(kg)': 'sum',
        '混投玻璃(kg)': 'sum'
    }).reset_index()
    
    fig_mix3 = go.Figure()
    fig_mix3.add_trace(go.Scatter(
        x=monthly_mixed['年月'],
        y=monthly_mixed['混投塑料瓶(kg)'],
        name='塑料瓶',
        line=dict(color='#e74c3c', width=2),
        mode='lines+markers'
    ))
    fig_mix3.add_trace(go.Scatter(
        x=monthly_mixed['年月'],
        y=monthly_mixed['混投纸张(kg)'],
        name='纸张',
        line=dict(color='#f39c12', width=2),
        mode='lines+markers'
    ))
    fig_mix3.add_trace(go.Scatter(
        x=monthly_mixed['年月'],
        y=monthly_mixed['混投玻璃(kg)'],
        name='玻璃',
        line=dict(color='#3498db', width=2),
        mode='lines+markers'
    ))
    fig_mix3.update_layout(
        title='各品类混投量月度变化趋势',
        yaxis_title='混投重量 (kg)',
        height=400
    )
    st.plotly_chart(fig_mix3, use_container_width=True)

with tab5:
    st.subheader("🏆 各街道综合评分")
    
    if weights_all_zero:
        st.error("⚠️ 所有评分权重均为0，综合得分将全部为0。请至少为一个维度设置大于0的权重。")
    
    st.markdown("##### 评分维度说明")
    score_desc_col1, score_desc_col2, score_desc_col3, score_desc_col4 = st.columns(4)
    with score_desc_col1:
        st.info("🎯 **分类准确率**\n\n准确分类的垃圾占总垃圾的比例，越高越好")
    with score_desc_col2:
        st.info("👥 **参与率**\n\n每日去投放站点投递的户数占总户数的比例，越高越好")
    with score_desc_col3:
        st.info("🚫 **混投控制**\n\n混投率越低得分越高，反映居民分类规范性")
    with score_desc_col4:
        st.info("📉 **垃圾减量**\n\n其他垃圾占比越低得分越高，反映源头减量效果")
    
    sub_score = calculate_subdistrict_scores(
        filtered_df,
        w_accuracy_norm,
        w_participation_norm,
        w_mixed_control_norm,
        w_reduction_norm
    )
    
    if weights_all_zero:
        weight_display = "全部为0 → 综合得分全部为0"
    else:
        weight_display = (f"分类准确率 {w_accuracy_norm*100:.1f}% ｜ "
                         f"参与率 {w_participation_norm*100:.1f}% ｜ "
                         f"混投控制 {w_mixed_control_norm*100:.1f}% ｜ "
                         f"垃圾减量 {w_reduction_norm*100:.1f}%")
    st.markdown(f"**当前权重配置**：{weight_display}")
    
    col_radar, col_rank = st.columns([1, 1])
    
    with col_radar:
        st.markdown("#### 🕸️ 各街道评分雷达图")
        radar_data = get_radar_chart_data(sub_score)
        categories = radar_data['categories']
        
        colors_radar = [
            '#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e'
        ]
        
        fig_radar = go.Figure()
        for idx, sd in enumerate(radar_data['subdistricts']):
            values = sd['values']
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                name=sd['name'],
                opacity=0.25,
                line=dict(color=colors_radar[idx % len(colors_radar)], width=2)
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                )
            ),
            showlegend=True,
            legend=dict(orientation='h', y=-0.15, font=dict(size=10)),
            height=550,
            margin=dict(l=40, r=40, t=30, b=60)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col_rank:
        st.markdown("#### 🏅 综合得分排行榜")
        display_score = sub_score[['排名', '街道', '分类准确率得分', '参与率得分', '混投控制得分', '垃圾减量得分', '综合得分']].copy()
        display_score = display_score.round(1)
        display_score.columns = ['排名', '街道', '分类准确率', '参与率', '混投控制', '垃圾减量', '综合得分']
        
        def highlight_top3(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            for i in range(min(3, len(df))):
                if i == 0:
                    styles.iloc[i] = 'background-color: #ffd700; color: #333; font-weight: bold'
                elif i == 1:
                    styles.iloc[i] = 'background-color: #c0c0c0; color: #333; font-weight: bold'
                elif i == 2:
                    styles.iloc[i] = 'background-color: #cd7f32; color: #fff; font-weight: bold'
            return styles
        
        st.dataframe(
            display_score.style.apply(highlight_top3, axis=None),
            use_container_width=True,
            height=450,
            hide_index=True
        )
    
    st.markdown("#### 📊 各维度得分对比")
    score_melted = sub_score.melt(
        id_vars=['街道'],
        value_vars=['分类准确率得分', '参与率得分', '混投控制得分', '垃圾减量得分'],
        var_name='维度',
        value_name='得分'
    )
    score_melted['维度'] = score_melted['维度'].str.replace('得分', '')
    
    fig_dim_bar = px.bar(
        score_melted,
        x='街道',
        y='得分',
        color='维度',
        barmode='group',
        color_discrete_map={
            '分类准确率': '#27ae60',
            '参与率': '#3498db',
            '混投控制': '#e74c3c',
            '垃圾减量': '#f39c12'
        },
        title='各街道各维度得分对比'
    )
    fig_dim_bar.update_layout(yaxis=dict(range=[0, 110]), height=400)
    st.plotly_chart(fig_dim_bar, use_container_width=True)
    
    st.markdown("#### 📈 综合得分排名柱状图")
    sub_score_sorted_asc = sub_score.sort_values('综合得分', ascending=True)
    
    bar_colors = []
    for rank in sub_score_sorted_asc['排名']:
        if rank == 1:
            bar_colors.append('#ffd700')
        elif rank == 2:
            bar_colors.append('#c0c0c0')
        elif rank == 3:
            bar_colors.append('#cd7f32')
        else:
            bar_colors.append('#3498db')
    
    fig_total = go.Figure(go.Bar(
        x=sub_score_sorted_asc['综合得分'],
        y=sub_score_sorted_asc['街道'],
        orientation='h',
        marker=dict(color=bar_colors),
        text=sub_score_sorted_asc['综合得分'].round(1).astype(str) + '分',
        textposition='auto'
    ))
    fig_total.update_layout(
        title='各街道综合得分排名（金🥇 银🥈 铜🥉）',
        xaxis_title='综合得分',
        yaxis_title='街道',
        xaxis=dict(range=[0, 110]),
        height=450
    )
    st.plotly_chart(fig_total, use_container_width=True)

st.markdown("---")
st.markdown("💡 **数据说明**：本看板数据为模拟数据，涵盖2025年1月至2026年5月，共8个街道、24个社区的垃圾分类数据。")
