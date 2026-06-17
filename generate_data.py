import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

subdistricts = ['和平街道', '解放街道', '建设街道', '红旗街道', '新华街道', '光明街道', '胜利街道', '人民街道']
communities = {
    '和平街道': ['和平里社区', '和平家园社区', '和平西苑社区'],
    '解放街道': ['解放门社区', '解放路社区', '解放桥社区'],
    '建设街道': ['建设路社区', '建设里社区', '建设花园社区'],
    '红旗街道': ['红旗村社区', '红旗路社区', '红旗花园社区'],
    '新华街道': ['新华路社区', '新华里社区', '新华园社区'],
    '光明街道': ['光明路社区', '光明里社区', '光明花园社区'],
    '胜利街道': ['胜利村社区', '胜利路社区', '胜利花园社区'],
    '人民街道': ['人民路社区', '人民里社区', '人民公园社区'],
}
all_communities = [(sub, com) for sub, coms in communities.items() for com in coms]

start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 5, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

waste_data = []
for date in date_range:
    month = date.month
    year = date.year
    for subdistrict, community in all_communities:
        base_kitchen = np.random.uniform(500, 2000)
        base_recyclable = np.random.uniform(200, 800)
        base_harmful = np.random.uniform(5, 30)
        base_other = np.random.uniform(300, 1200)
        
        accuracy_trend = 0.7 + (date.toordinal() - start_date.toordinal()) * 0.0003
        accuracy = np.clip(accuracy_trend + np.random.uniform(-0.08, 0.08), 0.55, 0.95)
        
        mixed_total = (base_kitchen + base_recyclable + base_harmful + base_other) * (1 - accuracy)
        kitchen_waste = base_kitchen * accuracy + np.random.uniform(0, 50)
        recyclable = base_recyclable * accuracy + np.random.uniform(0, 30)
        harmful = base_harmful * accuracy + np.random.uniform(0, 5)
        other = base_other * accuracy + np.random.uniform(0, 40)
        
        mixed_in_kitchen = mixed_total * np.random.uniform(0.4, 0.6)
        plastic_bottles = mixed_in_kitchen * np.random.uniform(0.3, 0.5)
        paper = mixed_in_kitchen * np.random.uniform(0.15, 0.3)
        glass = mixed_in_kitchen * np.random.uniform(0.05, 0.15)
        other_mixed = mixed_total - mixed_in_kitchen
        
        total_households = np.random.randint(800, 2500)
        participating_households = int(total_households * np.random.uniform(0.4, 0.85))
        participation_rate = participating_households / total_households
        
        waste_data.append({
            '日期': date.strftime('%Y-%m-%d'),
            '年份': year,
            '月份': month,
            '街道': subdistrict,
            '社区': community,
            '总户数': total_households,
            '参与户数': participating_households,
            '参与率': round(participation_rate, 4),
            '厨余垃圾量(kg)': round(kitchen_waste, 2),
            '可回收物量(kg)': round(recyclable, 2),
            '有害垃圾量(kg)': round(harmful, 2),
            '其他垃圾量(kg)': round(other, 2),
            '分类准确率': round(accuracy, 4),
            '混投总量(kg)': round(mixed_total, 2),
            '厨余袋中混投量(kg)': round(mixed_in_kitchen, 2),
            '混投塑料瓶(kg)': round(plastic_bottles, 2),
            '混投纸张(kg)': round(paper, 2),
            '混投玻璃(kg)': round(glass, 2),
            '其他混投量(kg)': round(other_mixed, 2),
        })

df = pd.DataFrame(waste_data)
df.to_csv('data/waste_data.csv', index=False, encoding='utf-8-sig')
print(f"已生成 {len(df)} 条数据，保存到 data/waste_data.csv")

print("\n数据概览:")
print(df.head())
print("\n统计信息:")
print(df[['厨余垃圾量(kg)', '可回收物量(kg)', '有害垃圾量(kg)', '其他垃圾量(kg)', '分类准确率', '参与率']].describe())
