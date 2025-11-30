from flask import Flask, jsonify, render_template, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import os

app = Flask(__name__)

CORS(app)

def parse_ctc(ctc_str):
   
    if not isinstance(ctc_str, str):
        return None

    match = re.search(r'₹?\s?[\d,]+(?:\s?-\s?[\d,]+)?', ctc_str)
    if not match:
        return None
    salary_str = match.group().replace('₹', '').strip()
    parts = salary_str.split('-')

    nums = []
    for part in parts:
        part = part.replace(',', '').strip()
        try:
            nums.append(float(part))
        except ValueError:
            continue
    if not nums:
        return None
    return nums[0] if len(nums) == 1 else sum(nums) / len(nums)


csv_rel_path = os.path.join(os.path.dirname(__file__), 'static/data/filtered_dataset.csv')
csv_alt_path = 'filtered_dataset.csv'

if os.path.exists(csv_rel_path):
    df = pd.read_csv(csv_rel_path)
elif os.path.exists(csv_alt_path):
    df = pd.read_csv(csv_alt_path)
else:
    abort(500, description='Dataset CSV file not found.')

if 'ctc' in df.columns and not pd.api.types.is_numeric_dtype(df['ctc']):
    df['ctc'] = df['ctc'].apply(parse_ctc)
    df['ctc'] = pd.to_numeric(df['ctc'], errors='coerce')
    df.dropna(subset=['ctc'], inplace=True)

df['location'] = df['location'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)

skill_avg_ctc_dict = df.groupby('skill_required')['ctc'].mean().to_dict()
skill_counts_dict = df['skill_required'].value_counts().to_dict()
location_counts_dict = df['location'].value_counts().to_dict()
avg_ctc_by_location_dict = df.groupby('location')['ctc'].mean().to_dict()

box_data_skill = {skill: group['ctc'].tolist() for skill, group in df.groupby('skill_required')}
box_data_location = {loc: group['ctc'].tolist() for loc, group in df.groupby('location')}
skill_avg_sorted_dict = dict(df.groupby('skill_required')['ctc'].mean().sort_values())

skill_company_counts_dict = (
    df.groupby('skill_required')['company_name']
      .nunique()
      .to_dict()
)

location_company_counts_dict = (
    df.groupby('location')['company_name']
      .nunique()
      .to_dict()
)


df_skill_avg = df.groupby('skill_required')['ctc'].mean().to_dict()
scatter_data = [
    {'skill': row['skill_required'], 'ctc': float(row['ctc']), 'avg_ctc': float(df_skill_avg[row['skill_required']])}
    for _, row in df.iterrows()
]

pivot_df = df.pivot_table(values='ctc', index='skill_required', columns='location', aggfunc='mean', fill_value=0)
heatmap_data = {
    'skills': pivot_df.index.tolist(),
    'locations': pivot_df.columns.tolist(),
    'matrix': pivot_df.values.tolist()
}

crosstab_df = pd.crosstab(df['location'], df['skill_required'])
stacked_bar_data = {
    'locations': crosstab_df.index.tolist(),
    'skills': crosstab_df.columns.tolist(),
    'values': crosstab_df.values.tolist()
}

ctc_min, ctc_max = df['ctc'].min(), df['ctc'].max()
bubble_data = []
for _, row in df.iterrows():
    size = 50 + (row['ctc'] - ctc_min) / (ctc_max - ctc_min) * 500
    bubble_data.append({
        'company_name': row['company_name'],
        'ctc': float(row['ctc']),
        'skill_required': row['skill_required'],
        'bubble_size': size
    })


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/1_avg_ctc_per_skill')
def api_avg_ctc_per_skill():
    sorted_items = sorted(skill_avg_ctc_dict.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:9]
    labels, values = zip(*top_items) if top_items else ([], [])
    return jsonify({'labels': list(labels), 'values': list(values)})

@app.route('/api/2_company_count_per_skill')
def api_company_count_per_skill():
    sorted_items = sorted(
        skill_company_counts_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_items = sorted_items[1:15]
    labels, values = zip(*top_items) if top_items else ([], [])
    return jsonify({'labels': list(labels), 'values': list(values)})


@app.route('/api/3_company_count_per_location')
def api_company_count_per_location():
    sorted_items = sorted(
        location_company_counts_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_items = sorted_items[:10]
    labels, values = zip(*top_items) if top_items else ([], [])
    return jsonify({'labels': list(labels), 'values': list(values)})


@app.route('/api/6_line_avg_ctc_skills')
def api_line_avg_ctc_skills():
    return jsonify({'labels': list(skill_avg_sorted_dict.keys()), 'values': list(skill_avg_sorted_dict.values())})

@app.route('/api/7_pie_skill_demand')
def api_pie_skill_demand():
    total = sum(skill_counts_dict.values())
    threshold = 0.02 * total
    labels, values, other = [], [], 0
    for skill, count in skill_counts_dict.items():
        if count >= threshold:
            labels.append(skill)
            values.append(count)
        else:
            other += count
    if other > 0:
        labels.append('Other')
        values.append(other)
    return jsonify({'labels': labels, 'values': values})

@app.route('/api/8_pie_location_distribution')
def api_pie_location_distribution():
    counts = df['location'].value_counts()
    top = counts.nlargest(8)
    other = counts[~counts.index.isin(top.index)].sum()
    labels, values = top.index.tolist(), top.values.tolist()
    if other > 0:
        labels.append('Other')
        values.append(int(other))
    return jsonify({'labels': labels, 'values': values, 'tooltip_label': 'Job Listings'})

@app.route('/api/9_avg_ctc_per_location')
def api_avg_ctc_per_location():
    return jsonify({'labels': list(avg_ctc_by_location_dict.keys()), 'values': list(avg_ctc_by_location_dict.values())})

@app.route('/api/10_scatter_ctc_vs_avg')
def api_scatter_ctc_vs_avg():
    top100 = df.nlargest(100, 'ctc')
    avg_map = df.groupby('skill_required')['ctc'].mean().to_dict()
    data = []
    for _, row in top100.iterrows():
        skill = row['skill_required']
        if skill in avg_map:
            data.append({'company': row['company_name'], 'skill': skill, 'ctc': float(row['ctc']), 'avg_ctc': float(avg_map[skill])})
    return jsonify(data)



@app.route('/api/11_heatmap_skill_location')
def api_heatmap_skill_location():
    top_locs = df['location'].value_counts().nlargest(10).index.tolist()  

    top_skills = df.groupby('skill_required')['ctc'].mean().nlargest(15).index.tolist()

    pivot = df.pivot_table(values='ctc', index='skill_required', columns='location', aggfunc='mean', fill_value=0)
    
    filtered = pivot.loc[pivot.index.isin(top_skills), top_locs]
    
    return jsonify({
        'skills': filtered.index.tolist(),
        'locations': filtered.columns.tolist(),
        'matrix': filtered.values.tolist()
    })

@app.route('/api/12_stacked_skills_location')
def api_stacked_skills_location():
    top_s = df['skill_required'].value_counts().nlargest(10).index.tolist()
    top_l = df['location'].value_counts().nlargest(8).index.tolist()
    sub = df[df['skill_required'].isin(top_s) & df['location'].isin(top_l)]
    ct = pd.crosstab(sub['location'], sub['skill_required'])
    return jsonify({'locations': ct.index.tolist(), 'skills': ct.columns.tolist(), 'values': ct.values.tolist(), 'tooltip_label': 'Job Listings'})

@app.route('/api/13_bubble_company_ctc')
def api_bubble_company_ctc():
    top = df.nlargest(100, 'ctc')
    mn, mx = top['ctc'].min(), top['ctc'].max()
    data = []
    for _, r in top.iterrows():
        size = 50 + (r['ctc'] - mn) / (mx - mn) * 500
        data.append({'company_name': r['company_name'], 'ctc': float(r['ctc']), 'skill_required': r['skill_required'], 'bubble_size': size})
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
