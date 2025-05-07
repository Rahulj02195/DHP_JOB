# from flask import Flask, jsonify, render_template
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# import re
# import os
# import json
# import random

# app = Flask(__name__)
# # Enable CORS for all routes
# CORS(app)

# # Define parse_ctc function first so it's available in the try block
# def parse_ctc(ctc_str):
#     """
#     Convert strings like '₹ 3,00,000 - 4,60,000 /year' into a numeric float.
#     Only uses the first occurrence in case of duplication.
#     1. Extract the first '₹ x - y' format using regex.
#     2. Remove '₹', commas, and other non-numeric characters.
#     3. Take average if it's a range.
#     """
#     if not isinstance(ctc_str, str):
#         return None

#     # Find the first salary range using regex
#     match = re.search(r'₹?\s?[\d,]+(?:\s?-\s?[\d,]+)?', ctc_str)
#     if not match:
#         return None
    
#     salary_str = match.group().replace('₹', '').strip()
#     parts = salary_str.split('-')

#     nums = []
#     for part in parts:
#         part = part.replace(',', '').strip()
#         try:
#             nums.append(float(part))
#         except ValueError:
#             continue
#     if not nums:
#         return None
#     elif len(nums) == 1:
#         return nums[0]
#     else:
#         return sum(nums) / len(nums)

# # Sample data - will be used if CSV file is not found
# sample_data = {
#     "skill_avg_ctc_dict": {
#         "Python": 750000, "Java": 850000, "JavaScript": 700000, "Data Science": 900000,
#         "ML": 950000, "AI": 1000000, "React": 780000, "Angular": 760000,
#         "Node.js": 800000, "SQL": 650000
#     },
#     "skill_counts_dict": {
#         "Python": 150, "Java": 130, "JavaScript": 120, "Data Science": 80,
#         "ML": 70, "AI": 60, "React": 100, "Angular": 90, "Node.js": 110, "SQL": 140
#     },
#     "location_counts_dict": {
#         "Bangalore": 200, "Mumbai": 150, "Hyderabad": 130, "Delhi": 120,
#         "Pune": 100, "Chennai": 90, "Kolkata": 70, "Noida": 60
#     },
#     "avg_ctc_by_location_dict": {
#         "Bangalore": 850000, "Mumbai": 900000, "Hyderabad": 800000, "Delhi": 820000,
#         "Pune": 780000, "Chennai": 760000, "Kolkata": 720000, "Noida": 750000
#     }
# }

# # Try to load data from CSV
# try:
#     # Use a relative file path that will work when deployed
#     csv_path = os.path.join(os.path.dirname(__file__), 'static/data/filtered_dataset.csv')
    
#     # For fallback, if the file doesn't exist at the relative path, use the provided path
#     if not os.path.exists(csv_path):
#         csv_path = "filtered_dataset.csv"
    
#     # If neither path works, create a sample dataset
#     if not os.path.exists(csv_path):
#         # Create a simple sample dataset with the right dimensions
#         skills = list(sample_data["skill_counts_dict"].keys())
#         locations = list(sample_data["location_counts_dict"].keys())
#         num_skills = len(skills)
#         num_locations = len(locations)
#         num_rows = 1000
        
#         # Create arrays of equal length for the DataFrame
#         skill_array = np.random.choice(skills, num_rows)
#         company_array = [f"Company{i}" for i in range(1, num_rows+1)]
#         location_array = np.random.choice(locations, num_rows)
#         ctc_array = np.random.randint(500000, 1500000, num_rows)
        
#         sample_df = pd.DataFrame({
#             'skill_required': skill_array,
#             'company_name': company_array,
#             'location': location_array,
#             'ctc': ctc_array
#         })
        
#         # Save to CSV for future use
#         os.makedirs(os.path.dirname(csv_path), exist_ok=True)
#         sample_df.to_csv(csv_path, index=False)
#         df = sample_df
#     else:
#         df = pd.read_csv(csv_path)
        
#     # Apply parse function for CTC if needed
#     if 'ctc' in df.columns and not pd.api.types.is_numeric_dtype(df['ctc']):
#         df['ctc'] = df['ctc'].apply(parse_ctc)
#         df['ctc'] = pd.to_numeric(df['ctc'], errors='coerce')
#         df.dropna(subset=['ctc'], inplace=True)
    
#     # Data processing
#     skill_avg_ctc_series = df.groupby('skill_required')['ctc'].mean()
#     skill_avg_ctc_dict = skill_avg_ctc_series.to_dict()
    
#     skill_counts_series = df['skill_required'].value_counts()
#     skill_counts_dict = skill_counts_series.to_dict()
    
#     # Process location data
#     df['location'] = df['location'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)
#     location_counts_series = df['location'].value_counts()
#     location_counts_dict = location_counts_series.to_dict()
    
#     box_data_skill = {}
#     for skill, group in df.groupby('skill_required'):
#         box_data_skill[skill] = group['ctc'].tolist()
    
#     box_data_location = {}
#     for loc, group in df.groupby('location'):
#         box_data_location[loc] = group['ctc'].tolist()
    
#     skill_avg_sorted_series = skill_avg_ctc_series.sort_values()
#     skill_avg_sorted_dict = skill_avg_sorted_series.to_dict()
    
#     avg_ctc_by_location_series = df.groupby('location')['ctc'].mean()
#     avg_ctc_by_location_dict = avg_ctc_by_location_series.to_dict()
    
#     scatter_data = []
#     for _, row in df.iterrows():
#         scatter_data.append({
#             'skill': row['skill_required'],
#             'ctc': float(row['ctc']),
#             'avg_ctc': float(skill_avg_ctc_dict[row['skill_required']])
#         })
    
#     pivot_df = df.pivot_table(values='ctc', index='skill_required', columns='location', aggfunc='mean').fillna(0)
#     heatmap_data = {
#         'skills': list(pivot_df.index),
#         'locations': list(pivot_df.columns),
#         'matrix': pivot_df.values.tolist()
#     }
    
#     crosstab_df = pd.crosstab(df['location'], df['skill_required'])
#     stacked_bar_data = {
#         'locations': crosstab_df.index.tolist(),
#         'skills': crosstab_df.columns.tolist(),
#         'values': crosstab_df.values.tolist()
#     }
    
#     ctc_min, ctc_max = df['ctc'].min(), df['ctc'].max()
#     bubble_data = []
#     for _, row in df.iterrows():
#         size = 100 + (row['ctc'] - ctc_min) / (ctc_max - ctc_min) * 900
#         bubble_data.append({
#             'company_name': row['company_name'],
#             'ctc': float(row['ctc']),
#             'skill_required': row['skill_required'],
#             'bubble_size': size
#         })
        
# except Exception as e:
#     print(f"Error loading/processing data: {e}")
#     # Use sample data as fallback
#     skill_avg_ctc_dict = sample_data["skill_avg_ctc_dict"]
#     skill_counts_dict = sample_data["skill_counts_dict"]
#     location_counts_dict = sample_data["location_counts_dict"]
#     avg_ctc_by_location_dict = sample_data["avg_ctc_by_location_dict"]
    
#     # Create empty datasets for other visualizations
#     box_data_skill = {k: [v] for k, v in skill_avg_ctc_dict.items()}
#     box_data_location = {k: [v] for k, v in avg_ctc_by_location_dict.items()}
#     skill_avg_sorted_dict = dict(sorted(skill_avg_ctc_dict.items(), key=lambda x: x[1]))
    
#     # Create scatter data with varied CTC values
#     scatter_data = []
#     import random
#     random.seed(123)  # For reproducibility
    
#     for skill, avg in skill_avg_ctc_dict.items():
#         # Create multiple data points for each skill with variations
#         for _ in range(5):
#             # Random variation around the average
#             variation = random.uniform(0.7, 1.3)
#             scatter_data.append({
#                 'skill': skill,
#                 'ctc': avg * variation,  # Individual CTC varies
#                 'avg_ctc': avg  # Keep the average constant
#             })
    
#     # Create meaningful heatmap data
#     skills = list(skill_avg_ctc_dict.keys())
#     locations = list(avg_ctc_by_location_dict.keys())
    
#     # Create a matrix with varied values
#     import random
#     random.seed(456)  # For reproducibility
    
#     matrix = []
#     for skill_idx, skill in enumerate(skills):
#         skill_row = []
#         base_val = skill_avg_ctc_dict[skill]
        
#         for loc_idx, loc in enumerate(locations):
#             # Create location-dependent skill salary variations
#             loc_factor = 0.8 + (loc_idx % 4) * 0.1  # Location factor
#             skill_factor = 0.9 + (skill_idx % 3) * 0.1  # Skill factor
#             random_factor = random.uniform(0.9, 1.1)  # Small random variation
            
#             # Calculate the CTC value for this skill-location combination
#             value = base_val * loc_factor * skill_factor * random_factor
#             skill_row.append(round(value, 2))
            
#         matrix.append(skill_row)
    
#     heatmap_data = {
#         'skills': skills,
#         'locations': locations,
#         'matrix': matrix
#     }
    
#     # Create stacked bar data - with more realistic distribution
#     stacked_bar_data = {
#         'locations': locations,
#         'skills': skills,
#         'values': []
#     }
    
#     # Create varied data for each location
#     import random
#     random.seed(42)  # For reproducibility
    
#     for i, location in enumerate(locations):
#         skill_values = []
#         for skill in skills:
#             # Create varied values based on skill popularity
#             base_count = skill_counts_dict.get(skill, 50)
#             location_factor = 0.5 + (i % 3) * 0.25  # Different locations have different demands
#             value = int(base_count * location_factor * random.uniform(0.8, 1.2))
#             skill_values.append(value)
#         stacked_bar_data['values'].append(skill_values)
    
#     # Create more informative bubble data with multiple companies per skill
#     bubble_data = []
#     company_prefixes = ['Tech', 'Global', 'Future', 'Data', 'Smart', 'Info', 'Next', 'Digi', 'Cloud', 'AI']
#     company_suffixes = ['Corp', 'Systems', 'Solutions', 'Tech', 'Group', 'Labs', 'Works', 'IT', 'Innovations', 'Soft']
    
#     import random
#     random.seed(789)  # For reproducibility
    
#     for skill in skills:
#         base_ctc = skill_avg_ctc_dict[skill]
        
#         # Create multiple companies for each skill
#         for i in range(3):
#             # Generate company name
#             prefix = random.choice(company_prefixes)
#             suffix = random.choice(company_suffixes)
#             company_name = f"{prefix}{suffix}"
            
#             # Create variation in CTC
#             ctc_factor = 0.85 + random.random() * 0.3  # Between 0.85 and 1.15
#             ctc = base_ctc * ctc_factor
            
#             # Bubble size represents relative difference in salary
#             size = 200 + (ctc_factor - 0.85) * 2000
            
#             bubble_data.append({
#                 'company_name': company_name,
#                 'ctc': float(ctc),
#                 'skill_required': skill,
#                 'bubble_size': size
#             })

# def parse_ctc(ctc_str):
#     """
#     Convert strings like '₹ 3,00,000 - 4,60,000 /year' into a numeric float.
#     Only uses the first occurrence in case of duplication.
#     1. Extract the first '₹ x - y' format using regex.
#     2. Remove '₹', commas, and other non-numeric characters.
#     3. Take average if it's a range.
#     """
#     if not isinstance(ctc_str, str):
#         return None

#     # Find the first salary range using regex
#     match = re.search(r'₹?\s?[\d,]+(?:\s?-\s?[\d,]+)?', ctc_str)
#     if not match:
#         return None
    
#     salary_str = match.group().replace('₹', '').strip()
#     parts = salary_str.split('-')

#     nums = []
#     for part in parts:
#         part = part.replace(',', '').strip()
#         try:
#             nums.append(float(part))
#         except ValueError:
#             continue
#     if not nums:
#         return None
#     elif len(nums) == 1:
#         return nums[0]
#     else:
#         return sum(nums) / len(nums)

# ###############################################################################
# # ROUTES
# ###############################################################################

# @app.route('/')
# def index():
#     return render_template('index.html')

# ###############################################################################
# # API ENDPOINTS
# ###############################################################################

# @app.route('/api/1_avg_ctc_per_skill')
# def api_avg_ctc_per_skill():
#     limited_items = list(skill_avg_ctc_dict.items())[:10]
#     labels, values = zip(*limited_items) if limited_items else ([], [])
    
#     return jsonify({
#         'labels': list(labels),
#         'values': list(values)
#     })

# @app.route('/api/2_company_count_per_skill')
# def api_company_count_per_skill():
#     return jsonify({
#         'labels': list(skill_counts_dict.keys()),
#         'values': list(skill_counts_dict.values())
#     })

# @app.route('/api/3_company_count_per_location')
# def api_company_count_per_location():
#     return jsonify({
#         'labels': list(location_counts_dict.keys()),
#         'values': list(location_counts_dict.values())
#     })

# @app.route('/api/4_boxplot_ctc_per_skill')
# def api_boxplot_ctc_per_skill():
#     return jsonify(box_data_skill)

# @app.route('/api/5_boxplot_ctc_per_location')
# def api_boxplot_ctc_per_location():
#     return jsonify(box_data_location)

# @app.route('/api/6_line_avg_ctc_skills')
# def api_line_avg_ctc_skills():
#     return jsonify({
#         'labels': list(skill_avg_sorted_dict.keys()),
#         'values': list(skill_avg_sorted_dict.values())
#     })

# @app.route('/api/7_pie_skill_demand')
# def api_pie_skill_demand():
#     total = sum(skill_counts_dict.values())
#     threshold = 0.02 * total  # 2% of total

#     labels = []
#     values = []
#     other_total = 0

#     for skill, count in skill_counts_dict.items():
#         if count >= threshold:
#             labels.append(skill)
#             values.append(count)
#         else:
#             other_total += count

#     # if other_total > 0:
#     #     labels.append("Other")
#     #     values.append(other_total)

#     return jsonify({
#         'labels': labels,
#         'values': values
#     })


# @app.route('/api/8_pie_location_distribution')
# def api_pie_location_distribution():
#     # Count the actual number of job listings per location
#     location_job_counts = df['location'].value_counts()
    
#     # Get the top 8 locations with the most job listings
#     top_locations = location_job_counts.nlargest(8)
    
#     # Calculate the sum of all remaining locations
#     other_total = location_job_counts[~location_job_counts.index.isin(top_locations.index)].sum()
    
#     # Prepare data for the pie chart
#     labels = top_locations.index.tolist()
#     values = top_locations.values.tolist()
    
#     # Add "Other" category if it's significant
#     if other_total > 0:
#         labels.append("Other")
#         values.append(int(other_total))
    
#     # Include a clearer explanation in the result
#     return jsonify({
#         'labels': labels,
#         'values': values,
#         'tooltip_label': 'Job Listings'  # Used in the tooltip to clarify what the values represent
#     })


# @app.route('/api/9_avg_ctc_per_location')
# def api_avg_ctc_per_location():
#     return jsonify({
#         'labels': list(avg_ctc_by_location_dict.keys()),
#         'values': list(avg_ctc_by_location_dict.values())
#     })

# @app.route('/api/10_scatter_ctc_vs_avg')
# def api_scatter_ctc_vs_avg():
#     # Create a dataset with the top 100 companies by CTC
#     top_companies_df = df.sort_values('ctc', ascending=False).head(100)
    
#     # Create a dataset with actual company names for the scatter plot
#     improved_scatter_data = []
    
#     # Get the average CTC per skill for the entire dataset
#     skill_avg_ctc = df.groupby('skill_required')['ctc'].mean().to_dict()
    
#     # For each of the top companies, create a scatter point
#     for _, row in top_companies_df.iterrows():
#         skill = row['skill_required']
#         # Only include if we have the skill average available
#         if skill in skill_avg_ctc:
#             improved_scatter_data.append({
#                 'company': row['company_name'],  # Include actual company name
#                 'skill': skill,
#                 'ctc': float(row['ctc']),
#                 'avg_ctc': float(skill_avg_ctc[skill])
#             })
    
#     return jsonify(improved_scatter_data)

# @app.route('/api/11_heatmap_skill_location')
# def api_heatmap_skill_location():
#     # Use pivot table to create a clean heatmap of skill vs location
#     pivot_df = df.pivot_table(
#         values='ctc', 
#         index='skill_required', 
#         columns='location', 
#         aggfunc='mean',
#         fill_value=0
#     )
    
#     # Get top 15 skills by overall average CTC
#     top_skills = df.groupby('skill_required')['ctc'].mean().nlargest(15).index.tolist()
    
#     # Get top 10 locations by job count
#     top_locations = df['location'].value_counts().nlargest(10).index.tolist()
    
#     # Filter the pivot table to include only top skills and locations
#     filtered_pivot = pivot_df.loc[
#         pivot_df.index.isin(top_skills), 
#         [col for col in pivot_df.columns if col in top_locations]
#     ]
    
#     # Create the heatmap data structure
#     heatmap_data = {
#         'skills': filtered_pivot.index.tolist(),
#         'locations': filtered_pivot.columns.tolist(),
#         'matrix': filtered_pivot.values.tolist()
#     }
    
#     return jsonify(heatmap_data)


# @app.route('/api/12_stacked_skills_location')
# def api_stacked_skills_location():
#     # Get top 10 skills by demand (job count)
#     top_skills = df['skill_required'].value_counts().nlargest(10).index.tolist()
    
#     # Get top 8 locations by job count
#     top_locations = df['location'].value_counts().nlargest(8).index.tolist()
    
#     # Filter dataframe to include only these top skills and locations
#     filtered_df = df[
#         df['skill_required'].isin(top_skills) & 
#         df['location'].isin(top_locations)
#     ]
    
#     # Create a cross-tabulation of skills by location
#     crosstab_df = pd.crosstab(filtered_df['location'], filtered_df['skill_required'])
    
#     # Generate the data structure for the stacked bar chart
#     improved_stacked_data = {
#         'locations': crosstab_df.index.tolist(),
#         'skills': crosstab_df.columns.tolist(),
#         'values': crosstab_df.values.tolist(),
#         'tooltip_label': 'Job Listings'  # Clarify what the values represent
#     }
    
#     return jsonify(improved_stacked_data)

# @app.route('/api/13_bubble_company_ctc')
# def api_bubble_company_ctc():
#     # Get top 100 companies by CTC
#     top_companies_df = df.sort_values('ctc', ascending=False).head(100)
    
#     # Calculate min and max CTC for scaling
#     ctc_min, ctc_max = top_companies_df['ctc'].min(), top_companies_df['ctc'].max()
    
#     # Create bubble chart data with actual company names
#     improved_bubble_data = []
    
#     # Process each company to create a data point
#     for _, row in top_companies_df.iterrows():
#         # Scale bubble size proportionally to CTC
#         size = 50 + (row['ctc'] - ctc_min) / (ctc_max - ctc_min) * 500
        
#         improved_bubble_data.append({
#             'company_name': row['company_name'],  # Use actual company name
#             'ctc': float(row['ctc']),
#             'skill_required': row['skill_required'],
#             'bubble_size': size
#         })
    
#     return jsonify(improved_bubble_data)

# ###############################################################################
# # RUN FLASK
# ###############################################################################
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)








from flask import Flask, jsonify, render_template, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import os

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Utility function to parse CTC strings
def parse_ctc(ctc_str):
    """
    Convert strings like '₹ 3,00,000 - 4,60,000 /year' into a numeric float.
    Only uses the first occurrence in case of duplication.
    1. Extract the first '₹ x - y' format using regex.
    2. Remove '₹', commas, and other non-numeric characters.
    3. Take average if it's a range.
    """
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

# Load dataset
csv_rel_path = os.path.join(os.path.dirname(__file__), 'static/data/filtered_dataset.csv')
csv_alt_path = 'filtered_dataset.csv'

if os.path.exists(csv_rel_path):
    df = pd.read_csv(csv_rel_path)
elif os.path.exists(csv_alt_path):
    df = pd.read_csv(csv_alt_path)
else:
    abort(500, description='Dataset CSV file not found.')

# Ensure numeric CTC
if 'ctc' in df.columns and not pd.api.types.is_numeric_dtype(df['ctc']):
    df['ctc'] = df['ctc'].apply(parse_ctc)
    df['ctc'] = pd.to_numeric(df['ctc'], errors='coerce')
    df.dropna(subset=['ctc'], inplace=True)

# Normalize location column
df['location'] = df['location'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)


# Precompute aggregations
skill_avg_ctc_dict = df.groupby('skill_required')['ctc'].mean().to_dict()
skill_counts_dict = df['skill_required'].value_counts().to_dict()
location_counts_dict = df['location'].value_counts().to_dict()
avg_ctc_by_location_dict = df.groupby('location')['ctc'].mean().to_dict()

box_data_skill = {skill: group['ctc'].tolist() for skill, group in df.groupby('skill_required')}
box_data_location = {loc: group['ctc'].tolist() for loc, group in df.groupby('location')}
skill_avg_sorted_dict = dict(df.groupby('skill_required')['ctc'].mean().sort_values())


# count distinct companies per skill
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



# Scatter data
df_skill_avg = df.groupby('skill_required')['ctc'].mean().to_dict()
scatter_data = [
    {'skill': row['skill_required'], 'ctc': float(row['ctc']), 'avg_ctc': float(df_skill_avg[row['skill_required']])}
    for _, row in df.iterrows()
]

# Heatmap data
pivot_df = df.pivot_table(values='ctc', index='skill_required', columns='location', aggfunc='mean', fill_value=0)
heatmap_data = {
    'skills': pivot_df.index.tolist(),
    'locations': pivot_df.columns.tolist(),
    'matrix': pivot_df.values.tolist()
}

# Stacked bar data
crosstab_df = pd.crosstab(df['location'], df['skill_required'])
stacked_bar_data = {
    'locations': crosstab_df.index.tolist(),
    'skills': crosstab_df.columns.tolist(),
    'values': crosstab_df.values.tolist()
}

# Bubble data (scatter of companies)
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

###############################################################################
# ROUTES
###############################################################################

@app.route('/')
def index():
    return render_template('index.html')

###############################################################################
# API ENDPOINTS
###############################################################################

# @app.route('/api/1_avg_ctc_per_skill')
# def api_avg_ctc_per_skill():
#     items = list(skill_avg_ctc_dict.items())[:10]
#     labels, values = zip(*items) if items else ([], [])
#     return jsonify({'labels': list(labels), 'values': list(values)})
@app.route('/api/1_avg_ctc_per_skill')
def api_avg_ctc_per_skill():
    # Sort the skills by average CTC in descending order
    sorted_items = sorted(skill_avg_ctc_dict.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:9]
    labels, values = zip(*top_items) if top_items else ([], [])
    return jsonify({'labels': list(labels), 'values': list(values)})


# @app.route('/api/2_company_count_per_skill')
# def api_company_count_per_skill():
#     return jsonify({'labels': list(skill_counts_dict.keys()), 'values': list(skill_counts_dict.values())})

@app.route('/api/2_company_count_per_skill')
def api_company_count_per_skill():
    # sort skills by number of distinct companies descending
    sorted_items = sorted(
        skill_company_counts_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    # take just the top 10
    top_items = sorted_items[1:15]
    labels, values = zip(*top_items) if top_items else ([], [])
    return jsonify({'labels': list(labels), 'values': list(values)})


# @app.route('/api/3_company_count_per_location')
# def api_company_count_per_location():
#     return jsonify({'labels': list(location_counts_dict.keys()), 'values': list(location_counts_dict.values())})

@app.route('/api/3_company_count_per_location')
def api_company_count_per_location():
    # sort by number of distinct companies, descending
    sorted_items = sorted(
        location_company_counts_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    # take top 10 locations
    top_items = sorted_items[:10]
    labels, values = zip(*top_items) if top_items else ([], [])
    return jsonify({'labels': list(labels), 'values': list(values)})

# @app.route('/api/4_boxplot_ctc_per_skill')
# def api_boxplot_ctc_per_skill():
#     return jsonify(box_data_skill)

# @app.route('/api/5_boxplot_ctc_per_location')
# def api_boxplot_ctc_per_location():
#     return jsonify(box_data_location)

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

# @app.route('/api/11_heatmap_skill_location')
# def api_heatmap_skill_location():
#     pivot = df.pivot_table(values='ctc', index='skill_required', columns='location', aggfunc='mean', fill_value=0)
#     top_skills = df.groupby('skill_required')['ctc'].mean().nlargest(15).index.tolist()
#     top_locs = df['location'].value_counts().nlargest(10).index.tolist()
#     filtered = pivot.loc[pivot.index.isin(top_skills), top_locs]
#     return jsonify({'skills': filtered.index.tolist(), 'locations': filtered.columns.tolist(), 'matrix': filtered.values.tolist()})

@app.route('/api/11_heatmap_skill_location')
def api_heatmap_skill_location():
    # Get top locations by the number of companies
    top_locs = df['location'].value_counts().nlargest(10).index.tolist()  # Ensure this accurately reflects location count
    # Get top skills
    top_skills = df.groupby('skill_required')['ctc'].mean().nlargest(15).index.tolist()
    # Pivot for heatmap data
    pivot = df.pivot_table(values='ctc', index='skill_required', columns='location', aggfunc='mean', fill_value=0)
    # Filter pivot table on top skills and top locations
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

###############################################################################
# RUN FLASK
###############################################################################

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
