import pandas as pd

# Load the first CSV file
df1 = pd.read_csv('/home/nick/data/HPA_FOV_data/whole_images.csv')

# Load the second CSV file
df2 = pd.read_csv('/home/nick/Dino4Cells_analysis/exploratory_configs/external_hpa_fov_cleaned.csv')

# Filter df1 to only include rows where ID is present in df2
df1_filtered = df1[df1['ID'].isin(df2['ID'])]

# Create a dictionary from df2 for easy lookup
file_mapping = df2.set_index('ID')['file'].to_dict()

# Update the 'file' column in the filtered df1
df1_filtered['file'] = df1_filtered['ID'].map(file_mapping)

df1_filtered_unique = df1_filtered.drop_duplicates(subset=['ID'])

# Optional: Save the updated DataFrame to a new CSV
df1_filtered_unique.to_csv('/home/nick/Dino4Cells_analysis/exploratory_configs/filtered_updated_whole_images.csv', index=False)