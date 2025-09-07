import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_deliveries = 500
data = {
    'Delivery_ID': range(1, num_deliveries + 1),
    'Delivery_Time': pd.to_datetime(pd.date_range(start='2024-01-01', periods=num_deliveries, freq='H')),
    'Origin_Latitude': np.random.uniform(34, 35, num_deliveries),
    'Origin_Longitude': np.random.uniform(-118, -117, num_deliveries),
    'Destination_Latitude': np.random.uniform(34, 35, num_deliveries),
    'Destination_Longitude': np.random.uniform(-118, -117, num_deliveries),
    'Distance_km': np.random.uniform(5, 25, num_deliveries),
    'Congestion_Level': np.random.choice(['Low', 'Medium', 'High'], size=num_deliveries, p=[0.6, 0.3, 0.1]),
    'Delivery_Cost': np.random.uniform(10, 50, num_deliveries)
}
df = pd.DataFrame(data)
#Adding a realistic relationship between congestion and cost.
df['Delivery_Cost'] = df['Delivery_Cost'] + (df['Congestion_Level'].map({'Low':0, 'Medium':5, 'High':15}))
# --- 2. Data Cleaning and Feature Engineering ---
# (In a real-world scenario, this section would involve more extensive cleaning)
# --- 3. Analysis ---
# Analyze the relationship between congestion and delivery cost
average_cost_by_congestion = df.groupby('Congestion_Level')['Delivery_Cost'].mean()
print("Average Delivery Cost by Congestion Level:")
print(average_cost_by_congestion)
# --- 4. Visualization ---
plt.figure(figsize=(8, 6))
sns.barplot(x=average_cost_by_congestion.index, y=average_cost_by_congestion.values)
plt.title('Average Delivery Cost vs. Congestion Level')
plt.xlabel('Congestion Level')
plt.ylabel('Average Delivery Cost')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'delivery_cost_vs_congestion.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis could involve predictive modeling (linear regression, etc.) to predict delivery costs based on congestion and other factors.  This example focuses on a basic visualization.