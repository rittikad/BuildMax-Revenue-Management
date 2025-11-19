# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:58:41 2025

@author: GROUP_1

Decision Variables:
x[i, j, t]: The number of rentals of equipment type j, with lease duration i, in time period t.
    - i: Lease durations (1 = 1-week, 2 = 4-weeks, 3 = 8-weeks, 4 = 16-weeks)
    - j: Equipment types (1 = Excavators, 2 = Cranes, 3 = Bulldozers)
    - t: Time periods (1 to 52 weeks)

    The decision variable `x[i, j, t]` represents the number of rental units for equipment type j (e.g., Excavators, Cranes, or Bulldozers) for a specific lease duration i (1-week, 4-weeks, 8-weeks, or 16-weeks) during the time period t (from 1 to 52 weeks).

Constraints:
1. Inventory Constraint: 
   sum(x[i, j, t] for i in lease_durations.keys()) <= inventory[j, t]
   This constraint ensures that the total number of rentals for equipment type j and time period t does not exceed the available inventory `I_jt[j, t]`.

2. Demand Constraint: 
   x[i, j, t] <= D_ijt[i, j, t]
   This constraint ensures that the number of rentals for each equipment type j, lease duration i, in time period t does not exceed the demand `D_ijt[i, j, t]`.

3. Inventory Update Constraint:
   inventory[j, t] == inventory[j, t - 1] + sum((x[i, j, t - durations_weeks[i]] if (t - durations_weeks[i]) >= 1 else 0 ) for i in lease_durations.keys()) - sum(x[i, j, t-1] for i in lease_durations.keys()) )
   This constraint ensures that inventory levels are updated each week based on the rentals and returns. The inventory at the start of week `t` is determined by the inventory at the start of week `t`, plus the units returned, minus the rentals made in week `t-1`.

4. Non-negativity Constraint:
   x[i, j, t] >= 0 for all i, j, t
   This constraint enforces that the number of rentals `x[i, j, t]` for each equipment type, lease duration, and time period must be non-negative (cannot rent negative units).

Objective Function:
Maximize Revenue = sum(P_ijt[i, j, t] * x[i, j, t] * T_i[i] for i in lease_durations.keys() for j in equipment_names.keys() for t in time_periods)
Where:
- `P_ijt[i, j, t]` is the price per day for equipment type j, lease duration i, in time period t.
- `T_i[i]` is the duration (in days) for lease duration i (1 = 7 days, 2 = 28 days, 3 = 56 days, 4 = 112 days).
    - Objective: The goal is to maximize total revenue by determining the optimal number of rentals `x[i, j, t]` for each equipment type, lease duration, and time period, while considering the price per day and lease duration.
"""

# Import necessary libraries
import numpy as np  # NumPy is used for numerical operations (though not directly used in this script, it's often included for array handling)
import pandas as pd  # Pandas is used for data manipulation and reading the Excel dataset
import matplotlib.pyplot as plt # Matplotlib is used for creating visualizations
import seaborn as sns # Seaborn is used for enhanced visualizations (e.g., bar plots)
from pyomo.environ import *  # Pyomo is used for formulating and solving optimization models

# Load Dataset from Excel File
# Define the name of the Excel file containing the dataset
file_name = 'IB9EO0_GROUP_1.xlsx'
df_data = pd.read_excel(file_name, sheet_name= 'BuildMax_Rentals_Dataset')  # Read the Excel file into a DataFrame

# Define Model Parameters and Decision Variables
# Map equipment type codes to human-readable names for easier interpretation in the model
equipment_names = {1: "Excavators", 2: "Cranes", 3: "Bulldozers"}

# Define the time periods as a list from 1 to 52 (representing weeks of the year)
time_periods = list(range(1, 53))

# Dictionary representing the lease durations and their corresponding values
# The key represents the lease type (1 for 1 week, 2 for 4 weeks, etc.), 
# and the value represents its duration in weeks.
T_i = {1: 7, 2: 28, 3: 56, 4: 112}

# Dictionary representing the lease durations and their corresponding values
lease_durations = {1: 1, 2: 4, 3: 8, 4: 16}

# Initial inventory for each equipment type (Excavators, Cranes, Bulldozers)
initial_inventory = {
    1: (df_data[f"Excavators - Start of Week Inventory"][0]),  # Excavators inventory
    2: (df_data[f"Cranes - Start of Week Inventory"][0]),      # Cranes inventory
    3: (df_data[f"Bulldozers - Start of Week Inventory"][0])    # Bulldozers inventory
}

# Extract Actual Revenue from the DataFrame for later comparison
actual_revenue = df_data.iloc[0, -1]  # The actual revenue is stored in the first row, last column

# Extract the actual revenue per equipment type (stored in subsequent rows for each equipment)
actual_revenue_per_equipment = {j: df_data.iloc[1 + j, -1] for j in equipment_names.keys()}

# Extract the purchase price for each equipment type
purchase_price = {j: df_data.iloc[5 + j, -1] for j in equipment_names.keys()}

# Define Pyomo Model
model = ConcreteModel()  # Instantiate a concrete optimization model

# Decision Variables:
# model.x[i, j, t] represents the number of rentals for lease type i, equipment j, and week t
model.x = Var(lease_durations.keys(), equipment_names.keys(), time_periods, domain=NonNegativeIntegers)

model.inventory = Var(equipment_names.keys(), time_periods, domain=NonNegativeIntegers)

# Price per equipment per duration and time period (P_ijt)
P_ijt = {}  # Initialize a dictionary to store price data

# Loop over time periods, equipment types, and lease durations to populate the price matrix
for t in time_periods:
    for j, eq_type in equipment_names.items():
        for i, duration in enumerate(["1-week", "4-weeks", "8-weeks", "16-weeks"], start=1):
            # Extract price per day for the corresponding equipment and lease duration for each week
            P_ijt[i, j, t] = df_data[f"{eq_type} - {duration} Price per day (£)"][t - 1]

# Demand data (D_ijt) - Represents the demand for each equipment type and lease duration
D_ijt = {}  # Initialize a dictionary to store demand data

for t in time_periods:
    for j, eq_type in equipment_names.items():
        for i, duration in enumerate(["1-week", "4-weeks", "8-weeks", "16-weeks"], start=1):
            # Extract demand data for each equipment type and lease duration
            D_ijt[i, j, t] = df_data[f"{eq_type} - {duration} Demand (units)"][t-1]
            

# Define Objective Function
# The objective is to maximize rental revenue, which is the price per unit (P_ijt) * number of rentals (model.x[i, j, t]) * number of days rented (T_i[i])
def revenue_objective(model):
    return sum(
        P_ijt[i, j, t] * model.x[i, j, t] * T_i[i] for i in lease_durations for j in equipment_names.keys() for t in time_periods)

# Set the objective function to maximize revenue
model.obj = Objective(rule=revenue_objective, sense=maximize)


# Define Constraints

# Demand Constraint: The number of rentals cannot exceed the demand for each equipment type and lease duration
def demand_constraint_rule(model, i, j, t):
    return model.x[i, j, t] <= D_ijt[i, j, t]

model.demand_constraints = Constraint(lease_durations.keys(), equipment_names.keys(), time_periods, rule=demand_constraint_rule)


# Dynamic Inventory Update Constraint: Update inventory based on previous rentals
# Define the inventory update constraint rule for the model
# This constraint updates the inventory based on leased equipment for each equipment type (j) and time period (t)
# For t = 1, it sets the inventory to the initial value.
# For subsequent periods, it adjusts the inventory based on leased equipment and previous period's inventory.
def inventory_update_constraint_rule(model, j, t):
    if t == 1:
        # For the first time period, set inventory to the initial value
        return model.inventory[j, t] == initial_inventory[j]
    else:
        # For other periods, update the inventory based on leases and previous inventory
        return (model.inventory[j, t] == model.inventory[j, t - 1]  
            + sum((model.x[i, j, t - lease_durations[i]] if (t - lease_durations[i]) >= 1 else 0 ) for i in lease_durations.keys()) 
            - sum(model.x[i, j, t-1] for i in lease_durations.keys()) )
    
# Apply the constraint to the model for all equipment types (j) and time periods (t)
model.inventory_update_constraint = Constraint(equipment_names.keys(), time_periods, rule=inventory_update_constraint_rule)


# Define the inventory constraint rule for the model
# This constraint ensures that the total leased equipment for each equipment type (j) 
# and time period (t) does not exceed the available inventory for that equipment at that time
def inventory_constraint_rule(model, j, t):
    return sum(model.x[i, j, t] for i in lease_durations.keys()) <= model.inventory[j,t]

# Apply the constraint to the model for all equipment types (j) and time periods (t)
model.inventory_constraint = Constraint(equipment_names.keys(), time_periods, rule=inventory_constraint_rule)


# Solve the Model Using a Solver
solver = SolverFactory('glpk')  # Choose the solver (GLPK is a common open-source solver)
results = solver.solve(model, tee=False)  # Solve the model without printing intermediate output

# Check if the solver was successful and load the solution if optimal
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    model.solutions.load_from(results)  # Load the optimal solution
else:
    print("Solve failed.")  # If the solver doesn't find an optimal solution, print a message

# Calculate Results

# Initialize dictionaries to store fleet utilization and unused inventory revenue loss
original_fleet_utilization = {}  
unused_inventory_revenue_loss = {}

# Loop through each equipment type (j) and its name (eq_type)
for j, eq_type in equipment_names.items():
    weekly_utilization = []  # List to store weekly utilization percentages
    total_revenue_loss = 0  # Variable to track total revenue loss due to unused inventory

    # Loop through each time period (t)
    for t in time_periods:
        # Get the inventory at the start of the week for the current equipment type
        inventory = df_data[f"{eq_type} - Start of Week Inventory"][t - 1]
        total_weekly_rentals = 0  # Variable to accumulate total rentals for the week

        # Loop through each lease duration (i)
        for i in lease_durations:
            # Map lease duration to a string (e.g., "1-week", "4-weeks")
            duration_str = {1: "1-week", 2: "4-weeks", 3: "8-weeks", 4: "16-weeks"}[i]
            accepted_col = f"{eq_type} - {duration_str} Accepted (units)"  # Column for accepted rentals
            price_col = f"{eq_type} - {duration_str} Price per day (£)"  # Column for price per day

            # Get rentals and price directly (will raise KeyError if missing)
            rentals = df_data[accepted_col][t - 1]
            total_weekly_rentals += rentals  # Add rentals to total weekly rentals

            price = df_data[price_col][t - 1]

            # Calculate unused inventory
            unused_inventory = inventory - rentals
            if unused_inventory > 0:
                days = T_i[i]  # Get the number of days for the lease duration
                revenue_loss = unused_inventory * price * days  # Calculate revenue loss due to unused inventory
                total_revenue_loss += revenue_loss  # Add to total revenue loss

        # Compute utilization for the week (percentage of rentals to inventory)
        if inventory > 0:
            utilization = (total_weekly_rentals / inventory) * 100
        else:
            utilization = 0  # If no inventory, utilization is 0
        weekly_utilization.append(utilization)

    # Compute 52-week average utilization
    average_utilization = sum(weekly_utilization) / len(weekly_utilization) if weekly_utilization else 0
    original_fleet_utilization[eq_type] = average_utilization  # Store average utilization
    unused_inventory_revenue_loss[eq_type] = total_revenue_loss  # Store total revenue loss for unused inventory



# Calculate the total optimized revenue for each equipment type using the decision variables
optimized_revenue_per_equipment = {
    j: sum(P_ijt[i, j, t] * value(model.x[i, j, t]) * T_i[i] for i in lease_durations.keys() for t in time_periods)
    for j in equipment_names.keys()
}

# Revenue Improvement per Equipment Type: Compare the optimized revenue with the actual revenue to see the improvement
revenue_improvement_per_equipment = {
    j: optimized_revenue_per_equipment[j] - actual_revenue_per_equipment[j]
    for j in equipment_names.keys()
}

# Revenue Improvement rate per Equipment Type: Calculate the percentage improvement in revenue per equipment type
revenue_improvement_rate_per_equipment = {
    j: (revenue_improvement_per_equipment[j] / actual_revenue_per_equipment[j]) * 100
    for j in equipment_names.keys()
}

# Actual ROI(Return on Investment) per Equipment Type: Calculate the Return on Investment for each equipment type
actual_roi_per_equipment = {
    j: ((actual_revenue_per_equipment[j] - purchase_price[j] * initial_inventory[j]) / (purchase_price[j] * initial_inventory[j])) * 100
    for j in equipment_names.keys()
}

# Optimised ROI(Return on Investment) per equipment type: Calculate the Return on Investment for each equipment type
optimized_roi_per_equipment = {
    j: ((optimized_revenue_per_equipment[j] - purchase_price[j] * initial_inventory[j]) / (purchase_price[j] * initial_inventory[j])) * 100
    for j in equipment_names.keys()
}


# Return On Investment improvement rate per equipment type: Calculate the percentage improvement in revenue per equipment type
overall_roi_per_equipment = {
    j: ((optimized_roi_per_equipment[j] - actual_roi_per_equipment[j]) / actual_roi_per_equipment[j]) * 100
    for j in equipment_names.keys()
}

# Overall ROI: The total improvement in revenue divided by the actual revenue
# Compute total initial investment cost
total_initial_investment = sum(purchase_price[j] * initial_inventory[j] for j in equipment_names)

# Calculate the total optimized revenue
optimized_revenue = value(model.obj)

# Revenue Improvement rate per Equipment Type: Calculate the percentage improvement in revenue per equipment type
revenue_improvement_rate = (optimized_revenue - actual_revenue) / actual_revenue * 100

# Calculate ROI metrics
roi_actual = ((actual_revenue - total_initial_investment) / total_initial_investment) * 100
roi_optimized = ((optimized_revenue - total_initial_investment) / total_initial_investment) * 100

overall_roi_change = (roi_optimized - roi_actual) / roi_actual * 100  # ROI improvement percentage


# Fleet Utilization: The ratio of rentals to available inventory
fleet_utilization = {
    equipment_names[j]: sum(
        (sum(value(model.x[i, j, t]) for i in lease_durations) / value(model.inventory[j, t])) 
        for t in time_periods if value(model.inventory[j, t]) > 0
    ) / len([t for t in time_periods if value(model.inventory[j, t]) > 0]) * 100
    for j in equipment_names.keys()
}


# Calculate the improved fleet utilization
improved_fleet_utilization = {}

for j, eq_type in equipment_names.items():
    current_utilization = fleet_utilization.get(eq_type, 0)  # Get current fleet utilization for the equipment type
    original_utilization = original_fleet_utilization.get(eq_type, 0)  # Get original fleet utilization

    if original_utilization > 0:
        # Calculate the improved utilization rate as a percentage increase/decrease
        improved_utilization = ((current_utilization - original_utilization) / original_utilization) * 100
    else:
        improved_utilization = 0  # If the original utilization is zero, no improvement possible
    
    improved_fleet_utilization[eq_type] = improved_utilization

# Revenue per Unit (RPU): Calculate the revenue per unit rented for each equipment type
rpu_per_equipment = {
    equipment_names[j]: optimized_revenue_per_equipment[j] / sum(value(model.x[i, j, t]) for i in lease_durations.keys() for t in time_periods)
    for j in equipment_names.keys()
}

# Lost Revenue Due to Rejected Rentals: Calculate lost revenue when demand cannot be met
lost_revenue = {
    equipment_names[j]: sum((D_ijt[i, j, t] - value(model.x[i, j, t])) * P_ijt[i, j, t] * T_i[i]
                            for i in lease_durations.keys() for t in time_periods if D_ijt[i, j, t] > value(model.x[i, j, t]))
    for j in equipment_names.keys()
}


# Visualize the results
# 1. Revenue Comparison (Actual vs Optimized)
# This section creates a side-by-side bar chart to compare actual vs optimized revenue for each equipment type.

plt.figure(figsize=(10,6))  # Set the figure size for the plot
bar_width = 0.3  # Width of each bar
index = range(len(equipment_names))  # X-axis positions for the bars, based on number of equipment types
fig, ax = plt.subplots(figsize=(10,6))  # Create a figure and axis object

# Create the first bar representing actual revenue for each equipment type (orange bars)
bar1 = ax.bar(index, list(actual_revenue_per_equipment.values()), bar_width, label="Actual Revenue", color='orange')

# Create the second bar representing optimized revenue for each equipment type (green bars)
bar2 = ax.bar([p + bar_width for p in index], list(optimized_revenue_per_equipment.values()), bar_width, label="Optimized Revenue", color='green')

# Label the x-axis with equipment names
ax.set_xlabel('Equipment Names')
# Label the y-axis with the revenue in GBP (£)
ax.set_ylabel('Revenue (£)')
# Set the title for the plot
ax.set_title('Revenue Comparison: Actual vs Optimized')
# Adjust the x-axis ticks to display equipment names centered between the bars
ax.set_xticks([p + bar_width / 2 for p in index])
ax.set_xticklabels(equipment_names.values())  # Set the equipment names as the x-tick labels
ax.legend()  # Add the legend to distinguish between actual and optimized revenue bars

# Adding the actual values as labels inside each bar
# This loop adds the actual value (in currency) at the top of each bar
for rect in bar1 + bar2:
    height = rect.get_height()  # Get the height of each bar (which corresponds to the revenue value)
    ax.annotate(f'£{height:,.2f}',  # Format the value as currency
                xy=(rect.get_x() + rect.get_width() / 2, height),  # Position the label at the center of each bar
                xytext=(0, 3),  # Offset the text vertically by 3 points
                textcoords="offset points",  # Specify that the offset is in points
                ha='center', va='bottom')  # Align the text horizontally and vertically

plt.show()  # Display the plot

# 2. Comparison of ROI (Actual vs Optimized)
plt.figure(figsize=(10,6))  
bar_width = 0.35  
index = range(len(equipment_names))  

fig, ax = plt.subplots(figsize=(10,6))  
bar1 = ax.bar(index, list(actual_roi_per_equipment.values()), bar_width, label="Actual ROI", color='lightcoral')  
bar2 = ax.bar([p + bar_width for p in index], list(optimized_roi_per_equipment.values()), bar_width, label="Optimized ROI", color='mediumseagreen')  

ax.set_xlabel('Equipment Names')  
ax.set_ylabel('ROI (%)')  
ax.set_title('Actual vs Optimized ROI per Equipment Type')  
ax.set_xticks([p + bar_width / 2 for p in index])  
ax.set_xticklabels(equipment_names.values())  
ax.legend()  

for rect in bar1 + bar2:  
    height = rect.get_height()  
    ax.annotate(f'{height:.2f}%',  
                xy=(rect.get_x() + rect.get_width() / 2, height),  
                xytext=(0, 3),  
                textcoords="offset points",  
                ha='center', va='bottom')  

plt.show()  

#3. Comparison of Fleet Utilization Rate: Actual vs Optimized
plt.figure(figsize=(10,6))  
bar_width = 0.35  
index = range(len(fleet_utilization))  

fig, ax = plt.subplots(figsize=(10,6))  
bar1 = ax.bar(index, list(original_fleet_utilization.values()), bar_width, label="Actual Fleet Utilization", color='lightblue')  
bar2 = ax.bar([p + bar_width for p in index], list(fleet_utilization.values()), bar_width, label="Optimized Fleet Utilization", color='yellowgreen')  

ax.set_xlabel('Equipment Names')  
ax.set_ylabel('Utilization Rate (%)')  
ax.set_title('Actual vs Optimized Fleet Utilization per Equipment Type')  
ax.set_xticks([p + bar_width / 2 for p in index])  
ax.set_xticklabels(equipment_names.values())  
ax.legend()  

for rect in bar1 + bar2:  
    height = rect.get_height()  
    ax.annotate(f'{height:.2f}%',  
                xy=(rect.get_x() + rect.get_width() / 2, height),  
                xytext=(0, 3),  
                textcoords="offset points",  
                ha='center', va='bottom')  

plt.show()  

#4. Comparison of Return On Investment: Actual vs Optimized
# Create the comparison bar chart
plt.figure(figsize=(8, 6))
categories = ['Actual ROI', 'Optimized ROI']
roi_values = [roi_actual, roi_optimized]

# Create the bar chart
plt.bar(categories, roi_values, color=['lightcoral', 'mediumseagreen'])

# Add title and labels
plt.title('Comparison of Actual ROI vs Optimized ROI')
plt.ylabel('ROI (%)')

# Annotate the bars with their values
for i, value in enumerate(roi_values):
    plt.text(i, value + 0.5, f'{value:.2f}%', ha='center', va='bottom')

# Show the graph
plt.show()


#5. Comparison of Revenue: Actual vs Optimized
# Create the comparison bar chart
plt.figure(figsize=(10, 6))
categories = ['Optimised Revenue', 'Actual Revenue']
revenues = [optimized_revenue, actual_revenue]

# Create the bar chart
plt.bar(categories, revenues, color=['mediumseagreen', 'lightcoral'])

# Add title and labels
plt.title('Comparison of Optimised vs Actual Revenue')
plt.ylabel('Revenue (£)')

# Annotate the bars with their values
for i, value in enumerate(revenues):
    plt.text(i, value + 1000000, f'£{value:,.2f}', ha='center', va='bottom')

# Show the graph
plt.show()


# Output the Results
# Print all the key results, such as fleet utilization, revenue per unit, lost revenue, and optimized revenue

print("\n====== All Key Results ======")

print("Original Fleet Utilization Rate:")
for k, v in original_fleet_utilization.items():
    print(f"{k}: {v:.2f}%")

print("\nUnused Inventory Revenue Loss:")
for k, v in unused_inventory_revenue_loss.items():
    print(f"{k}: £{v:,.2f}")


# Optimized Fleet Utilization Rate
print("\nOptimized Fleet Utilization Rate:")
for k, v in fleet_utilization.items():
    print(f"{k}: {v:.2f}%")
  
# Improved Fleet Utilization Rate
print("\nImproved Fleet Utilization Rate:")
for k, v in improved_fleet_utilization.items():
    print(f"{k}: {v:.2f}%")

# Revenue per Unit (RPU)
print("\nRevenue per Unit (RPU):")
for k, v in rpu_per_equipment.items():
    print(f"{k}: £{v:,.2f}")

# Lost Revenue Due to Rejected Rentals
print("\nLost Revenue Due to Rejected Rentals:")
for k, v in lost_revenue.items():
    print(f"{k}: £{v:,.2f}")

# Optimized Revenue Per Equipment Type
print("\nOptimized Revenue Per Equipment Type:")
for k, v in optimized_revenue_per_equipment.items():
    print(f"{equipment_names[k]}: £{v:,.2f}")
    
# Print revenue improvement per equipment type
print("\nRevenue Improvement Per Equipment Type:")
for j in equipment_names.keys():
    print(f"{equipment_names[j]}: £{revenue_improvement_per_equipment[j]:,.2f}")
    
# Print percentage revenue improvement per equipment type
print("\nRevenue Improvement Rate Per Equipment Type:")
for j in equipment_names.keys():
    print(f"{equipment_names[j]}: {revenue_improvement_rate_per_equipment[j]:.2f}%")
    
# Print actual ROI per equipment type
print("\nActual ROI Per Equipment Type:")
for j in equipment_names.keys():
    print(f"{equipment_names[j]}: {actual_roi_per_equipment[j]:.2f}%")

# Print optimized ROI per equipment type
print("\nOptimized ROI Per Equipment Type:")
for j in equipment_names.keys():
    print(f"{equipment_names[j]}: {optimized_roi_per_equipment[j]:.2f}%")

# Print ROI improvement per equipment type
print("\nOverall ROI Improvement Per Equipment Type:")
for j in equipment_names.keys():
    print(f"{equipment_names[j]}: {overall_roi_per_equipment[j]:.2f}%")

# Overall Revenue Performance
print("\nOverall Revenue Performance:")
print(f"Optimal rental revenue achieved: £{optimized_revenue:,.2f}")
print(f"Actual rental revenue: £{actual_revenue:,.2f}")
print(f"Revenue improvement: £{(optimized_revenue - actual_revenue):,.2f}")
print(f"Revenue improvement rate: {revenue_improvement_rate:.2f}%")
print(f"Actual ROI: {roi_actual:.2f}%")
print(f"Optimized ROI: {roi_optimized:.2f}%")
print(f"Overall ROI Improvement: {overall_roi_change:.2f}%")

