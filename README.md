# BuildMax Revenue Management Project

## Overview
BuildMax Rentals is a leading construction equipment rental company operating across Europe and North America.  
This project applies **Revenue Management (RM)** principles to optimise fleet utilisation, maximise revenue, and support strategic decision-making for one of BuildMax's UK branches.

**Business Challenge:**  
- Limited fleet and fixed capacity  
- Seasonal fluctuations in demand  
- High-value clients and corporate discounts  
- Operational constraints like one-way rentals, maintenance, and logistics  
- Need to balance short-term high-margin rentals with long-term contracts  

**Goal:** Develop a data-driven RM strategy using historical rental data to **increase revenue, improve ROI, and optimise fleet utilisation**.

---

## Objectives
1. Assess the suitability of Revenue Management for BuildMax Rentals.  
2. Formulate and implement a **Linear Programming model** to maximise revenue while respecting inventory and demand constraints.  
3. Analyse results to measure **revenue growth, ROI, and fleet utilisation improvements**.  
4. Recommend **practical RM strategies** and identify limitations of the model.  

---

## 🛠️ Skills & Tools Used

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Seaborn-4C8CBF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Analytics-000000?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Data%20Visualization-ff9800?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Business%20Insights-006400?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Revenue%20Management-8B0000?style=for-the-badge"/>
</p>

**Business & Analytical Skills**  
- Revenue Management strategy formulation  
- Fleet allocation and utilisation optimisation  
- ROI and revenue impact analysis  
- Dynamic pricing and market segmentation analysis  

**Soft/Professional Skills**  
- Stakeholder-focused reporting (executive summary, presentation)  
- Translating data into actionable business insights  
- Strategic decision support  

---

## Approach / Methodology

### 1. Data Analysis
- Explored historical weekly rental requests, pricing, and equipment returns.  
- Identified **seasonal demand trends** for Excavators, Cranes, and Bulldozers.  
- Evaluated **fleet utilisation rates** and calculated revenue potential from unused inventory.  

### 2. Revenue Optimisation Model
**Objective:** Maximise total rental revenue while optimising fleet allocation.  

**Decision Variables:**  
- 𝑥[i,j,t] = number of rentals for equipment type j, lease duration i, in week t  
- Lease durations: 1-week, 4-weeks, 8-weeks, 16-weeks  
- Equipment types: Excavators, Cranes, Bulldozers  

**Constraints:**  
1. **Inventory Limit:** Total rentals cannot exceed available fleet.  
2. **Demand Constraint:** Rentals cannot exceed predicted market demand.  
3. **Dynamic Inventory Management:** Accounts for equipment returns and ongoing rentals across weeks.  
4. **Non-Negativity & Integer Constraint:** Rentals must be non-negative integers.  

**Implementation:**  
- Model built using **Pyomo** in Python  
- Solved with **GLPK solver**  
- Inputs: historical demand, fleet inventory, and pricing per equipment type  

---

## Key Results & Insights

**1. Revenue & ROI Improvements**  
| Metric | Actual | Optimised | Improvement |
|--------|--------|-----------|-------------|
| Total Revenue (£) | 160,473,614 | 178,478,636 | +11.22% |
| ROI | 264.13% | 304.99% | +15.47% |
| Excavator Utilisation | 51.3% | 79.99% | +28.69% |
| Crane Utilisation | 41.34% | 68.95% | +27.61% |
| Bulldozer Utilisation | 34.97% | 49.85% | +14.88% |

- Cranes achieved the highest revenue growth (£7.69M), followed by Bulldozers (£5.73M) and Excavators (£4.59M).  
- Lost revenue due to unused inventory: £1.68M  
- Rejected rentals due to demand constraints: £3.03M  
- Seasonal patterns: Bulldozers peak in spring, Excavators peak in winter  

**2. Fleet Optimisation**  
- Original fleet utilisation was below 52% for all equipment types  
- Optimisation increased utilisation by **28–29% for Excavators and Cranes, 15% for Bulldozers**  

**3. Insights**  
- Revenue management can significantly improve revenue **without increasing fleet size**  
- Dynamic allocation prioritises high-value rentals and mitigates seasonal and operational fluctuations  
- Highlighted need for predictive maintenance and logistics optimisation to fully leverage RM  

---

## Recommendations
- Implement **dynamic pricing** based on seasonal demand and market trends  
- Optimise **long-term vs short-term lease allocations** for mining and oil sector clients  
- Maintain **stable pricing for government contracts** to ensure predictable revenue  
- Deploy **real-time fleet tracking and predictive maintenance systems**  
- Apply **corporate discount strategies** strategically to balance volume and revenue  

---

## Business Impact
- **Revenue Growth:** £160M → £178M (+11%)  
- **ROI Improvement:** 264% → 305% (+15%)  
- **Fleet Utilisation:** Significant increase across all equipment types  
- **Strategic Advantage:** Enhanced competitiveness through data-driven decisions  

---

## Quick Links
- [Code](https://github.com/rittikad/BuildMax-Revenue-Management/blob/main/Code.py) – Python scripts for analysis and optimisation  
- [Data](https://github.com/rittikad/BuildMax-Revenue-Management/blob/main/Data.xlsx) – Historical rental data and processed datasets  
- [Report](https://github.com/rittikad/BuildMax-Revenue-Management/blob/main/Report.pdf)  
- [Presentation](https://github.com/rittikad/BuildMax-Revenue-Management/blob/main/Presentation.pptx)
