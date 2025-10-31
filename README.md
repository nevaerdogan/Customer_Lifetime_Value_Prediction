💰 FLO Customer Lifetime Value (CLTV) Prediction
An advanced customer lifetime value prediction project using probabilistic models (BG-NBD and Gamma-Gamma)
to forecast future customer value and develop data-driven retention strategies.

Business Problem
FLO wants to establish a roadmap for sales and marketing activities. To enable the company to make medium-to-long-term plans, it is necessary to estimate the potential value that existing customers will provide to the company in the future.
Key Questions to Answer:

What is the expected revenue from each customer in the next 6 months?
Which customers are most valuable for long-term business?
How should we allocate marketing resources across customer segments?
Who should be targeted for VIP programs and retention campaigns?

What is CLTV?
Customer Lifetime Value (CLTV) is the total net profit a company expects to earn from a customer throughout their entire relationship.
Why CLTV Matters:

Resource Optimization: Focus marketing budget on high-value customers
Targeted Campaigns: Personalize strategies based on predicted value
VIP Identification: Recognize and nurture most valuable customers
Churn Prevention: Identify at-risk high-value customers early
ROI Maximization: Improve return on customer acquisition costs

Dataset
The dataset contains historical shopping behavior from 20,000 OmniChannel customers
who made purchases in 2020-2021.

Analysis Period: Last purchase date + 2 days (June 1, 2021)

Analysis Workflow
Data Preparation
    ↓
Feature Engineering
    ↓
Outlier Detection & Treatment
    ↓
CLTV Metrics Calculation
    ↓
BG-NBD Model (Purchase Frequency)
    ↓
Gamma-Gamma Model (Monetary Value)
    ↓
CLTV Prediction (6 months)
    ↓
Customer Segmentation
    ↓
Actionable Strategies
Key Steps

Data Preparation:
Outlier detection and suppression
Creating omnichannel metrics
Date conversion and validation


CLTV Data Structure:
recency_cltv_weekly: Weeks between first and last purchase
T_weekly: Customer age in weeks
frequency: Total number of purchases
monetary_cltv_avg: Average spending per transaction


Model Building:
BG-NBD for transaction prediction
Gamma-Gamma for monetary value estimation
Combined CLTV calculation


Segmentation:
4-tier customer classification (A, B, C, D)
Segment-specific strategies



Mathematical Models
1- BG-NBD (Beta Geometric/Negative Binomial Distribution)
Purpose: Predicts the number of future transactions
Assumptions:

Customers make purchases randomly around their transaction rate
Heterogeneity in transaction rates across customers
After each transaction, customers may become inactive with a certain probability

Output: Expected number of purchases in the next 3 and 6 months
Formula Components:

λ (lambda): Transaction rate
p (probability): Dropout probability
Gamma and Beta distributions for heterogeneity

2- Gamma-Gamma Model
Purpose: Estimates the average monetary value per transaction
Assumptions:

Monetary value of transactions varies randomly around customer's average
Average transaction values vary across customers
Monetary value is independent of purchase frequency

Output: Expected average profit per transaction
Why Gamma-Gamma?

Handles positive continuous values (transaction amounts)
Accounts for customer heterogeneity
Robust to outliers with proper penalization

3- CLTV Calculation
pythonCLTV = (Expected Number of Transactions) × (Expected Average Profit) × (Profit Margin)
For 6-month prediction:
pythonCLTV_6_months = BG-NBD_prediction(6_months) × Gamma-Gamma_avg_profit × profit_margin
Parameters:

time=6: 6-month prediction period
freq="W": Weekly frequency
discount_rate=0.01: 1% monthly discount rate

Project Structure
flo-cltv-prediction/
│
├── flo_cltv_prediction.py           # Main analysis script
├── README.md                         # Project documentation
│
├── data/
│   └── flo_data_20k.csv             # Customer data (not included)
│
└── outputs/
    └── vip_program_customers.csv    # High-value customers for VIP program



Neva Erdogan 🔗 www.linkedin.com/in/nevaerdogan
