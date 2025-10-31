##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma Models
##############################################################

# DUE TO PRIVACY REASONS I CANNOT PROVIDE RELEVANT FLO DATASETS THAT HAVE BEEN USED IN THIS PROJECT
###############################################################
# Business Problem
###############################################################
# FLO wants to set a roadmap for sales and marketing activities.
# In order for the company to make medium-long term plans, it is necessary to estimate
# the potential value that existing customers will provide to the company in the future.

###############################################################
# Dataset Story
###############################################################

# The dataset consists of information obtained from the past shopping behaviors of customers
# who made their last purchases as OmniChannel (shopping both online and offline) in 2020 - 2021.

# master_id: Unique customer number
# order_channel: Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel: Channel where the last purchase was made
# first_order_date: Date of the customer's first purchase
# last_order_date: Date of the customer's last purchase
# last_order_date_online: Date of the customer's last purchase on the online platform
# last_order_date_offline: Date of the customer's last purchase on the offline platform
# order_num_total_ever_online: Total number of purchases made by the customer on the online platform
# order_num_total_ever_offline: Total number of purchases made by the customer offline
# customer_value_total_ever_offline: Total amount paid by the customer in offline purchases
# customer_value_total_ever_online: Total amount paid by the customer in online purchases
# interested_in_categories_12: List of categories the customer shopped in the last 12 months


###############################################################
# TASKS
###############################################################
# TASK 1: Data Preparation
           # 1. Read the flo_data_20K.csv data and create a copy of the dataframe.
           # 2. Define the outlier_thresholds and replace_with_thresholds functions required to suppress outliers.
           # Note: When calculating cltv, frequency values must be integers. Therefore, round the lower and upper limits with round().
           # 3. If there are outliers in the variables "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online", suppress them.
           # 4. Omnichannel customers shop from both online and offline platforms. Create new variables for the total number of purchases and spending for each customer.
           # 5. Examine the variable types. Convert the type of variables expressing date to date.

# TASK 2: Creating CLTV Data Structure
           # 1. Take 2 days after the date of the last purchase in the dataset as the analysis date.
           # 2. Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
           # Monetary value will be expressed as the average value per purchase, and recency and tenure values will be expressed in weekly terms.


# TASK 3: Building BG/NBD and Gamma-Gamma Models, Calculating CLTV
           # 1. Fit the BG/NBD model.
                # a. Estimate the expected purchases from customers within 3 months and add it to the cltv dataframe as exp_sales_3_month.
                # b. Estimate the expected purchases from customers within 6 months and add it to the cltv dataframe as exp_sales_6_month.
           # 2. Fit the Gamma-Gamma model. Estimate the average value customers will leave and add it to the cltv dataframe as exp_average_value.
           # 3. Calculate 6-month CLTV and add it to the dataframe with the name cltv.
                # a. Standardize the cltv values you calculated and create a scaled_cltv variable.
                # b. Observe the top 20 people with the highest cltv value.

# TASK 4: Creating Segments According to CLTV
           # 1. Divide all your customers into 4 groups (segments) according to 6-month standardized CLTV and add the group names to the dataset. Add it to the dataframe with the name cltv_segment.
           # 2. Make brief 6-month action recommendations to management for 2 groups you will choose from among the 4 groups.

# TASK 5: Functionalize the entire process.


###############################################################
# TASK 1: Data Preparation
###############################################################


import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 1. Read the OmniChannel.csv data and create a copy of the dataframe.
df_ = pd.read_csv("/datasets/flo_data_20k.csv")
df = df_.copy()
df.head()
df.shape


# 2. Define the outlier_thresholds and replace_with_thresholds functions required to suppress outliers.
# Note: When calculating cltv, frequency values must be integers.
# Therefore, round the lower and upper limits with round().
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = round(low_limit, 0)
    dataframe.loc[dataframe[variable] > up_limit, variable] = round(up_limit, 0)
    print(f"{variable}: low_limit={low_limit:.2f}, up_limit={up_limit:.2f}")

# Suppress outliers (make it permanent)
columns = ["order_num_total_ever_online",
           "order_num_total_ever_offline",
           "customer_value_total_ever_offline",
           "customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df, col)
df.shape
# 4. Omnichannel customers shop from both online and offline platforms.
# Create new variables for the total number of purchases and spending for each customer.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

# 5. Examine the variable types. Convert the type of variables expressing date to date.
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

###############################################################
# TASK 2: Creating CLTV Data Structure
###############################################################

# 1. Take 2 days after the date of the last purchase in the dataset as the analysis date.
df["last_order_date"].max()  # 2021-05-30
analysis_date = df["last_order_date"].max() + pd.Timedelta(2, "D")
# analysis_date = dt.datetime(2021, 6, 1)
df.info()

# 2. Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = round(((df["last_order_date"] - df["first_order_date"]).dt.days) / 7)
cltv_df["T_weekly"] = round(((analysis_date - df["first_order_date"]).dt.days)/7)

#cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[ns]')) / 7
#cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[ns]')) / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]



"""cltv_df["recency_cltv_weekly"] = round(((df["last_order_date"] - df["first_order_date"]).dt.days) / 7)
cltv_df["T_weekly"] = round(((analysis_date - df["first_order_date"]).dt.days)/7)"""

cltv_df.head()

###############################################################
# TASK 3: Building BG/NBD and Gamma-Gamma Models, Calculating 6-month CLTV

# BG/NBD => Expected Number of Transaction
# Gamma-Gamma => Expected Average Profit
###############################################################
cltv_df['frequency'].min()
# 1. Build the BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Estimate the expected purchases from customers within 3 months and add it to the cltv dataframe as exp_sales_3_month.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])


# Estimate the expected purchases from customers within 6 months and add it to the cltv dataframe as exp_sales_6_month.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df.head()
# Examine the top 10 people who will make the most purchases in 3 and 6 months. Is there a difference?
cltv_df.sort_values("exp_sales_3_month", ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month", ascending=False)[:10]


# 2. Fit the Gamma-Gamma model. Estimate the average value customers will leave and add it to the cltv dataframe as exp_average_value.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()



# 3. Calculate 6-month CLTV and add it to the dataframe with the name cltv.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv
cltv_df.head()


# Observe the top 20 people with the highest CLTV value.
cltv_df.sort_values("cltv", ascending=False)[:20]

###############################################################
# TASK 4: Creating Segments According to CLTV
###############################################################

# 1. Divide all your customers into 4 groups (segments) according to 6-month standardized CLTV
# and add the group names to the dataset.
# Assign with the name cltv_segment.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# 2. Is it logical to divide customers into 4 groups according to CLTV scores? Should there be fewer or more? Comment.
cltv_df.groupby("cltv_segment").agg({"cltv": ["count", "mean", "std", "median"]})



cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 7, labels=["G","F","E" ,"D", "C", "B", "A"])
cltv_df.groupby("cltv_segment").agg({"cltv": ["count", "mean", "std", "median"]})


# 3. Make brief 6-month action recommendations to management for 2 groups you will choose from among the 5 groups.

#region 1. VIP Programme for High Value and Frequent Shoppers
"""1. VIP Programme for High Value and Frequent Shoppers

-- Only those in segment A are eligible.

-- Customers must be frequent shoppers.

-- Monetary value must be above average (more than 75% of the monetary value)"""

cltv_df["monetary_cltv_avg"].describe()
# 75%  182.4500

# In the 'A' segment, we select customers who shop frequently and spend more than 75 percent of the average
vip_customers = cltv_df[(cltv_df['cltv_segment'] == 'A') &
                   (cltv_df['frequency'] > cltv_df['frequency'].median()) &
                   (cltv_df['monetary_cltv_avg'] > 182.4500)]

# We save VIP customer ids to CSV file
vip_customers['customer_id'].to_csv('vip_program_customers.csv', index=False)

###############################################################
# BONUS: Functionalize the entire process.
###############################################################

def create_cltv_df(dataframe):

    # Data Preparation
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    # dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Creating CLTV data structure
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # Building BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # # Building Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # CLTV prediction
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentation
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)
cltv_df.head(10)