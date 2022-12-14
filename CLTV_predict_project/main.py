import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)
pd.set_option('display.float_format',lambda x : '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler
df_ = pd.read_csv("CLTV_predict_project/flo_data_20k.csv")
df = df_.copy()
df.info()
df.describe().T
df.columns

def outlier_thresholds(dataframe,variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

def replace_with_thresholds(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]< low_limit),variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable]> up_limit), variable] = round(up_limit,0)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    print(col,check_outlier(df,col))
    replace_with_thresholds(df, col)

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] =df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

date_col = df.columns[df.columns.str.contains("date")]
date_col
df.info
df[date_col] = df[date_col].apply(pd.to_datetime)
df["last_order_date"].max()
analysis_date = dt.datetime(2021,6,1)

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((analysis_date -  df["first_order_date"]).dt.days) / 7
cltv_df["frequency"] = df["total_order"]
cltv_df["monetary_cltv_avg"] = df["total_value"] / df["total_order"]

cltv_df.head()


(analysis_date - df["first_order_date"]) / dt.timedelta(weeks=1)

#model
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# 3 aylık tahmin
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                cltv_df['frequency'],
                                cltv_df['recency_cltv_weekly'],
                                cltv_df['T_weekly'])
#6 aylık tahmin
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                cltv_df['frequency'],
                                cltv_df['recency_cltv_weekly'],
                                cltv_df['T_weekly'])

cltv_df.head()
#3 ve 6 aylık en çok satın alım yapanlar
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]
cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

#Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'],cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df.head()

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6, # ay
                                   freq="W", # T'nin frekans bilgisi
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

cltv_df.head()


# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df.sort_values("cltv",ascending=False)[:20]

#segment
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
cltv_df.groupby("cltv_segment").agg(["max","mean","count"])