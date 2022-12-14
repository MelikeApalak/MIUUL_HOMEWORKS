import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

df = pd.read_csv('recommendation_systems_project/armut_data.csv')
df.head()
df.shape
#df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]

"""df["CreateDate"]
df['CreateDate'] = pd.to_datetime(df['CreateDate'], format='%Y-%m-%d')
df['Month'] = df['CreateDate'].dt.month
df['Year'] = df['CreateDate'].dt.year
df['Year'].dtype
df["New_Date"] = df["Year"].astype(str) + "-" + df["Month"].astype(str)
df["SepetId"] = df["UserId"].astype(str) + "_" + df["New_Date"].astype(str)"""

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df.head()
df["SepetId"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head(20)
invoice_product_df = df.groupby(['SepetId','Hizmet']).count().unstack().fillna(0).applymap(lambda x:1 if x>0 else 0)
invoice_product_df.head()

#birliktelik kuralları

frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()
