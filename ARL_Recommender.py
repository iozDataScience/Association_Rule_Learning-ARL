# Dataset Story
# Online Retail II
# The data set named is the data set of a UK-based online store.
# Includes sales between 01/12/2009 and 09/12/2011.
# The product catalog of this company includes souvenirs. They can also be considered as promotional items.
# There is also information that most of its customers are wholesalers.

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
from helpers.helpers import retail_data_prep, check_df

df_ = pd.read_excel("dersler/hafta_3/online_retail_II.xlsx", sheet_name="Year 2010-2011") # read data
df = df_.copy()

# Task1 : Perform Data Preprocessing
# We do data preprocessing with our previously defined function.
df = retail_data_prep(df)

# Task 2: Generate association rules through Germany customers.

df_ger = df[df["Country"] == "Germany"]
check_df(df_ger)

# In order to create an association rule, we encode the data set as 1 and 0 as requested by apriori. We write the necessary function for this.

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

ger_inv_pro_df = create_invoice_product_df(df_ger, id=True)

# We determine the rules with the data we transform.
frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

################
# We define the function that finds the product name according to the id no information written, and find the names of the 3 products given in the assignment with this function.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

urun1_name = check_id(df_ger, 21987)    # ['PACK OF 6 SKULL PAPER CUPS']
urun2_name = check_id(df_ger, 23235)    # ['STORAGE TIN VINTAGE LEAF']
urun3_name = check_id(df_ger, 22747)    # ["POPPY'S PLAYHOUSE BATHROOM"]

#Task 4: Make a product recommendation for the users in the basket.
# We define the function that prepares the product list to be recommended for the products in the basket.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    recommendation_list = list(dict.fromkeys(recommendation_list))  #aynı ürünün mükerrer gelmesini engellemek için dublicate id leri siliyoruz
    return recommendation_list[0:rec_count]

recommend_product1 = arl_recommender(rules, 21987, 1)
recommend_product2 = arl_recommender(rules, 23235, 2)
recommend_product3 = arl_recommender(rules, 22747, 3)

# Task 5: What are the names of the proposed products?
#We define a function that browses the site and lists the product names according to the suggested product id list.

def get_itemname(rec_list, dataframe):
    recname = []
    for i in range(len(rec_list)):
        a = (check_id(dataframe,rec_list[i]))
        recname.append(a)
    return recname

suggest1 = get_itemname(recommend_product1, df)
suggest2 = get_itemname(recommend_product2, df)
suggest3 = get_itemname(recommend_product3, df)
