#FLO DATASET CLTV DOSYASI

#-----KÜTÜPHANE VE AYARLAR-------#
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None


#-----GEREKLİ FONKSİYONLAR-------#
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0) #loc etikete göre işlem yapar
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


#-----VERİ ÖN HAZIRLIK-------#
df_=pd.read_csv("flo_data_20k.csv")
df=df_.copy()

df.head()
df.columns
df.shape
df.describe().T
df.isnull().sum() #boş değer yok(not null)
df.dtypes #sütunların tipleri hakkında bilgi
df.info() #tarih bilgileri object olarak kullanılmış


#-----AYKIRI DEĞER BASKILAMA-------#
columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)


#-----VERİ HAZIRLAMA-------#
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)


#-----YENİ DATAFRAME HAZIRLAMA-------#
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()


#-----BG-NBD MODELİNİN HAZIRLANMASI(İLERİDEKİ TAHMİNİ SATIN ALMA SAYISINI MODELLER)-------#
bgf=BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df.sort_values("exp_sales_3_month",ascending=False).head(10)
cltv_df.sort_values("exp_sales_6_month",ascending=False).head(10)


#-----GAMMA-GAMMA MODELİNİN HAZIRLANMASI(AVERAGE PROFİT MODELLER,ORTALAMA BEKLENEN KAR)-------#
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])


#-----BG-NBD ve GAMA-GAMA MODELİ ile CLTV'NİN HESAPLANMASI-------#
cltv=ggf.customer_lifetime_value(bgf,
                            cltv_df["frequency"],
                            cltv_df["recency_cltv_weekly"],
                            cltv_df["T_weekly"],
                            cltv_df["monetary_cltv_avg"],
                            time=6,
                            discount_rate=0.01,
                            freq="W")

cltv_df["cltv"]=cltv
cltv_df.sort_values("cltv",ascending=False)[:20]


#-----CLTV'YE GÖRE SEGMENTLERİN OLUŞTURULMASI-------#
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head(20)