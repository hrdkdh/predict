import os
from statistics import mean
import time
import json
import shutil
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
import requests as requests
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split as train_test_split
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler

# warnings.filterwarnings(action="ignore") #판다스 워닝 끄기

__all__ = [
    "Crawler",
    "DataPreprocessor",
    "ModelMaker",
    "Predictor"
]

class Crawler():
    def __init__(self, crawl_page_max=50, perPage=100):
        """
        crawl_page_max : 몇 페이지까지 수집할 것인가
        perPage : 한 페이지에 몆 건까지 출력할 것인가
        """
        self.df = None
        self.save_file_name = "crawled_data.xlsx"
        self.df_kospi = None
        self.df_investor = None
        self.crawl_page_max = crawl_page_max
        self.info_for_crawl = {
            "CR" : {
                "url" : "https://finance.daum.net/api/exchanges/FRX.KRWUSD/days",
                "param" : {
                    "symbolCode": "FRX.KRWUSD",
                    "terms": "days",
                    "perPage" : perPage
                },
                "map_for_df" : [
                    {"col" : "date", "item_key" : "date" },
                    {"col" : "CR", "item_key" : "basePrice" }
                ]
            },
            "NASDAQ" : {
                "url" : "https://finance.daum.net/api/quote/US.COMP/days",
                "param" : {
                    "symbolCode": "US.COMP",
                    "pagination": "true",
                    "perPage" : perPage
                },
                "map_for_df" : [
                    {"col" : "date", "item_key" : "date" },
                    {"col" : "NASDAQ", "item_key" : "tradePrice" }
                ]
            },
            "DOW" : {
                "url" : "https://finance.daum.net/api/quote/US.DJI/days",
                "param" : {
                    "symbolCode": "US.DJI",
                    "pagination": "true",
                    "perPage" : perPage
                },
                "map_for_df" : [
                    {"col" : "date", "item_key" : "date" },
                    {"col" : "DOW", "item_key" : "tradePrice" }
                ]
            },
            "NIKKEI" : {
                "url" : "https://finance.daum.net/api/quote/JP.NI225/days",
                "param" : {
                    "symbolCode": "JP.NI225",
                    "pagination": "true",
                    "perPage" : perPage
                },
                "map_for_df" : [
                    {"col" : "date", "item_key" : "date" },
                    {"col" : "NIKKEI", "item_key" : "tradePrice" }
                ]
            },
            "SHANGHAI" : {
                "url" : "https://finance.daum.net/api/quote/CN000003/days",
                "param" : {
                    "symbolCode": "CN000003",
                    "pagination": "true",
                    "perPage" : perPage
                },
                "map_for_df" : [
                    {"col" : "date", "item_key" : "date" },
                    {"col" : "SHANGHAI", "item_key" : "tradePrice" }
                ]
            },
            "INDI" : {
                "url" : "https://finance.daum.net/api/investor/KOSPI/days",
                "param" : {
                    "market": "KOSPI",
                    "perPage" : perPage,
                    "fieldName": "changeRate",
                    "order": "desc",
                    "details": "true",
                    "pagination": "true",
                },
                "map_for_df" : [
                    {"col" : "date", "item_key" : "date" },
                    {"col" : "INDI", "item_key" : "individualStraightPurchasePrice" },
                    {"col" : "FOREIGN", "item_key" : "foreignStraightPurchasePrice" },
                    {"col" : "ORG", "item_key" : "institutionStraightPurchasePrice" }
                ]
            },
            "GOLD" : {
                "url" : "https://finance.daum.net/api/domestic/exchanges/COMMODITY-GOLD/days",
                "param" : {
                    "symbolCode": "COMMODITY-GOLD",
                    "perPage": perPage,
                    "fieldName": "changeRate",
                    "order": "desc",
                    "pagination": "true"
                },
                "map_for_df" : [
                    {"col" : "date", "item_key" : "date" },
                    {"col" : "GOLD", "item_key" : "internationalGoldPrice" }
                ]
            },
            "WTI" : {
                "url" : "https://finance.daum.net/api/domestic/exchanges/COMMODITY-/CLc1/days",
                "param" : {
                    "symbolCode": "COMMODITY-/CLc1",
                    "perPage": perPage,
                    "fieldName": "changeRate",
                    "order": "desc",
                    "pagination": "true"
                },
                "map_for_df" : [
                    {"col" : "date", "item_key" : "date" },
                    {"col" : "WTI", "item_key" : "tradePrice" }
                ]
            }
        }

    def crawlDataKOSPI(self):
        now_ymd = datetime.now().strftime("%Y%m%d")
        header = {
            "referer": "http://data.krx.co.kr",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
        }
        url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        form_data = {
            "bld": "dbms/MDC/STAT/standard/MDCSTAT00301",
            "locale": "ko_KR",
            "tboxindIdx_finder_equidx0_1": "코스피",
            "indIdx": "1",
            "indIdx2": "001",
            "codeNmindIdx_finder_equidx0_1": "코스피",
            "param1indIdx_finder_equidx0_1": "",
            "strtDd": "20010102",
            "endDd": now_ymd,
            "share": "2",
            "money": "3",
            "csvxls_isNo": "false",
        }
        res = requests.post(url, data=form_data, headers=header)
        json_parsed = res.json()
        result = []
        for item in json_parsed["output"]:
            this_dic = {
                "date" : item["TRD_DD"].replace("/" ,"-"),
                "KOSPI" : float(item["CLSPRC_IDX"].replace(",", "")),
                "KOSPI_START" : float(item["OPNPRC_IDX"].replace(",", "")),
                "KOSPI_HIGH" : float(item["HGPRC_IDX"].replace(",", "")),
                "KOSPI_LOW" : float(item["LWPRC_IDX"].replace(",", "")),
                "KOSPI_TRADE_VOL" : float(item["ACC_TRDVOL"].replace(",", "")),
                "KOSPI_TRADE_VAL" : float(item["ACC_TRDVAL"].replace(",", "")),
                "KOSPI_HIGH_LOW_GAP" : float(item["LWPRC_IDX"].replace(",", "")) - float(item["HGPRC_IDX"].replace(",", ""))
            }
            result.append(this_dic)
        this_df = pd.DataFrame(result)
        return this_df

    def crawlData(self, want_data_names, save=False):
        header = {
            "referer": "https://finance.daum.net",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
        }
        if type(want_data_names) != list:
            want_data_names = [want_data_names]
        
        for want_data_name in want_data_names:
            want_data_name_for_print = want_data_name
            if want_data_name in ["FOREIGN", "ORG"]:
                want_data_name = "INDI"
            elif want_data_name in ["KOSPI", "KOSPI_START", "KOSPI_TRADE_VOL", "KOSPI_TRADE_VAL", "KOSPI_HIGH", "KOSPI_LOW", "KOSPI_HIGH_LOW_GAP"]:
                print("\n{} : 데이터 수집중...".format(want_data_name_for_print), end="\r")
                if self.df is None or want_data_name not in self.df.columns.to_list():
                    this_df = self.crawlDataKOSPI()
                    if self.df is None:
                        self.df = this_df
                    else:
                        self.df = self.df.merge(this_df, how="left", on="date")
                continue

            result = []
            url = self.info_for_crawl[want_data_name]["url"]
            param = self.info_for_crawl[want_data_name]["param"]
            for page_no in range(1, self.crawl_page_max+1):
                print_prefix = ""
                if page_no == 1:
                    print_prefix = "\n"

                print("{}{} : {}번째 페이지 데이터 수집중...".format(print_prefix, want_data_name_for_print, page_no), end="\r")

                if want_data_name in ["INDI", "FOREIGN", "ORG"] and want_data_name not in self.df.columns.to_list():
                    continue
                
                param["page"] = page_no

                req_done = False
                retry_cnt = 0
                while req_done is False:
                    try:
                        res = requests.get(url, headers=header, params=param)
                        json_parsed = res.json()
                    except:
                        retry_cnt += 1
                        print("{}{} : {}번째 페이지 데이터 수집중...(에러로 재시도{})".format(print_prefix, want_data_name_for_print, page_no, retry_cnt), end="\r")
                        sec = random.randrange(1, 5)
                        time.sleep(sec) #대기 후 재시도
                    else:
                        req_done = True

                for item in json_parsed["data"]:
                    this_dic = {}
                    for map in self.info_for_crawl[want_data_name]["map_for_df"]:
                        this_value = item[map["item_key"]]
                        if map["col"] == "date":
                            this_value = this_value[:10]
                        else:
                            this_value = float(this_value)
                        this_dic[map["col"]] = this_value
                    result.append(this_dic)
            if want_data_name in ["INDI", "FOREIGN", "ORG"] and want_data_name not in self.df.columns.to_list():
                this_df = pd.DataFrame(result)
                if self.df is None:
                    self.df = this_df
                else:
                    self.df = self.df.merge(this_df, how="left", on="date")
            else:
                this_df = pd.DataFrame(result)
                if want_data_name in ["NASDAQ", "DOW"]: #미국 지수는 하루 전으로 일자를 잡아야 합리적임. 새벽에 결과가 나오는데다 당일의 경우 실시간 값을 출력하기 때문에 정확하지 않으므로...
                    new_data = []
                    for i in range(0, len(this_df)):
                        if i+1 == len(this_df):
                            new_data.append(np.nan)
                        else:
                            new_data.append(this_df.loc[i+1, want_data_name])
                    this_df[want_data_name] = new_data
                if self.df is None:
                    self.df = this_df
                else:
                    self.df = self.df.merge(this_df, how="left", on="date")

        remove_col = []
        for col in self.df.columns.to_list():
            if col not in want_data_names:
                remove_col.append(col)
        remove_col.remove("date")
        self.df.drop(columns=remove_col, inplace=True)

        if save is True:
            self.df.to_excel(self.save_file_name)
        return self.df

    def loadFromSavedFile(self, cols):
        df = pd.read_excel(self.save_file_name, index_col=0)
        selected_cols = cols.copy()
        selected_cols = ["date"] + cols
        self.df = df.loc[:, selected_cols]
        return self.df

    def removeNan(self): #결측치를 앞뒤 일자의 평균값으로 대체
        #1. 제일 첫번째 행에 NaN 값이 하나라도 있다면 제거한다.
        df_nan_removed = self.df.copy()
        for item in self.df.iterrows():
            idx = item[0]
            if True in [v for k, v in self.df.iloc[idx].isnull().items()]:
                df_nan_removed.drop(index=idx, inplace=True)
            else:
                break

        #2. 제일 마지막 행에 NaN 값이 하나라도 있다면 제거한다.
        for i in range(1, len(self.df)):
            idx = len(self.df) - i
            if True in [v for k, v in self.df.iloc[idx].isnull().items()]:
                df_nan_removed.drop(index=idx, inplace=True)
            else:
                break

        #3. 루프를 돌면서 결측치를 평균값으로 대체한다.
        df_nan_removed.reset_index(inplace=True, drop=True)
        for item in df_nan_removed.iterrows():
            idx = item[0]
            if True in [v for k, v in item[1].isnull().items()]:
                nan_col_list = [k for k, v in item[1].isnull().items() if v is True]
                for nan_col in nan_col_list:
                    nan_idx = 1
                    prev_day_price = df_nan_removed.iloc[idx+nan_idx][nan_col]
                    while True:
                        if str(prev_day_price) == "nan":
                            nan_idx += 1
                            prev_day_price = df_nan_removed.iloc[idx+nan_idx][nan_col]
                        else:
                            break

                    next_day_price = df_nan_removed.iloc[idx-1][nan_col]
                    while True:
                        if str(next_day_price) == "nan":
                            nan_idx += 1
                            next_day_price = df_nan_removed.iloc[idx-nan_idx][nan_col]
                        else:
                            break
                    average = (prev_day_price + next_day_price)/2
                    df_nan_removed.loc[idx, nan_col] = average

        for item in df_nan_removed.iterrows():
            idx = item[0]
            if True in [v for k, v in item[1].isnull().items()]:
                print(item[1])
        return df_nan_removed

class DataPreprocessor():
    def __init__(self, df_crawled, cols, scale_method="minmax", model_save_path="saved_model"):
        self.model_save_path = model_save_path
        self.df = df_crawled.copy()
        self.cols = cols
        self.scale_method = scale_method
        self.scale_info = {}
    
    def getOutlierDf(self, df):
        df_outlier = df.copy()
        df_outlier["_outlier"] = 0
        df_outlier["_outlier_from"] = ""
        df_outlier["_over"] = 0
        df_outlier["_under"] = 0
        
        for col in self.cols:
            q1 = df[col].quantile(.25)
            q3 = df[col].quantile(.75)
            iqr = q3-q1
            iqr_under = q1 - (iqr*1.5)
            iqr_over = q3 + (iqr*1.5)
            df_outlier.loc[(df[col] < iqr_under), "_outlier"] = 1
            df_outlier.loc[(df[col] < iqr_under), "_under"] = 1
            df_outlier.loc[(df[col] < iqr_under), "_outlier_from"] = df_outlier.loc[(df[col] < iqr_under), "_outlier_from"].values + col + "(under), "

            df_outlier.loc[(df[col] > iqr_over), "_outlier"] = 1
            df_outlier.loc[(df[col] > iqr_over), "_over"] = 1
            df_outlier.loc[(df[col] > iqr_over), "_outlier_from"] = df_outlier.loc[(self.df[col] > iqr_over), "_outlier_from"].values + col + "(over), "
        df_outlier.drop(index=df_outlier[(df_outlier["_outlier"] == 0)].index, inplace=True)
        return df_outlier

    def removeOutlier(self):
        df = self.df.copy()
        df_outlier = self.getOutlierDf(df)
        df.drop(index = df_outlier.index.to_list(), inplace=True)
        df.reset_index(inplace = True, drop=True)
        self.df = df

    def sortByDate(self):
        self.df.sort_values(by="date", inplace = True)
        self.df.reset_index(inplace = True, drop=True)
    
    def norm(self, df_org):
        df = df_org.copy()
        df = (df - df.mean()) / df.std()
        return df, df_org.mean(), df_org.std()

    def makeDiffRatio(self):
        for col in self.cols:
            self.df["X_{}_DIFF".format(col)] = 0.0
            for idx in range(0, len(self.df)): #각 행을 돌면서
                if idx > 0:
                    self.df.loc[idx, "X_{}_DIFF".format(col)] = self.df.loc[idx, "{}".format(col)] / self.df.loc[idx-1, "{}".format(col)]

    def makeDiffByRange(self, minAR=2, maxAR=11):
        maxAR += 1
        for idx, col in enumerate(self.cols):
            print("차분 생성중... ({:.2f}%)".format(100*(idx+1)/len(self.cols)), end="\r")
            for before_day in range(minAR, maxAR):
                result_list = []
                this_col_name = "X_{}_DIFF{}".format(col, before_day)
                for idx in range(0, len(self.df)): #각 행을 돌면서
                    before_day_idx = idx - before_day
                    if before_day_idx < 0:
                        this_value = np.nan
                    else:
                        this_value = self.df.loc[idx, "{}".format(col)] - self.df.loc[before_day_idx, "{}".format(col)]
                    result_list.append(this_value)
                result_df = pd.DataFrame({ this_col_name : result_list })
                self.df = pd.concat([self.df, result_df], axis=1)
        print("\n")

    def makeAR(self, minAR=2, maxAR=11): 
        # maxAR += 1
        for idx, col in enumerate(self.cols):
            print("자기상관 생성중... ({:.2f}%)".format(100*(idx+1)/len(self.cols)), end="\r")
            for before_day in range(minAR, maxAR):
                result_list = []
                # result_diff_list = []
                this_col_name = "X_{}_AR{}".format(col, before_day)
                # this_diff_col_name = "X_{}_DIFF_AR{}".format(col, before_day)
                for idx in range(0, len(self.df)): #각 행을 돌면서
                    before_day_idx = idx - before_day
                    if before_day_idx < 0:
                        this_value = np.nan
                        # this_diff_value = np.nan
                    else:
                        this_value = self.df.loc[before_day_idx, "{}".format(col)]
                        # this_diff_value = self.df.loc[before_day_idx, "X_{}_DIFF".format(col)]
                    result_list.append(this_value)
                    # result_diff_list.append(this_diff_value)
                result_df = pd.DataFrame({ this_col_name : result_list })
                # result_diff_df = pd.DataFrame({ this_diff_col_name : result_list })
                self.df = pd.concat([self.df, result_df], axis=1)
                # self.df = pd.concat([self.df, result_diff_df], axis=1)
        print("\n")

    def makeMA(self, minMA=1, maxMA=11):
        maxMA += 2
        pb_max_cnt = len(self.cols) * (maxMA - minMA)
        pb_now_cnt = 0
        for col in self.cols:
            for mean_period in range(minMA, maxMA):
                pb_now_cnt += 1
                print("이동평균 생성중... ({:.2f}%)".format((100*pb_now_cnt)/pb_max_cnt), end="\r")

                result_list = []
                this_col_name = "X_{}_MA{}".format(col, mean_period)
                for idx3 in range(0, len(self.df)): #각 행을 돌면서
                    if idx3 < mean_period:
                        this_value = 0.0
                    else:
                        this_sum = 0
                        this_cnt = 0
                        for before_day in range(-mean_period, 0): #이전 일자의 행들을 돌면서
                            before_idx = idx3 + before_day
                            this_sum += self.df.loc[before_idx, "{}".format(col)]
                            this_cnt += 1
                        this_value = this_sum/this_cnt
                    result_list.append(this_value)
                result_df = pd.DataFrame({ this_col_name : result_list })
                self.df = pd.concat([self.df, result_df], axis=1)                
                
                # result_list = []
                # this_col_name = "X_{}_DIFF_MA{}".format(col, mean_period)
                # for idx3 in range(0, len(self.df)): #각 행을 돌면서
                #     if idx3 < mean_period:
                #         this_value = 0.0
                #     else:
                #         this_sum = 0
                #         this_cnt = 0
                #         for before_day in range(-mean_period, 0): #이전 일자의 행들을 돌면서
                #             before_idx = idx3 + before_day
                #             this_sum += self.df.loc[before_idx, "X_{}_DIFF".format(col)]
                #             this_cnt += 1
                #         this_value = this_sum/this_cnt
                #     result_list.append(this_value)
                # result_df = pd.DataFrame({ this_col_name : result_list })
                # self.df = pd.concat([self.df, result_df], axis=1)
        print("\n")

    def makeTargetYs(self, next_day_len):
        for idx, day in enumerate(range(1, next_day_len+1)):
            print("라벨 생성중... ({:.2f}%)".format(100*(idx+1)/next_day_len), end="\r")
            result_list = []
            this_col_name = "Y_KOSPI_nextday_{}".format(day)
            for idx in range(0, len(self.df)): #각 행을 돌면서
                if idx+day >= len(self.df):
                    this_value = 0.0
                else:
                    this_value = self.df.loc[idx+day, "KOSPI"]
                result_list.append(this_value)
            result_df = pd.DataFrame({ this_col_name : result_list})
            self.df = pd.concat([self.df, result_df], axis=1)

        self.y_list = [x for x in self.df.columns if x[:2] == "Y_"]

    def cutoffData(self, cutoff_len_head, cutoff_len_tail):
        """
        AR과 MA로 인해 초기 0값을 가지고 있는 행들을 제거한다.
        """
        if cutoff_len_tail == 0:
            self.df = self.df.iloc[cutoff_len_head+1:]
        else:
            self.df = self.df.iloc[cutoff_len_head+1:-cutoff_len_tail]
        self.df.reset_index(inplace=True, drop=True)

    def scalingForModeling(self):
        self.scale_info["how"] = self.scale_method
        if self.scale_method == "minmax":
            scaler = MinMaxScaler()
            for col in self.cols:
                scaler.fit(self.df.loc[:, [col]])
                self.df[col] = scaler.transform(self.df.loc[:, [col]])
                self.scale_info[col] = {"min" : scaler.data_min_[0], "max" : scaler.data_max_[0]}
        elif self.scale_method == "norm":
            for col in self.cols:
                self.df[col], mean, std = self.norm(self.df.loc[:, [col]])
                self.scale_info[col] = {"mean" : mean[col], "std" : std[col]}
        scale_info_str = json.dumps(self.scale_info)
        open("kospi_predictor_model/{}/scale_info.txt".format(self.model_save_path), "w").write(scale_info_str)

    def scalingForPredict(self):
        scale_info_str = open("kospi_predictor_model/{}/scale_info.txt".format(self.model_save_path), "r").readlines()[0]
        scale_info = json.loads(scale_info_str)
        if self.scale_method == "minmax":
            for col in self.cols:
                min = scale_info[col]["min"]
                max = scale_info[col]["max"]
                self.df[col] = (self.df.loc[:, [col]] - min) / (max - min)
        elif self.scale_method == "norm":
            for col in self.cols:
                mean = scale_info[col]["mean"]
                std = scale_info[col]["std"]
                self.df[col] = (self.df.loc[:, [col]] - mean) / std

    def getRemoveColByCorr(self, threshold= 0.7):
        corr = []
        for col in [x for x in self.df.columns.to_list() if "X_" in x]:
            corr.append({
                "X" : col,
                "corr" : self.df[["Y_KOSPI_nextday_1", col]].corr().iloc[0, 1]
            })
        self.df_corr = pd.DataFrame(corr)
        self.df_corr.sort_values(by=["corr"], ascending=False, inplace=True)
        self.df_corr.reset_index(inplace=True, drop=True)
        self.remove_cols = self.df_corr.loc[(abs(self.df_corr["corr"]) < threshold), "X"].to_list()

    def getRemoveColByRegression(self, threshold=0.7):
        reg = []
        y = [x for x in self.df.columns.to_list() if "Y_" in x][0]
        for col in [x for x in self.df.columns.to_list() if "X_" in x]:
            res = ols("{} ~ {}".format(y, col), data=self.df).fit()
            reg.append({
                "X" : col,
                "rsquared" : res.rsquared
            })
        self.df_reg = pd.DataFrame(reg)
        self.df_reg.sort_values(by=["rsquared"], ascending=False, inplace=True)
        self.df_reg.reset_index(inplace=True, drop=True)
        self.remove_cols = self.df_reg.loc[(abs(self.df_reg["rsquared"]) < threshold), "X"].to_list()

    def splitData(self, test_size=0.2, train_size=0.8):
        df_splited = train_test_split(self.df, test_size=test_size, train_size=train_size, random_state=8699)
        self.df_train = df_splited[0]
        self.df_test = df_splited[1]

class ModelMaker():
    def __init__(self, y_cols, df_train, df_test, perceptron_vol=None, dense_vol1=5, dense_vol2=5, activation_method="relu", optimizer_str="RMSprop", model_save_path="saved_model"):
        self.y_cols = y_cols
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.perceptron_vol = perceptron_vol
        self.dense_vol1 = dense_vol1
        self.dense_vol2 = dense_vol2
        self.activation_method = activation_method
        self.optimizer = optimizer_str
        self.model_save_path = model_save_path
        self.x_cols = [x for x in self.df_train.columns if x[:2] == "X_"]

    def _constructModel(self, learning_rate):
        if self.perceptron_vol is None:
            self.perceptron_vol = len(self.x_cols)*6
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=(1, len(self.x_cols))))
        # self.model.add(tf.keras.layers.BatchNormalization())
        # self.model.add(tf.keras.layers.Dropout(rate=0.2))
        for i in range(self.dense_vol1):
            self.model.add(tf.keras.layers.Dense(self.perceptron_vol, activation=self.activation_method))
        self.model.add(tf.keras.layers.Dense(len(self.y_cols)))
        for i in range(self.dense_vol2):
            self.model.add(tf.keras.layers.Dense(self.perceptron_vol, activation=self.activation_method))
        self.model.add(tf.keras.layers.Dense(len(self.y_cols)))

        if self.optimizer == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif self.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        self.model.summary()

    def _makeTensorboardDir(self):
        for folder in os.listdir("kospi_predictor_model/logs/"):
            if "." not in folder:
                shutil.rmtree("kospi_predictor_model/logs/"+folder)
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir_name = "kospi_predictor_model/logs/{}".format(now)
        os.makedirs(log_dir_name)
        self.log_dir_name = log_dir_name

    def makeModel(self, EPOCHS=200, batch_size=64, learning_rate=0.001, save_by_checkpoint=True):
        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None, EPOCHS=EPOCHS):
                percent = round(((epoch+1)/EPOCHS)*100)
                print("EPOCH : {}/{}({}%) loss : {:.4f}, mae : {:.4f}, mse : {:.4f} / val_loss : {:.4f}, val_mae : {:.4f}, val_mse : {:.4f}{}"
                .format(epoch+1, EPOCHS, percent, 
                        logs["loss"], logs["mae"], logs["mse"], logs["val_loss"], logs["val_mae"], logs["val_mse"], " "*30),
                        end="\r")

        X_train = self.df_train[self.x_cols].to_numpy().reshape(-1, 1, len(self.x_cols))
        y_train = self.df_train[self.y_cols].to_numpy()
        
        self.checkpoint_folder = "kospi_predictor_model/checkpoint/"
        for file in os.listdir(self.checkpoint_folder):
            os.remove(self.checkpoint_folder+file)
        self.checkpoint_path = self.checkpoint_folder + "cp-{epoch:04d}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path, 
            verbose=0, 
            monitor="val_loss",
            save_weights_only=True,
            save_best_only = True,
            save_freq="epoch"
        )
        
        self._makeTensorboardDir()
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir_name)

        self._constructModel(learning_rate=learning_rate)

        self.history = self.model.fit(
            X_train, y_train,
            batch_size = batch_size,
            callbacks=[cp_callback, CustomCallback(), tb_callback],
            epochs=EPOCHS, validation_split = 0.1,
            verbose = 0
        )
        print("모델 생성 완료")
        if save_by_checkpoint is True:
            self.saveModelByCheckpoint()
        else:
            self._saveModel()

        self.df_history = pd.DataFrame({
            "loss" : self.history.history["loss"],
            "mae" : self.history.history["mae"],
            "mse" : self.history.history["mse"],
            "val_loss" : self.history.history["val_loss"],
            "val_mae" : self.history.history["val_mae"],
            "val_mse" : self.history.history["val_mse"]
        })
        plt.figure(figsize=(20, 8))
        plt.plot(self.df_history.loc[10:, ["loss", "val_loss"]], label=["loss", "val_loss"])
        plt.xlabel("EPOCHS")
        plt.legend()
        plt.show()

    def saveModelByCheckpoint(self, cp_index=-1):
        self.ckpts = [x for x in os.listdir(self.checkpoint_folder) if x.split(".")[-1] == "index"]
        self.ckpts.sort()
        this_checkpoint_path = self.checkpoint_folder + "/" + self.ckpts[cp_index].split(".")[0] + ".ckpt"
        self.model.load_weights(this_checkpoint_path)
        self._saveModel()
        print("check point : {}".format(self.ckpts[cp_index].split(".")[0]))

    def _saveModel(self):
        model_file_path = "kospi_predictor_model/{}".format(self.model_save_path)
        self.model.save(model_file_path, overwrite=True)
        x_cols_str = json.dumps(self.x_cols)
        y_cols_str = json.dumps(self.y_cols)
        open("kospi_predictor_model/{}/x_cols_info.txt".format(self.model_save_path), "w").write(x_cols_str)
        open("kospi_predictor_model/{}/y_cols_info.txt".format(self.model_save_path), "w").write(y_cols_str)
        print("모델 저장 완료")

    def validateModel(self):
        X_test = self.df_test[self.x_cols].to_numpy().reshape(-1, 1, len(self.x_cols))
        y_test = self.df_test[self.y_cols].to_numpy()
        predicted = self.model.predict(X_test)

        self.df_predicted = None
        test_error = []
        for i in range(len(self.y_cols)):
            this_df = pd.DataFrame({
                "org_{}".format(i) : y_test[:, i],
                "predicted_{}".format(i) : predicted[:, i],
                "predict error_{}".format(i) : abs((y_test - predicted)[:, i]),
                "predict error(sqr)_{}".format(i) : abs((y_test - predicted)[:, i]) ** 2
            })
            this_test_mae = this_df.loc[:, ["predict error_{}".format(i)]].mean().values[0]
            this_test_mse = this_df.loc[:, ["predict error(sqr)_{}".format(i)]].mean().values[0]
            test_error.append({
                "y_no" : i,
                "test_mae" : this_test_mae,
                "test_mse" : this_test_mse
            })
            if self.df_predicted is None:
                self.df_predicted = this_df
            else:
                self.df_predicted = pd.concat([self.df_predicted, this_df], axis=1)
        self.test_error = pd.DataFrame(test_error)
        test_error_str = json.dumps(test_error)
        open("kospi_predictor_model/{}/test_error.txt".format(self.model_save_path), "w").write(test_error_str)
        # plt.figure(1, figsize=(16,8))
        # plt.plot(self.df_predicted)

        plt.figure(figsize=(20, 16))
        for i in range(len(self.y_cols)):
            plt.subplot(4, int(len(self.y_cols)/4), i+1)
            plt.scatter(x=self.df_predicted["org_{}".format(i)], y=self.df_predicted["predicted_{}".format(i)])

class Predictor():
    def __init__(self, df, scale_method="minmax", model_save_path = "saved_model"):
        self.model_save_path = model_save_path
        self.scale_method = scale_method
        scale_info_str = open("kospi_predictor_model/{}/scale_info.txt".format(self.model_save_path), "r").readlines()[0]
        x_cols_str = open("kospi_predictor_model/{}/x_cols_info.txt".format(self.model_save_path), "r").readlines()[0]
        y_cols_str = open("kospi_predictor_model/{}/y_cols_info.txt".format(self.model_save_path), "r").readlines()[0]
        self.scale_info = json.loads(scale_info_str)
        self.x_cols = json.loads(x_cols_str)
        self.y_cols = json.loads(y_cols_str)
        self.df_for_predict = df.copy()
        self.model = tf.keras.models.load_model("kospi_predictor_model/{}/".format(self.model_save_path))

    def _getHolidays(self, year):
        header = {
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.109 Safari/537.36"
        }

        url_code = "https://open.krx.co.kr/contents/COM/GenerateOTP.jspx"
        param_code = {
            "bld": "MKD/01/0110/01100305/mkd01100305_01",
            "name": "form"
        }
        res_code = requests.get(url_code, headers=header, params=param_code)
        code = res_code.text
        url = "https://open.krx.co.kr/contents/OPN/99/OPN99000001.jspx"
        form_data = {
            "search_bas_yy": year,
            "gridTp": "KRX",
            "pagePath": "/contents/MKD/01/0110/01100305/MKD01100305.jsp",
            "code": code,
            "pageFirstCall": "Y"
        }
        res = requests.post(url, headers=header, data=form_data)
        holiday = res.json()
        return [x["calnd_dd"] for x in holiday["block1"]]

    def _getPredDateRange(self, history_before=1):
        start = datetime.strptime(self.df_for_predict.iloc[-history_before]["date"], "%Y-%m-%d")
        period = len(self.predicted[0])
        self.pred_date_range = []
        holidays = self._getHolidays(start.year)
        
        def _temp(start, period):
            end = start + timedelta(days=period)
            this_date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end-start).days+1)][1:]
            for date in this_date_range:
                yoil_no = datetime.strptime(date, "%Y-%m-%d").weekday()
                yoil = ["월", "화", "수", "목", "금", "토", "일"][yoil_no]
                if yoil in ["토", "일"] or date in holidays:
                    if date in self.pred_date_range:
                        self.pred_date_range.remove(date)
                else:
                    if date not in self.pred_date_range:
                        self.pred_date_range.append(date)
            if len(self.pred_date_range) < len(self.predicted[0]):
                period_new = period + len(self.predicted[0]) - len(self.pred_date_range)
                _temp(start, period_new)
        _temp(start, period)
        return self.pred_date_range

    def predict(self, history_before = 1):
        x_for_predict = self.df_for_predict.loc[len(self.df_for_predict)-history_before, self.x_cols].to_numpy().reshape(-1, 1, len(self.x_cols)).astype(np.float64)
        self.predicted = self.model.predict(x_for_predict)
        pred_date_range = self._getPredDateRange(history_before)

        df_predicted = pd.DataFrame({"date" : pred_date_range, "KOSPI" : self.predicted[0], "cate" : "predict"})
        df_before = self.df_for_predict.loc[:, ["date", "KOSPI"]]
        df_before["cate"] = "history"
        self.df_result = pd.concat([df_before, df_predicted], ignore_index=True)
        self.df_result["md"] = ""
        prev_month = ""
        for item in self.df_result.iterrows():
            month = item[1]["date"][5:7]
            day = item[1]["date"][8:10]
            if month[0:1] == "0":
                month = month[1:2]
            if day[0:1] == "0":
                day = day[1:2]

            if month != prev_month:
                self.df_result.loc[item[0], "md"] = month + "." + day
            else:
                self.df_result.loc[item[0], "md"] = day
            prev_month = month

        #원래 금액으로 역정규화
        if self.scale_method == "minmax":
            min = self.scale_info["KOSPI"]["min"]
            max = self.scale_info["KOSPI"]["max"]
            self.df_result["KOSPI_price"] = ((max - min) * self.df_result["KOSPI"]) + min
        elif self.scale_method == "norm":
            mean = self.scale_info["KOSPI"]["mean"]
            std = self.scale_info["KOSPI"]["std"]
            self.df_result["KOSPI_price"] = (std * self.df_result["KOSPI"]) + mean
        else:
            self.df_result["KOSPI_price"] = self.df_result["KOSPI"]

        return self.df_result

    def showPredictionPlot(self):
        plt.figure(figsize=(30, 12))
        plt.title("KOSPI PREDICTION", fontsize="60")
        plt.plot(self.df_result.loc[(self.df_result["cate"] == "history"), "KOSPI_price"], label = "history")
        plt.plot(self.df_result.loc[(self.df_result["cate"] == "predict"), "KOSPI_price"], label = "predict")
        for item in self.df_result.loc[(self.df_result["cate"] == "predict")].iterrows():
            plt.text(item[0], item[1]["KOSPI_price"], "{:.0f}".format(item[1]["KOSPI_price"]))
        plt.xticks(self.df_result.index, self.df_result["md"])
        plt.legend(fontsize="30")
        plt.show()

    def showPredictionByHistory(self, df_result, history_before = -20):
        history = df_result.loc[(df_result["cate"] == "history"), ["date", "md", "KOSPI_price"]]
        predict = df_result.loc[(df_result["cate"] == "predict"), ["date", "md", "KOSPI_price"]]
        history.rename(columns={"KOSPI_price" : "KOSPI_history"}, inplace=True)
        predict.rename(columns={"KOSPI_price" : "KOSPI_predict"}, inplace=True)
        history.reset_index(drop=True, inplace=True)
        predict.reset_index(drop=True, inplace=True)

        start_index_no = history[(history["date"] == predict.loc[0, "date"])].index[0]
        history = history.loc[start_index_no:]
        history.reset_index(drop=True, inplace=True)

        df_validate = predict.merge(history, left_index=True, right_index=True, how="left")

        plt.figure(figsize=(30, 12))
        plt.title("PREDICTION Validate", fontsize="60")
        plt.plot(df_validate.loc[:, "KOSPI_history"], label = "history")
        plt.plot(df_validate.loc[:, "KOSPI_predict"], label = "predict")
        plt.xticks(df_validate.index, df_validate["md_x"])
        plt.legend(fontsize="30")
        plt.show()

    def saveModelToJS(self):
        js_folder = "kospi_predictor_model/{}_js".format(self.model_save_path)
        for file in os.listdir(js_folder):
            os.remove(js_folder+"/"+file)
        tfjs.converters.save_keras_model(self.model, js_folder)