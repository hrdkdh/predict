import os
import json
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
import requests as _requests
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split as train_test_split
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler

warnings.filterwarnings(action="ignore") #판다스 워닝 끄기

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
        self.df_investor = None
        self.crawl_page_max = crawl_page_max
        self.info_for_crawl = {
            "KOSPI" : {
                "url" : "https://finance.daum.net/api/market_index/days",
                "param" : {
                    "market": "KOSPI",
                    "pagination": "true",
                    "perPage" : perPage
                },
                "map_for_df" : [
                    {"col" : "date", "item_key" : "date" },
                    {"col" : "KOSPI", "item_key" : "tradePrice" }
                ]
            },
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
            }
        }

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
            result = []
            url = self.info_for_crawl[want_data_name]["url"]
            param = self.info_for_crawl[want_data_name]["param"]
            for page_no in range(1, self.crawl_page_max+1):
                print_prefix = ""
                if page_no == 1:
                    print_prefix = "\n"

                print("{}{} : {}번째 페이지 데이터 수집중...".format(print_prefix, want_data_name_for_print, page_no), end="\r")
                
                if self.df_investor is not None:
                    continue
                
                param["page"] = page_no
                res = _requests.get(url, headers=header, params=param)
                json_parsed = res.json()
                for item in json_parsed["data"]:
                    this_dic = {}
                    for map in self.info_for_crawl[want_data_name]["map_for_df"]:
                        this_value = item[map["item_key"]]
                        if map["col"] == "date":
                            this_value = this_value[:10]
                        this_dic[map["col"]] = this_value
                    result.append(this_dic)
            if want_data_name in ["INDI", "FOREIGN", "ORG"]:
                if self.df_investor is None:
                    this_df = pd.DataFrame(result)
                    self.df_investor = this_df
                    if self.df is None:
                        self.df = this_df
                    else:
                        self.df = self.df.merge(this_df, how="left", on="date")
            else:
                this_df = pd.DataFrame(result)
                if self.df is None:
                    self.df = this_df
                else:
                    self.df = self.df.merge(this_df, how="left", on="date")
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
    def __init__(self, df_crawled, cols, scale_method="minmax"):
        self.df = df_crawled.copy()
        self.cols = cols
        self.scale_method = scale_method
        self.scaled_tag = ""
        self.scale_info = {}
        pass
    
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

    def scalingForModeling(self):
        self.scaled_tag = "_scaled"
        self.scale_info["how"] = self.scale_method
        if self.scale_method == "minmax":
            scaler = MinMaxScaler()
            for col in self.cols:
                scaler.fit(self.df.loc[:, [col]])
                self.df[col+self.scaled_tag] = scaler.transform(self.df.loc[:, [col]])
                self.scale_info[col] = {"min" : scaler.data_min_[0], "max" : scaler.data_max_[0]}
        elif self.scale_method == "norm":
            for col in self.cols:
                self.df[col+self.scaled_tag], mean, std = self.norm(self.df.loc[:, [col]])
                self.scale_info[col] = {"mean" : mean[col], "std" : std[col]}
                
        scale_info_str = json.dumps(self.scale_info)
        open("kospi_predictor_model/saved_model/scale_info.txt", "w").write(scale_info_str)

    def scalingForPredict(self):
        self.scaled_tag = "_scaled"
        scale_info_str = open("kospi_predictor_model/saved_model/scale_info.txt", "r").readlines()[0]
        scale_info = json.loads(scale_info_str)
        if self.scale_method == "minmax":
            for col in self.cols:
                min = scale_info[col]["min"]
                max = scale_info[col]["max"]
                self.df[col+self.scaled_tag] = (self.df.loc[:, [col]] - min) / (max - min)
        elif self.scale_method == "norm":
            for col in self.cols:
                mean = scale_info[col]["mean"]
                std = scale_info[col]["std"]
                self.df[col+self.scaled_tag] = (self.df.loc[:, [col]] - mean) / std

    def makeDiff(self):
        for col in self.cols:
            self.df["X_{}{}_DIFF".format(col, self.scaled_tag)] = 0.0
            for idx in range(0, len(self.df)): #각 행을 돌면서
                if idx > 0:
                    self.df.loc[idx, "X_{}{}_DIFF".format(col, self.scaled_tag)] = self.df.loc[idx, "{}{}".format(col, self.scaled_tag)] - self.df.loc[idx-1, "{}{}".format(col, self.scaled_tag)]
    
    def makeAR(self, minAR=2, maxAR=11):
        maxAR += 1
        for col in self.cols:
            for before_day in range(minAR, maxAR):
                self.df["X_{}{}_AR{}".format(col, self.scaled_tag, before_day)] = 0.0
                for idx in range(0, len(self.df)): #각 행을 돌면서
                    before_day_idx = idx - before_day
                    if before_day_idx < 0:
                        continue
                    self.df.loc[idx, "X_{}{}_AR{}".format(col, self.scaled_tag, before_day)] = self.df.loc[before_day_idx, "{}{}".format(col, self.scaled_tag)]

    def makeMA(self, minMA=1, maxMA=11):
        maxMA += 1
        for col in self.cols:
            for mean_period in range(minMA, maxMA):
                self.df["X_{}{}_MA{}".format(col, self.scaled_tag, mean_period)] = 0.0
                for idx in range(0, len(self.df)): #각 행을 돌면서
                    if idx < mean_period:
                        continue
                    this_sum = 0
                    this_cnt = 0
                    for before_day in range(-mean_period, 0): #이전 일자의 행들을 돌면서
                        before_idx = idx + before_day
                        this_sum += self.df.loc[before_idx, "{}{}".format(col, self.scaled_tag)]
                        this_cnt += 1
                    self.df.loc[idx, "X_{}{}_MA{}".format(col, self.scaled_tag, mean_period)] = this_sum/this_cnt

    def makeTargetXs(self, len_x_ARMA):
        self.makeDiff()
        self.makeAR(1, len_x_ARMA)
        self.makeMA(2, len_x_ARMA)

    def makeTargetYs(self, next_day_len):
        for day in range(1, next_day_len+1):
            self.df["Y_KOSPI{}_nextday_{}".format(self.scaled_tag, day)] = 0.0
            for idx in range(0, len(self.df)): #각 행을 돌면서
                    if idx+day >= len(self.df):
                        continue
                    self.df.loc[idx, "Y_KOSPI{}_nextday_{}".format(self.scaled_tag, day)] = self.df.loc[idx+day, "KOSPI{}".format(self.scaled_tag)]
        self.y_list = [x for x in self.df.columns if x[:2] == "Y_"]

    def cutoffData(self, cutoff_len_head, cutoff_len_tail):
        """
        AR과 MA로 인해 초기 0값을 가지고 있는 행들을 제거한다.
        """
        self.df_cutoff = self.df.iloc[cutoff_len_head+1:-cutoff_len_tail]
        self.df_cutoff.reset_index(inplace=True, drop=True)

    def splitData(self, test_size=0.2, train_size=0.8):
        df_splited = train_test_split(self.df_cutoff, test_size=test_size, train_size=train_size, random_state=8699)
        self.df_train = df_splited[0]
        self.df_test = df_splited[1]

class ModelMaker():
    def __init__(self, y_cols, df_train, df_test, save_path="saved_model"):
        self.save_path = save_path
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.x_cols = [x for x in self.df_train.columns if x[:2] == "X_"]
        self.y_cols = y_cols

    def _constructModel(self, learning_rate):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(1, len(self.x_cols))),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(len(self.x_cols)*10, activation='relu'),
            tf.keras.layers.Dense(len(self.x_cols)*10, activation='relu'),
            # tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(len(self.x_cols)*10, activation='relu'),
            tf.keras.layers.Dense(len(self.x_cols)*10, activation='relu'),
            tf.keras.layers.Dense(len(self.y_cols))
        ])
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        self.model.summary()

    def makeModel(self, EPOCHS=200, learning_rate=0.001, history_plot_cutoff = 20):
        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None, EPOCHS=EPOCHS):
                percent = round(epoch/EPOCHS*100)
                print("EPOCH : {}/{}({}%) loss : {:.4f}, mae : {:.4f}, mse : {:.4f} / val_loss : {:.4f}, val_mae : {:.4f}, val_mse : {:.4f}{}"
                .format(epoch, EPOCHS, percent, 
                        logs["loss"], logs["mae"], logs["mse"], logs["val_loss"], logs["val_mae"], logs["val_mse"], " "*30),
                        end="\r")

        X_train = self.df_train[self.x_cols].to_numpy().reshape(-1, 1, len(self.x_cols))
        y_train = self.df_train[self.y_cols].to_numpy()
        
        checkpoint_folder = "kospi_predictor_model/checkpoint/"
        for file in os.listdir(checkpoint_folder):
            os.remove(checkpoint_folder+file)
        checkpoint_path = checkpoint_folder + "cp-{epoch:04d}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=0, 
            monitor="val_loss",
            save_weights_only=True,
            save_best_only = True,
            save_freq="epoch"
        )

        self._constructModel(learning_rate=learning_rate)

        self.history = self.model.fit(
            X_train, y_train,
            # batch_size = batch_size,
            callbacks=[cp_callback, CustomCallback()],
            epochs=EPOCHS, validation_split = 0.1,
            verbose = 0
        )
        print("모델 생성 완료")
        latest = tf.train.latest_checkpoint(checkpoint_folder)
        self.model.load_weights(latest)

        print("모델 저장 완료")
        model_file_path = "kospi_predictor_model/{}".format(self.save_path)
        self.model.save(model_file_path, overwrite=True)
        x_cols_str = json.dumps(self.x_cols)
        y_cols_str = json.dumps(self.y_cols)
        open("kospi_predictor_model/{}/x_cols_info.txt".format(self.save_path), "w").write(x_cols_str)
        open("kospi_predictor_model/{}/y_cols_info.txt".format(self.save_path), "w").write(y_cols_str)

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

    def validateModel(self, y_no):
        X_test = self.df_test[self.x_cols].to_numpy().reshape(-1, 1, len(self.x_cols))
        y_test = self.df_test[self.y_cols].to_numpy()
        predicted = self.model.predict(X_test)
        self.df_predicted = pd.DataFrame({
            "org" : y_test[:, y_no],
            "predicted" : predicted[:, y_no],
            "predict error" : abs((y_test - predicted)[:, y_no]),
            "predict error(sqr)" : abs((y_test - predicted)[:, y_no]) ** 2
        })
        self.validate_mae = self.df_predicted[:, ["predict error"]].mean()
        self.validate_mse = self.df_predicted[:, ["predict error(sqr)"]].mean()
        validate_str = json.dumps({
            "validate_mae" : self.validate_mae,
            "validate_mse" : self.validate_mse
        })
        open("kospi_predictor_model/{}/validate_info.txt".format(self.save_path), "w").write(validate_str)
        plt.figure(figsize=(16,8))
        plt.plot(self.df_predicted)

class Predictor():
    def __init__(self, df, scaled=True, scale_method="minmax", model_path = "saved_model"):
        self.model_path = model_path
        self.scaled = scaled
        self.scale_method = scale_method
        self.scaled_tag = ""
        if self.scaled is True:
            self.scaled_tag = "_scaled"
        scale_info_str = open("kospi_predictor_model/{}/scale_info.txt".format(self.model_path), "r").readlines()[0]
        x_cols_str = open("kospi_predictor_model/{}/x_cols_info.txt".format(self.model_path), "r").readlines()[0]
        y_cols_str = open("kospi_predictor_model/{}/y_cols_info.txt".format(self.model_path), "r").readlines()[0]
        self.scale_info = json.loads(scale_info_str)
        self.x_cols = json.loads(x_cols_str)
        self.y_cols = json.loads(y_cols_str)
        self.df_for_predict = df.copy()
        self.model = tf.keras.models.load_model("kospi_predictor_model/{}/".format(self.model_path))

    def _getPredDateRange(self):
        start = datetime.strptime(self.df_for_predict.iloc[-1]["date"], "%Y-%m-%d")
        period = len(self.predicted[0])
        self.pred_date_range = []
        
        def _temp(start, period):
            end = start + timedelta(days=period)
            this_date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end-start).days+1)][1:]
            for date in this_date_range:
                yoil_no = datetime.strptime(date, "%Y-%m-%d").weekday()
                yoil = ["월", "화", "수", "목", "금", "토", "일"][yoil_no]
                if yoil in ["토", "일"]:
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

    def predict(self):
        x_for_predict = self.df_for_predict.loc[len(self.df_for_predict)-1, self.x_cols].to_numpy().reshape(-1, 1, len(self.x_cols)).astype(np.float64)
        self.predicted = self.model.predict(x_for_predict)
        pred_date_range = self._getPredDateRange()

        df_predicted = pd.DataFrame({"date" : pred_date_range, "KOSPI{}".format(self.scaled_tag) : self.predicted[0], "cate" : "predict"})
        df_before = self.df_for_predict.loc[:, ["date", "KOSPI{}".format(self.scaled_tag)]]
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

        if self.scaled is True:
            #원래 금액으로 역정규화
            if self.scale_method == "minmax":
                min = self.scale_info["KOSPI"]["min"]
                max = self.scale_info["KOSPI"]["max"]
                self.df_result["KOSPI_price"] = ((max - min) * self.df_result["KOSPI{}".format(self.scaled_tag)]) + min
            elif self.scale_method == "norm":
                mean = self.scale_info["KOSPI"]["mean"]
                std = self.scale_info["KOSPI"]["std"]
                self.df_result["KOSPI_price"] = (std * self.df_result["KOSPI{}".format(self.scaled_tag)]) + mean

        else:
            self.df_result["KOSPI_price"] = self.df_result["KOSPI"]

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

    def saveModelToJS(self):
        tfjs.converters.save_keras_model(self.model, "kospi_predictor_model/{}_js".format(self.model_path))