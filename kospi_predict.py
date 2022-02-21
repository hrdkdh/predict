import os
import json
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
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
            }
        }

    def crawlData(self, wantDataNames):
        header = {
            "referer": "https://finance.daum.net",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
        }
        if type(wantDataNames) != list:
            wantDataNames = [wantDataNames]
        
        for wandDataName in wantDataNames:
            result = []
            url = self.info_for_crawl[wandDataName]["url"]
            param = self.info_for_crawl[wandDataName]["param"]
            for page_no in range(1, self.crawl_page_max+1):
                print_prefix = ""
                if page_no == 1:
                    print_prefix = "\n"

                print("{}{} : {}번째 페이지 데이터 수집중...".format(print_prefix, wandDataName, page_no), end="\r")
                param["page"] = page_no
                res = _requests.get(url, headers=header, params=param)
                json_parsed = res.json()
                for item in json_parsed["data"]:
                    this_dic = {}
                    for map in self.info_for_crawl[wandDataName]["map_for_df"]:
                        this_value = item[map["item_key"]]
                        if map["col"] == "date":
                            this_value = this_value[:10]
                        this_dic[map["col"]] = this_value
                    result.append(this_dic)
            this_df = pd.DataFrame(result)
            if self.df is None:
                self.df = this_df
            else:
                self.df = self.df.merge(this_df, on="date")
        return self.df

class DataPreprocessor():
    def __init__(self, df_crawled, cols):
        self.df = df_crawled.copy()
        self.cols = cols
        self.minmax_info = {}
        pass
    
    def sortByDate(self):
        self.df.sort_values(by="date", inplace = True)
        self.df.reset_index(inplace = True, drop=True)
        
    def scalingForModeling(self):
        scaler = MinMaxScaler()
        for col in self.cols:
            scaler.fit(self.df.loc[:, [col]])
            self.df[col+"_scaled"] = scaler.transform(self.df.loc[:, [col]])
            self.minmax_info[col] = {"min" : scaler.data_min_[0], "max" : scaler.data_max_[0]}
        minmax_info_str = json.dumps(self.minmax_info)
        open("kospi_predictor_model/saved_model/minmax_info.txt", "w").write(minmax_info_str)

    def scalingForPredict(self):
        minmax_info_str = open("kospi_predictor_model/saved_model/minmax_info.txt", "r").readlines()[0]
        minmax_info = json.loads(minmax_info_str)
        for col in self.cols:
            min = minmax_info[col]["min"]
            max = minmax_info[col]["max"]
            self.df[col+"_scaled"] = (self.df.loc[:, [col]] - min) / (max - min)

    def makeDiff(self):
        for col in self.cols:
            self.df["X_{}_scaled_DIFF".format(col)] = 0.0
            for idx in range(0, len(self.df)): #각 행을 돌면서
                if idx > 0:
                    self.df.loc[idx, "X_{}_scaled_DIFF".format(col)] = self.df.loc[idx, "{}_scaled".format(col)] - self.df.loc[idx-1, "{}_scaled".format(col)]
    
    def makeAR(self, minAR=2, maxAR=11):
        maxAR += 1
        for col in self.cols:
            for before_day in range(minAR, maxAR):
                self.df["X_{}_scaled_AR{}".format(col, before_day)] = 0.0
                for idx in range(0, len(self.df)): #각 행을 돌면서
                    before_day_idx = idx - before_day
                    if before_day_idx < 0:
                        continue
                    self.df.loc[idx, "X_{}_scaled_AR{}".format(col, before_day)] = self.df.loc[before_day_idx, "{}_scaled".format(col)]

    def makeMA(self, minMA=1, maxMA=11):
        maxMA += 1
        for col in self.cols:
            for mean_period in range(minMA, maxMA):
                self.df["X_{}_scaled_MA{}".format(col, mean_period)] = 0.0
                for idx in range(0, len(self.df)): #각 행을 돌면서
                    if idx < mean_period:
                        continue
                    this_sum = 0
                    this_cnt = 0
                    for before_day in range(-mean_period, 0): #이전 일자의 행들을 돌면서
                        before_idx = idx + before_day
                        this_sum += self.df.loc[before_idx, "{}_scaled".format(col)]
                        this_cnt += 1
                    self.df.loc[idx, "X_{}_scaled_MA{}".format(col, mean_period)] = this_sum/this_cnt

    def makeTargetXs(self, len_x_ARMA):
        self.makeDiff()
        self.makeAR(1, len_x_ARMA)
        self.makeMA(2, len_x_ARMA)

    def makeTargetYs(self, next_day_len):
        for day in range(1, next_day_len+1):
            self.df["Y_KOSPI_scaled_nextday_{}".format(day)] = 0.0
            for idx in range(0, len(self.df)): #각 행을 돌면서
                    if idx+day >= len(self.df):
                        continue
                    self.df.loc[idx, "Y_KOSPI_scaled_nextday_{}".format(day)] = self.df.loc[idx+day, "KOSPI_scaled"]
        self.y_list = [x for x in self.df.columns if "Y_" in x]

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
    def __init__(self, y_cols, df_train, df_test):
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.x_cols = [x for x in self.df_train.columns if "X_" in x]
        self.y_cols = y_cols

    def _constructModel(self, learning_rate):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(1, len(self.x_cols))),
            tf.keras.layers.Dense(len(self.x_cols)*7, activation='relu'),
            tf.keras.layers.Dense(len(self.x_cols)*7, activation='relu'),
            tf.keras.layers.Dense(len(self.x_cols)*7, activation='relu'),
            tf.keras.layers.Dense(len(self.x_cols)*7, activation='relu'),
            tf.keras.layers.Dense(len(self.x_cols)*7, activation='relu'),
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
        model_file_path = "kospi_predictor_model/saved_model"
        self.model.save(model_file_path, overwrite=True)
        x_cols_str = json.dumps(self.x_cols)
        y_cols_str = json.dumps(self.y_cols)
        open("kospi_predictor_model/saved_model/x_cols_info.txt", "w").write(x_cols_str)
        open("kospi_predictor_model/saved_model/y_cols_info.txt", "w").write(y_cols_str)

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
            "predict error" : abs((y_test - predicted)[:, y_no])
        })
        plt.figure(figsize=(16,8))
        plt.plot(self.df_predicted)

class Predictor():
    def __init__(self, df):
        minmax_info_str = open("kospi_predictor_model/saved_model/minmax_info.txt", "r").readlines()[0]
        x_cols_str = open("kospi_predictor_model/saved_model/x_cols_info.txt", "r").readlines()[0]
        y_cols_str = open("kospi_predictor_model/saved_model/y_cols_info.txt", "r").readlines()[0]
        self.minmax_info = json.loads(minmax_info_str)
        self.x_cols = json.loads(x_cols_str)
        self.y_cols = json.loads(y_cols_str)
        self.df_for_predict = df.copy()
        self.model = tf.keras.models.load_model("kospi_predictor_model/saved_model/")

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

        df_predicted = pd.DataFrame({"date" : pred_date_range, "KOSPI_scaled" : self.predicted[0], "cate" : "predict"})
        df_before = self.df_for_predict.loc[:, ["date", "KOSPI_scaled"]]
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
        min = self.minmax_info["KOSPI"]["min"]
        max = self.minmax_info["KOSPI"]["max"]
        self.df_result["KOSPI_price"] = ((max - min) * self.df_result["KOSPI_scaled"]) + min

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