{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#pip install "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from datetime import datetime\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.model_selection import train_test_split\n",
    "import gc #garbage collector\n",
    "import sys\n",
    "import os\n",
    "from pandas.api.types import CategoricalDtype\n",
    "#from xgboost import XGBRegressor as xgb\n",
    "import xgboost as xgb\n",
    "from itertools import product\n",
    "from statistics import stdev \n",
    "import statsmodels.api as sm\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.backends.backend_pdf\n",
    "from datetime import date\n",
    "import pickle\n",
    "import pyodbc\n",
    "import time\n",
    "from tqdm import tqdm\n",
    "import copy\n",
    "from pathlib import Path\n",
    "import datetime\n",
    "from datetime import date\n",
    "from xgboost import XGBRegressor\n",
    "from dateutil.relativedelta import *\n",
    "from pandas import ExcelWriter\n",
    "from datetime import datetime\n",
    "from math import sqrt\n",
    "\n",
    "from sklearn2pmml import sklearn2pmml\n",
    "from sklearn_pandas import DataFrameMapper\n",
    "from sklearn2pmml import PMMLPipeline\n",
    "from sklearn2pmml import Pipeline\n",
    "from sklearn.preprocessing import LabelBinarizer\n",
    "from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain\n",
    "from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler\n",
    "from sklearn2pmml.preprocessing.xgboost import make_xgboost_column_transformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn2pmml.preprocessing import PMMLLabelEncoder\n",
    "from sklearn2pmml import make_pmml_pipeline\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "\n",
    "sns.set()\n",
    "\n",
    "import warnings\n",
    "warnings.simplefilter(action='ignore', category=FutureWarning)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "############################\n",
    "##### Helper Functions #####\n",
    "############################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_na_report(df):\n",
    "    \"\"\"\n",
    "    create a report of variables whose values are NA across all table, and eliminate them\n",
    "    \"\"\"\n",
    "    print(\"\\nDataset has {} entries and {} features\".format(*df.shape))\n",
    "    na_report = df.isna().sum()                          #create a list of columns and sum of null values\n",
    "    na_report = pd.DataFrame(na_report, index=None)     #convert list to a dataframe\n",
    "    na_report.reset_index(inplace=True)                  #reset index to range(0 : len(df)-1)\n",
    "    na_report.columns = [\"variable\", \"na_count\"]        #set column names\n",
    "    na_report[\"perc\"] = na_report[\"na_count\"]/df.shape[0]           #add a new column which is percentage of null vales by column name\n",
    "    gc.collect()                                                #garbage collecter for memory efficiency\n",
    "    return na_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_dtype_report(df):\n",
    "    \"\"\"\n",
    "    define the following reports:\n",
    "    num_cols: indicating which variables are numeric\n",
    "    char_cols: indicating which variables are of type character\n",
    "    \"\"\"\n",
    "    df.replace({True: 1, False: 0}, inplace=True)\n",
    "    df_cols = df.columns\n",
    "    num_cols = list(df._get_numeric_data().columns)\n",
    "    char_cols = list(set(df_cols) - set(num_cols))\n",
    "    gc.collect()\n",
    "    return num_cols, char_cols"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def eliminate_missing(df, na_report, max_threshold):\n",
    "    ''' \n",
    "    # define maximum threshold of missing percentage\n",
    "    # eliminate columns that do not fit the threshold\n",
    "    '''\n",
    "    cols_before = df.columns.tolist()\n",
    "    cols_ = df.columns[df.columns.isin(list(na_report[na_report['perc'] < max_threshold]['variable']) or df.columns.isin(list(key_var)))]\n",
    "    df = df[cols_]\n",
    "    print(len(cols_before)-len(cols_),\"columns eliminated due to missing values\")\n",
    "    del cols_, cols_before\n",
    "    gc.collect()\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def eliminate_single_unique(df):\n",
    "    ''' \n",
    "    # eliminate columns having a single unique value \n",
    "    '''\n",
    "    cols_before = df.columns.tolist()\n",
    "    unique_report = df.nunique()                          \n",
    "    unique_report = pd.DataFrame(unique_report, index=None)    \n",
    "    unique_report.reset_index(inplace=True)\n",
    "    unique_report.columns = [\"variable\", \"number_unique\"]        #set column names\n",
    "    unique_report = unique_report[~unique_report['variable'].isin(key_var)]  #discard key variables  \n",
    "    u_cols = df.columns[~df.columns.isin(list(unique_report[unique_report['number_unique'] == 1]['variable']))]\n",
    "    df = df[u_cols]\n",
    "    print(len(cols_before)-len(u_cols),\"columns eliminated due to single unique values, namely:\",list(set(cols_before) - set(u_cols)))\n",
    "    del unique_report, u_cols, cols_before\n",
    "    gc.collect()\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_encoding_matrix(df):\n",
    "    \"\"\"\n",
    "    create a matrix (or dataframe) with 2 columns:\n",
    "    variable: categorical columns of dataframe\n",
    "    levels: levels of unique values in categorical columns (including missing values)\n",
    "    \"\"\"\n",
    "    missing_string = \"MISSING\"\n",
    "    df_temp = df.copy()\n",
    "    cols = list(set(list(df_temp.select_dtypes(include=['category', 'object']).columns)) - set(key_vars))\n",
    "    assert len(cols) == len(set(cols)) , \"please ensure that you are using unique column names.\"\n",
    "    enc_df = pd.DataFrame(columns = [\"variable\", \"levels\"])\n",
    "    for i in cols:\n",
    "        if (pd.api.types.is_categorical_dtype(df_temp[i])):\n",
    "            df_temp[i] = df_temp[i].cat.add_categories([missing_string])       \n",
    "        df_temp[i] = df_temp[i].fillna(value=missing_string)\n",
    "        colLevels = list(df_temp[i].unique())\n",
    "        if missing_string not in colLevels:\n",
    "           colLevels.append(missing_string)\n",
    "        enc_df = enc_df.append({'variable' : i , 'levels' : colLevels} , ignore_index=True)\n",
    "    del df_temp, cols\n",
    "    gc.collect()\n",
    "    return enc_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def encode_from_matrix(df, enc_df):\n",
    "    \"\"\"\n",
    "    transforms categorical columns into numeric values based on encoded matrix\n",
    "    enc_df: result of create_encoding_matrix() function composed of 2 columns:\n",
    "        variable: categorical columns of dataframe\n",
    "        levels: levels of unique values in categorical columns (including missing values)\n",
    "    \"\"\"\n",
    "    missing_string = \"MISSING\"\n",
    "    df_temp = df.copy()\n",
    "    matrix_cols = list(enc_df['variable'].values)\n",
    "    for i in matrix_cols:\n",
    "        if (pd.api.types.is_categorical_dtype(df_temp[i])):\n",
    "            df_temp[i] = df_temp[i].cat.add_categories([missing_string])\n",
    "        df_temp[i] = df_temp[i].fillna(value=missing_string)\n",
    "        df_temp[i] = df_temp[i].astype('category')\n",
    "        temp_levels = list(enc_df[enc_df['variable']==i]['levels'])[0]\n",
    "        df_temp[i] = pd.Categorical(df_temp[i], categories=temp_levels).codes +1\n",
    "        del temp_levels\n",
    "    del matrix_cols\n",
    "    gc.collect()\n",
    "    return df_temp"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Modelleme asamalari için gerekli fonksiyonlar"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cal_mae_and_mape(df, sample='train', pred_name='predicted'):\n",
    "    temp = df.copy()\n",
    "    mae_ = mean_absolute_error(temp.loc[temp['sample'] == sample]['actual'], temp.loc[temp['sample'] == sample][pred_name])   \n",
    "    mape_ = np.mean(np.abs((temp.loc[temp['sample'] == sample]['actual']-temp.loc[temp['sample'] == sample][pred_name])/temp.loc[temp['sample'] == sample]['actual'])) \n",
    "    del temp\n",
    "    gc.collect()\n",
    "    return mae_,mape_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cal_mae_and_rmse(df, sample='train', pred_name='predicted'):\n",
    "    temp = df.copy()\n",
    "    mae_ = mean_absolute_error(temp.loc[temp['sample'] == sample]['actual'], temp.loc[temp['sample'] == sample][pred_name])   \n",
    "    rmse_ = sqrt(mean_squared_error(temp.loc[temp['sample'] == sample]['actual'], temp.loc[temp['sample'] == sample][pred_name]))\n",
    "    del temp\n",
    "    gc.collect()\n",
    "    return mae_,rmse_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def expand_grid(dictionary):\n",
    "   \"\"\"Create a dataframe from every combination of given values.\"\"\"\n",
    "   return pd.DataFrame([row for row in product(*dictionary.values())],\n",
    "                       columns=dictionary.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def set_split_index(df, splitting_perc, seed, min_date):\n",
    "    # allocate Development, and OOT validation data\n",
    "    oot_index = df.loc[df['refdate'] >= min_date].index.values\n",
    "    dev_index = df.loc[df['refdate'] < min_date].index.values  \n",
    "    \n",
    "    y_all = df['target'].iloc[dev_index]\n",
    "    \n",
    "    train_index, test_index, y_train, y_test = train_test_split(dev_index, y_all, test_size=splitting_perc, random_state=seed)   #, stratify=y_all çikarildi\n",
    "    print(\"\\nShape of dataframes; train: {}, test: {}, oot: {}\".format(train_index.shape,test_index.shape,oot_index.shape))\n",
    "    print(\"\\nPercentage of dataframes; train: {:.2%}, test: {:.2%}, oot: {:.2%}\".format(len(train_index)/len(df),len(test_index)/len(df),len(oot_index)/len(df)))\n",
    "    return train_index, test_index, oot_index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def split_df_by_index(df, key_var, train_index, test_index, oot_index):\n",
    "    # first get TARGET arrays\n",
    "    y_train, y_test, y_oot = df['target'].iloc[train_index], df['target'].iloc[test_index], df['target'].iloc[oot_index]\n",
    "    \n",
    "    # get key values apart including TARGET\n",
    "    cols_ = df.columns[df.columns.isin(list(key_var))] #including TARGET\n",
    "    df_key_var = df[cols_]  #save key values\n",
    "    df_n = df.drop(cols_, axis=1)\n",
    "    \n",
    "    # set dataframes\n",
    "    X_train, X_test, X_oot = df_n.iloc[train_index], df_n.iloc[test_index], df_n.iloc[oot_index]\n",
    "    train_keys, test_keys, oot_keys = df_key_var.iloc[train_index], df_key_var.iloc[test_index], df_key_var.iloc[oot_index]\n",
    "    \n",
    "    dtrain = xgb.DMatrix(X_train, label=y_train)\n",
    "    dtest = xgb.DMatrix(X_test, label=y_test)\n",
    "    doot = xgb.DMatrix(X_oot, label=y_oot)\n",
    "    del df_n\n",
    "    gc.collect()\n",
    "    return dtrain, dtest, doot, y_train, y_test, y_oot, train_keys, test_keys, oot_keys"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_performance_measurement_results(df, title):\n",
    "    df_sum = df.copy()\n",
    "    \n",
    "    divide_by_score = np.arange(0, 100001, 1000)\n",
    "    divide_by_cnt = 10\n",
    "   \n",
    "    df_sum['score_grp'] = pd.cut(df_sum['predicted'], divide_by_score).astype('category')\n",
    "    temp_score = df_sum.groupby(\"score_grp\")[\"actual\", \"predicted\", \"count\",\"score_grp\"].agg(\"sum\").reset_index() #, observed=True\n",
    "    temp_score=temp_score[temp_score['count']!=0]\n",
    "    temp_score.index = np.arange(1, len(temp_score) + 1)\n",
    "    \n",
    "    df_sum['row'] = np.arange(1, df_sum.shape[0]+1)\n",
    "    df_sum['row_grp'] = pd.cut(df_sum['row'], divide_by_cnt).astype('category')\n",
    "    temp_row = df_sum.groupby(\"row_grp\")[\"actual\", \"predicted\", \"count\",\"score_grp\"].agg(\"sum\").reset_index() #, observed=True \n",
    "    temp_row=temp_row[temp_row['count']!=0]\n",
    "    temp_row.index = np.arange(1, len(temp_row) + 1)\n",
    "    \n",
    "    temp_score['act_avg_pd'] = temp_score[\"actual\"]/temp_score[\"count\"]\n",
    "    temp_score['pred_avg_pd'] = temp_score[\"predicted\"]/temp_score[\"count\"]\n",
    "    temp_score['dist'] = temp_score[\"count\"]/df_sum.shape[0]\n",
    "    \n",
    "    temp_row['act_avg_pd'] = temp_row[\"actual\"]/temp_row[\"count\"]\n",
    "    temp_row['pred_avg_pd'] = temp_row[\"predicted\"]/temp_row[\"count\"]\n",
    "    temp_row['dist'] = temp_row[\"count\"]/df_sum.shape[0]\n",
    "    \n",
    "    fig_score = plt.figure()\n",
    "    plt.plot(temp_score[\"act_avg_pd\"], '-r')  # red\n",
    "    plt.plot(temp_score[\"pred_avg_pd\"], '--b')  # blue\n",
    "    #plt.plot(temp_row[\"count\"],marker=\".\",linestyle=\"\",color=\"tan\")\n",
    "    plt.legend(['Actual', 'Predicted'], loc='upper right', shadow=True)\n",
    "    plt.title(title+\" - Average PD by Score\")\n",
    "    plt.xlabel(\"Score bins\")\n",
    "    plt.ylabel(\"pd\")\n",
    "    plt.close()\n",
    "    \n",
    "    fig_row = plt.figure()\n",
    "    plt.plot(temp_row[\"act_avg_pd\"], '-r')  # red\n",
    "    plt.plot(temp_row[\"pred_avg_pd\"], '--b')  # blue\n",
    "    plt.legend(['Actual', 'Predicted'], loc='upper right', shadow=True)\n",
    "    plt.title(title+\" - Average PD by Population Bins\")\n",
    "    plt.xlabel(\"Population bins\")\n",
    "    plt.ylabel(\"pd\")\n",
    "    plt.close()\n",
    "    del df_sum\n",
    "    gc.collect()\n",
    "    return temp_score, temp_row, fig_score, fig_row"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#gridsearch kapalıysa buradaki parametrelerle aşağıdaki akış çalışır:\n",
    "def set_params(seed): \n",
    "    params = {\n",
    "    # Parameters that we are going to tune.\n",
    "    'eta' : 0.01,\n",
    "    'max_depth': 15, #15\n",
    "    'min_child_weight': 250, #75\n",
    "    'subsample': 1,\n",
    "    'colsample_bytree': 1,\n",
    "    'gamma' : 40,\n",
    "    'reg_alpha' :0.1,\n",
    "    #'scale_pos_weight':  5,\n",
    "    # Other parameters\n",
    "    'objective': \"reg:squarederror\",\n",
    "    'eval_metric' : \"mae\",\n",
    "    'booster' : \"gbtree\",\n",
    "    'seed' : seed\n",
    "    }\n",
    "    return params \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "################################\n",
    "### Part 6 - Run GRID search ###\n",
    "################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def execute_grid_search(df, key_var, grid_params, splitting_perc, min_date, seed, random_search, diff_threshold, grid_size):\n",
    "        \n",
    "    train_index, test_index, oot_index = set_split_index(df, splitting_perc, seed, min_date)\n",
    "    dtrain, dtest, doot, y_train, y_test, y_oot, train_keys, test_keys, oot_keys = split_df_by_index(df, key_var, train_index, test_index, oot_index)\n",
    "    \n",
    "    evallist = [(dtrain, 'train'),(dtest, 'eval')]\n",
    "    \n",
    "    grid_params = expand_grid(grid_params)\n",
    "    \n",
    "    if random_search:\n",
    "        grid_params = grid_params.sample(n=grid_size, replace = False, random_state=seed)\n",
    "        \n",
    "    xgb_gs = pd.DataFrame() #create xgboost grid search dataframe\n",
    "    \n",
    "    iter_count = 0\n",
    "    for i in range(grid_params.shape[0]):\n",
    "        params = {\n",
    "            'eta' : grid_params['eta'].iloc[i],\n",
    "            'max_depth': grid_params['max_depth'].iloc[i],\n",
    "            'min_child_weight': grid_params['min_child_weight'].iloc[i],\n",
    "            'subsample': grid_params['subsample'].iloc[i],\n",
    "            'colsample_bytree': grid_params['colsample_bytree'].iloc[i],\n",
    "            'gamma' : grid_params['gamma'].iloc[i],  # [10, 20, 30, 40]\n",
    "            'reg_alpha' : grid_params['reg_alpha'].iloc[i],\n",
    "            # Other parameters\n",
    "            'objective': grid_params['objective'].iloc[i],\n",
    "            'eval_metric' : grid_params['eval_metric'].iloc[i],\n",
    "            'booster' : grid_params['booster'].iloc[i],\n",
    "            'seed' : seed           \n",
    "            }\n",
    "        iter_count = iter_count+1\n",
    "        print('\\niteration:', iter_count)\n",
    "        bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=500, evals=evallist, early_stopping_rounds=20, verbose_eval=50 )\n",
    "        \n",
    "        pred_train = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)\n",
    "        pred_test = bst.predict(dtest, ntree_limit=bst.best_ntree_limit) \n",
    "        pred_oot = bst.predict(doot, ntree_limit=bst.best_ntree_limit)\n",
    "        \n",
    "        train_df = pd.DataFrame({'sample': 'train', \n",
    "                                  'actual': y_train, \n",
    "                                  'predicted' : pred_train\n",
    "                                #  'FM_CUSTOMER_ID':, \n",
    "                                #  'REPORT_DATE': , \n",
    "                                  })\n",
    "        test_df = pd.DataFrame({'sample': 'test', \n",
    "                                  'actual': y_test, \n",
    "                                  'predicted' : pred_test\n",
    "                                #  'FM_CUSTOMER_ID':, \n",
    "                                #  'REPORT_DATE': , \n",
    "                                  })\n",
    "        oot_df = pd.DataFrame({'sample': 'oot', \n",
    "                                  'actual': y_oot, \n",
    "                                  'predicted' : pred_oot\n",
    "                                #  'FM_CUSTOMER_ID':, \n",
    "                                #  'REPORT_DATE': , \n",
    "                                  })\n",
    "        predictions = train_df.append([test_df, oot_df])\n",
    "        ##predictions[['actual', 'predicted']] = predictions[['actual', 'predicted']].apply(lambda x: round(100*x, 2))\n",
    "\n",
    "        mae_train, rmse_train = cal_mae_and_rmse(predictions, sample='train',pred_name='predicted')\n",
    "        mae_test, rmse_test = cal_mae_and_rmse(predictions, sample='test',pred_name='predicted')\n",
    "        mae_oot, rmse_oot = cal_mae_and_rmse(predictions, sample='oot', pred_name='predicted')\n",
    "        \n",
    "        perf_sum = predictions.groupby(\"sample\").agg(\"mean\").reset_index()\n",
    "        #perf_sum[['actual', 'predicted']] = perf_sum[['actual', 'predicted']].apply(lambda x: round(100*x, 2))\n",
    "        \n",
    "        perf_sum['mae'] = 0\n",
    "        perf_sum.loc[perf_sum['sample'] == 'train', 'mae'] = mae_train\n",
    "        perf_sum.loc[perf_sum['sample'] == 'test', 'mae'] = mae_test\n",
    "        perf_sum.loc[perf_sum['sample'] == 'oot', 'mae'] = mae_oot\n",
    "        \n",
    "        perf_sum['rmse'] = 0\n",
    "        perf_sum.loc[perf_sum['sample'] == 'train', 'rmse'] = rmse_train\n",
    "        perf_sum.loc[perf_sum['sample'] == 'test', 'rmse'] = rmse_test\n",
    "        perf_sum.loc[perf_sum['sample'] == 'oot', 'rmse'] = rmse_oot\n",
    "\n",
    "        \n",
    "        perf_sum['eta'] = grid_params['eta'].iloc[i]\n",
    "        perf_sum['max_depth'] = grid_params['max_depth'].iloc[i]\n",
    "        perf_sum['min_child_weight'] = grid_params['min_child_weight'].iloc[i]\n",
    "        perf_sum['subsample'] = grid_params['subsample'].iloc[i]\n",
    "        perf_sum['colsample_bytree'] = grid_params['colsample_bytree'].iloc[i]\n",
    "        perf_sum['gamma'] = grid_params['gamma'].iloc[i]\n",
    "        perf_sum['reg_alpha'] = grid_params['reg_alpha'].iloc[i]\n",
    "        perf_sum['objective'] = grid_params['objective'].iloc[i]\n",
    "        perf_sum['eval_metric'] = grid_params['eval_metric'].iloc[i]\n",
    "        perf_sum['booster'] = grid_params['booster'].iloc[i]\n",
    "        perf_sum['grid_iter'] = iter_count\n",
    "        \n",
    "        xgb_gs = xgb_gs.append(perf_sum, ignore_index=True)\n",
    "        \n",
    "        del rmse_train, rmse_test, rmse_oot, mae_train, mae_test, mae_oot, bst, predictions, train_df, test_df, oot_df, pred_train, pred_test, pred_oot\n",
    "        gc.collect()\n",
    "        \n",
    "    print(\"\\n----selecting best grid iteration----\")\n",
    "    #select best grid results\n",
    "    xgb_gs_train = xgb_gs.loc[xgb_gs['sample'] == \"train\"]\n",
    "    xgb_gs_train = xgb_gs_train.sort_values('mae', ascending=False)\n",
    "    \n",
    "    best_gs_iter = np.nan\n",
    "    xgb_gs_train['trainMae'] = np.nan\n",
    "    xgb_gs_train['testMae'] = np.nan\n",
    "    xgb_gs_train['ootMae'] = np.nan\n",
    "    xgb_gs_train['testDiff'] = np.nan\n",
    "    xgb_gs_train['ootDiff'] = np.nan\n",
    "    xgb_gs_train['avgDiff'] = np.nan\n",
    "    \n",
    "    for i in xgb_gs_train.index:\n",
    "        iter_ = xgb_gs_train.loc[i, 'grid_iter']\n",
    "            \n",
    "        train_mae = xgb_gs.loc[(xgb_gs['grid_iter'] == iter_) & (xgb_gs['sample']=='train'), 'mae'].iloc[0]\n",
    "        test_mae = xgb_gs.loc[(xgb_gs['grid_iter'] == iter_) & (xgb_gs['sample']=='test'), 'mae'].iloc[0]\n",
    "        oot_mae = xgb_gs.loc[(xgb_gs['grid_iter'] == iter_) & (xgb_gs['sample']=='oot'), 'mae'].iloc[0]    \n",
    "        \n",
    "        xgb_gs_train.loc[i, 'trainMae'] = train_mae\n",
    "        xgb_gs_train.loc[i, 'testMae'] = test_mae\n",
    "        xgb_gs_train.loc[i, 'ootMae'] = oot_mae\n",
    "        xgb_gs_train.loc[i, 'testDiff'] = abs(train_mae - test_mae)\n",
    "        xgb_gs_train.loc[i, 'ootDiff'] = abs(train_mae - oot_mae)\n",
    "        xgb_gs_train.loc[i, 'avgDiff'] = (abs(train_mae - test_mae) + abs(train_mae - oot_mae))/2\n",
    "\n",
    "    del i, iter_, train_mae, test_mae, oot_mae\n",
    "    gc.collect()\n",
    "\n",
    "    xgb_gs_train_base = xgb_gs_train.copy()\n",
    "    xgb_gs_train = xgb_gs_train.loc[xgb_gs_train['mae'] > 0].copy()\n",
    "    minAvgDiff = xgb_gs_train['avgDiff'].min()\n",
    "\n",
    "    if minAvgDiff <= diff_threshold:\n",
    "        xgb_gs_train_f = xgb_gs_train.loc[xgb_gs_train['avgDiff'] <= diff_threshold].copy()\n",
    "        min_train_mae = xgb_gs_train_f['trainMae'].min()\n",
    "        \n",
    "        xgb_gs_train_f['StdDev'] = np.nan\n",
    "        xgb_gs_train_f['trainMaePrevious'] = min_train_mae\n",
    "        xgb_gs_train_f['trainMaePreviousDiff'] = np.nan\n",
    "        xgb_gs_train_f['finalEvaluationScore'] = np.nan\n",
    "        \n",
    "        for i in xgb_gs_train_f.index: \n",
    "            train_mae = xgb_gs_train_f.loc[i, 'trainMae']\n",
    "            test_mae = xgb_gs_train_f.loc[i, 'testMae']\n",
    "            oot_mae = xgb_gs_train_f.loc[i, 'ootMae']\n",
    "            xgb_gs_train_f.loc[i, 'StdDev'] = stdev([train_mae,test_mae,oot_mae])\n",
    "            xgb_gs_train_f.loc[i, 'trainMaePreviousDiff'] = train_mae - min_train_mae\n",
    "            xgb_gs_train_f.loc[i, 'finalEvaluationScore'] = ((0.50 * xgb_gs_train_f.loc[i, 'trainMaePreviousDiff']) + (0.20 * xgb_gs_train_f.loc[i, 'avgDiff']) + (0.3 * xgb_gs_train_f.loc[i, 'StdDev']))\n",
    "            \n",
    "            \n",
    "        xgb_gs_train = pd.merge(xgb_gs_train,xgb_gs_train_f[['grid_iter','StdDev', 'trainMaePrevious', 'trainMaePreviousDiff','finalEvaluationScore']],on='grid_iter', how='left')\n",
    "        min_final_eva_score = copy.copy(xgb_gs_train_f['finalEvaluationScore'].min())\n",
    "        best_gs_iter = xgb_gs_train_f.loc[xgb_gs_train_f['finalEvaluationScore'] <= min_final_eva_score, 'grid_iter'].iloc[0]\n",
    "        xgb_gs_train = pd.merge(xgb_gs_train_base,xgb_gs_train[['grid_iter','StdDev', 'trainMaePrevious', 'trainMaePreviousDiff','finalEvaluationScore']],on='grid_iter', how='left')\n",
    "        del i, train_mae, test_mae, oot_mae\n",
    "        gc.collect()\n",
    "    else:\n",
    "        min_diff = xgb_gs_train['avgDiff'].min()\n",
    "        min_diff_mae = xgb_gs_train.loc[xgb_gs_train['avgDiff'] <= minAvgDiff, 'trainMae'].iloc[0]\n",
    "        for i in xgb_gs_train.index:\n",
    "            #iter_ = j['grid_iter']\n",
    "            diff_increase = xgb_gs_train.loc[i, 'avgDiff'] - min_diff\n",
    "            mae_decrease = xgb_gs_train.loc[i, 'trainMae'] - min_diff_mae\n",
    "            xgb_gs_train.loc[i, 'finalEvaluationScore'] = 5*diff_increase - mae_decrease\n",
    "        \n",
    "        min_final_eva_score = xgb_gs_train['finalEvaluationScore'].min()\n",
    "        best_gs_iter = xgb_gs_train.loc[xgb_gs_train['finalEvaluationScore'] <= min_final_eva_score, 'grid_iter'].iloc[0]\n",
    "        xgb_gs_train = xgb_gs_train.reset_index(drop=True)\n",
    "        xgb_gs_train = pd.merge(xgb_gs_train_base,xgb_gs_train[['grid_iter','finalEvaluationScore']],on='grid_iter', how='left')\n",
    "        del i, diff_increase, mae_decrease, min_diff, min_diff_mae\n",
    "        gc.collect()\n",
    "    print(\"best grid iteration:\", best_gs_iter)\n",
    "    \n",
    "    params = {\n",
    "    # Parameters that we are going to tune.\n",
    "    'eta' : xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'eta'].iloc[0],\n",
    "    'max_depth': xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'max_depth'].iloc[0],\n",
    "    'min_child_weight': xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'min_child_weight'].iloc[0],\n",
    "    'subsample': xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'subsample'].iloc[0],\n",
    "    'colsample_bytree': xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'colsample_bytree'].iloc[0],\n",
    "    'gamma' : xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'gamma'].iloc[0],\n",
    "    'reg_alpha' : xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'reg_alpha'].iloc[0],\n",
    "    # Other parameters\n",
    "    'objective': xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'objective'].iloc[0],\n",
    "    'eval_metric' : xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'eval_metric'].iloc[0],\n",
    "    'booster' : xgb_gs_train.loc[xgb_gs_train['grid_iter'] == best_gs_iter,'booster'].iloc[0],\n",
    "    'seed' : seed\n",
    "    }\n",
    "    xgb_gs_train[['actual', 'predicted']] = xgb_gs_train[['actual', 'predicted']].apply(lambda x: round(x, 2))\n",
    "\n",
    "    return xgb_gs_train, best_gs_iter, params"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###############################################################\n",
    "### Part 7 - Run final model with the best grid parameters ###\n",
    "###############################################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def execute_final_model(df, key_var, params, splitting_perc, seed, num_boost_round, early_stopping_rounds, verbose_eval):\n",
    "    print(\"\\n----final model----\")\n",
    "    train_index, test_index, oot_index = set_split_index(df, splitting_perc, seed, min_date)\n",
    "    dtrain, dtest, doot, y_train, y_test, y_oot, train_keys, test_keys, oot_keys = split_df_by_index(df, key_var, train_index, test_index, oot_index)\n",
    "    print(\"\\n\")\n",
    "    \n",
    "    evallist = [(dtrain, 'train'),(dtest, 'eval')]\n",
    "    \n",
    "    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round, evals=evallist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval )\n",
    "    \n",
    "    pred_train = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)\n",
    "    pred_test = bst.predict(dtest, ntree_limit=bst.best_ntree_limit) \n",
    "    pred_oot = bst.predict(doot, ntree_limit=bst.best_ntree_limit)\n",
    "    \n",
    "    train_df = pd.DataFrame({'sample': 'train', \n",
    "                              'actual': y_train, \n",
    "                              'predicted' : pred_train,\n",
    "                              'contactcode': train_keys[\"contactcode\"], \n",
    "                              'refdate': train_keys[\"refdate\"]\n",
    "                              #'APP_ID': train_keys[\"APP_ID\"]\n",
    "                              })\n",
    "    \n",
    "    test_df = pd.DataFrame({'sample': 'test', \n",
    "                             'actual': y_test, \n",
    "                             'predicted' : pred_test,\n",
    "                              'contactcode': test_keys[\"contactcode\"], \n",
    "                              'refdate': test_keys[\"refdate\"]\n",
    "                             #'APP_ID': test_keys[\"APP_ID\"]\n",
    "                              })\n",
    "    oot_df = pd.DataFrame({'sample': 'oot', \n",
    "                            'actual': y_oot, \n",
    "                            'predicted' : pred_oot,\n",
    "                              'contactcode': oot_keys[\"contactcode\"], \n",
    "                              'refdate': oot_keys[\"refdate\"]\n",
    "                             #'APP_ID': oot_keys[\"APP_ID\"]\n",
    "                              })\n",
    "    predictions = train_df.append([test_df, oot_df])\n",
    "    del train_df, test_df, oot_df, pred_train, pred_test, pred_oot\n",
    "    gc.collect()\n",
    "    perf_sum = predictions.groupby(\"sample\").agg(\"mean\").reset_index()#.drop(columns='contactcode')\n",
    "    #perf_sum[['actual', 'predicted']] = perf_sum[['actual', 'predicted']].apply(lambda x: round(100*x, 2))\n",
    "    predictions.loc[:,'count'] = 1\n",
    "    \n",
    "    mae_train,rmse_train = cal_mae_and_rmse(predictions, sample='train', pred_name='predicted')\n",
    "    mae_test, rmse_test = cal_mae_and_rmse(predictions, sample='test', pred_name='predicted')\n",
    "    mae_oot, rmse_oot = cal_mae_and_rmse(predictions, sample='oot', pred_name='predicted')\n",
    "   \n",
    "    perf_sum['mae'] = 0\n",
    "    perf_sum.loc[perf_sum['sample'] == 'train', 'mae'] = mae_train\n",
    "    perf_sum.loc[perf_sum['sample'] == 'test', 'mae'] = mae_test\n",
    "    perf_sum.loc[perf_sum['sample'] == 'oot', 'mae'] = mae_oot\n",
    "    \n",
    "    perf_sum['rmse'] = 0\n",
    "    perf_sum.loc[perf_sum['sample'] == 'train', 'rmse'] = rmse_train\n",
    "    perf_sum.loc[perf_sum['sample'] == 'test', 'rmse'] = rmse_test\n",
    "    perf_sum.loc[perf_sum['sample'] == 'oot', 'rmse'] = rmse_oot\n",
    "    \n",
    "  \n",
    "    save_params = params.copy()\n",
    "    save_params[\"num_boost_round\"] = num_boost_round\n",
    "    save_params[\"early_stopping_rounds\"] = early_stopping_rounds\n",
    "    save_params[\"verbose_eval\"] = verbose_eval\n",
    "    perf_sum[\"parameters\"] = str(save_params)\n",
    "    \n",
    "    ## feature importance \n",
    "    \n",
    "    # ‘weight’: the number of times a feature is used to split the data across all trees.\n",
    "    # ‘gain’: the average gain across all splits the feature is used in.\n",
    "    # ‘cover’: the average coverage across all splits the feature is used in.\n",
    "    # ‘total_gain’: the total gain across all splits the feature is used in.\n",
    "    # ‘total_cover’: the total coverage across all splits the feature is used in.\n",
    "    \n",
    "    importance_xgb = pd.DataFrame.from_dict(data=bst.get_score(importance_type='weight'), orient='index')\n",
    "    importance_xgb = importance_xgb.reset_index()\n",
    "    importance_xgb.columns = [\"features\",\"weight\"]\n",
    "    \n",
    "    for i in ['gain', 'cover', 'total_gain', 'total_cover']:\n",
    "        temp_imp = pd.DataFrame.from_dict(data=bst.get_score(importance_type=i), orient='index')\n",
    "        temp_imp.columns = [i]\n",
    "        importance_xgb = importance_xgb.merge(temp_imp, left_on=\"features\", right_index=True)\n",
    "        del temp_imp\n",
    "        gc.collect()\n",
    "        \n",
    "    importance_xgb[\"total_gain_score\"] =  importance_xgb[\"total_gain\"] / importance_xgb[\"total_gain\"].sum()\n",
    "    importance_xgb = importance_xgb.sort_values('total_gain_score', ascending=False)\n",
    "    del rmse_train, rmse_test, rmse_oot, mae_train, mae_test, mae_oot, save_params\n",
    "    gc.collect()\n",
    "    return bst, predictions, perf_sum, importance_xgb "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## performance measurement"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def performance_measurement():\n",
    "    # performance measurement\n",
    "    train_sum_by_pred_score, train_sum_by_pred_row, fig_train_score, fig_train_row = create_performance_measurement_results(predictions[predictions['sample'] == 'train'], title=\"Train\")\n",
    "    test_sum_by_pred_score, test_sum_by_pred_row, fig_test_score, fig_test_row = create_performance_measurement_results(predictions[predictions['sample'] == 'test'], title=\"Test\")\n",
    "    oot_sum_by_pred_score, oot_sum_by_pred_row, fig_oot_score, fig_oot_row = create_performance_measurement_results(predictions[predictions['sample'] == 'oot'], title=\"Oot\")\n",
    "    return train_sum_by_pred_score, train_sum_by_pred_row, fig_train_score, fig_train_row, test_sum_by_pred_score, test_sum_by_pred_row, fig_test_score, fig_test_row, oot_sum_by_pred_score, oot_sum_by_pred_row, fig_oot_score, fig_oot_row\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "############################\n",
    "##performance for segments##\n",
    "############################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculate_mae_mape_for_segment(dfPredictions, segmentType, segmentTypeGroup, pred_name):\n",
    "  \n",
    "    try:\n",
    "        temp = dfPredictions.copy()\n",
    "        \n",
    "        if segmentType == \"all\":\n",
    "           mae_ = mean_absolute_error(temp['actual'], temp[pred_name])\n",
    "           mape_ = np.mean(np.abs((temp['actual']-temp[pred_name])/temp['actual']))\n",
    "        else:\n",
    "           mae_ = mean_absolute_error(temp.loc[temp[segmentType] == segmentTypeGroup]['actual'], temp.loc[temp[segmentType] == segmentTypeGroup][pred_name])\n",
    "           mape_ = np.mean(np.abs((temp.loc[temp[segmentType] == segmentTypeGroup]['actual']-temp.loc[temp[segmentType] == segmentTypeGroup][pred_name])/temp.loc[temp[segmentType] == segmentTypeGroup]['actual'])) \n",
    "        \n",
    "    except:    \n",
    "        print('Mae is not calculated for:' , segmentType, segmentTypeGroup)\n",
    "        mae_ = \"not calculated\"\n",
    "        mape_ = \"not calculated\"\n",
    "\n",
    "    del temp\n",
    "    gc.collect()\n",
    "    \n",
    "    return mae_,mape_ "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluate_performance_by_segment(df):\n",
    "    # performance measurement by different segmentation scenarios    \n",
    "    temp=df.copy()\n",
    "    #validation for PERIOD\n",
    "    temp['date'] = pd.to_datetime(temp['refdate'])\n",
    "    temp['year'] = temp['date'].dt.year\n",
    "    temp['month'] = temp['date'].dt.month\n",
    "    temp['PERIOD'] = temp['year'].astype(str) + temp['month'].astype(str)\n",
    "    \n",
    "    temp=temp.drop(['date','year','month'],axis=1)\n",
    "    #validation for ACTUALINCOMEBUCKET\n",
    "    divide_by_score=np.arange(0, 100001, 1000)\n",
    "    temp['PREDINCOMEBUCKET'] = pd.cut(temp['predicted'], divide_by_score).astype('category')\n",
    "   \n",
    "    temp['ACTINCOMEBUCKET'] = pd.cut(temp['actual'], divide_by_score).astype('category')\n",
    "\n",
    "    \n",
    "    segment_summary_all_sample = temp.groupby(\"count\")[\"actual\", \"predicted\", \"count\"].agg(\"sum\")\n",
    "    segment_summary_all_sample.insert(loc = 0, column = 'segmentType', value = \"all\")\n",
    "    segment_summary_all_sample.insert(loc = 0, column = 'segmentTypeGroup', value = \"all\")\n",
    "    \n",
    "    segment_summary_period = temp.groupby(\"PERIOD\")[\"actual\", \"predicted\", \"count\"].agg(\"sum\").reset_index() \n",
    "    segment_summary_period=segment_summary_period[segment_summary_period['count']!=0]\n",
    "    segment_summary_period.insert(loc = 0, column = 'segmentType', value = \"PERIOD\")\n",
    "    segment_summary_period.rename(columns={'PERIOD':'segmentTypeGroup'}, inplace = True) \n",
    "\n",
    "    segment_summary_income_bucket = temp.groupby(\"PREDINCOMEBUCKET\")[\"actual\", \"predicted\", \"count\"].agg(\"sum\").reset_index() \n",
    "    segment_summary_income_bucket=segment_summary_income_bucket[segment_summary_income_bucket['count']!=0]\n",
    "    segment_summary_income_bucket.insert(loc = 0, column = 'segmentType', value = \"PREDINCOMEBUCKET\")\n",
    "    segment_summary_income_bucket.rename(columns={'PREDINCOMEBUCKET':'segmentTypeGroup'}, inplace = True)\n",
    "\n",
    "    segment_summary_decincome_bucket = temp.groupby(\"ACTINCOMEBUCKET\")[\"actual\", \"predicted\", \"count\"].agg(\"sum\").reset_index() \n",
    "    segment_summary_decincome_bucket=segment_summary_decincome_bucket[segment_summary_decincome_bucket['count']!=0]\n",
    "    segment_summary_decincome_bucket.insert(loc = 0, column = 'segmentType', value = \"ACTINCOMEBUCKET\")\n",
    "    segment_summary_decincome_bucket.rename(columns={'ACTINCOMEBUCKET':'segmentTypeGroup'}, inplace = True)\n",
    "\n",
    "\n",
    "    segment_summary_all = segment_summary_all_sample.append([segment_summary_period, segment_summary_income_bucket, segment_summary_decincome_bucket])\n",
    "    del segment_summary_all_sample, segment_summary_period, segment_summary_income_bucket, segment_summary_decincome_bucket\n",
    "    \n",
    "    segment_summary_all[\"avg_pd\"] = segment_summary_all[\"predicted\"]/segment_summary_all[\"count\"]\n",
    "   \n",
    "    segment_summary_all[\"mae\"] = np.nan\n",
    "    segment_summary_all[\"mape\"] = np.nan\n",
    "\n",
    "    \n",
    "    for segmentType in pd.unique(segment_summary_all['segmentType']):\n",
    "        for segmentTypeGroup in pd.unique(segment_summary_all.loc[segment_summary_all['segmentType']== segmentType]['segmentTypeGroup']):\n",
    "            maeSegment,mapeSegment = calculate_mae_mape_for_segment(temp, segmentType, segmentTypeGroup, \"predicted\")\n",
    "            segment_summary_all.loc[(segment_summary_all['segmentType'] == segmentType) & (segment_summary_all['segmentTypeGroup'] == segmentTypeGroup), 'mae'] = maeSegment\n",
    "            segment_summary_all.loc[(segment_summary_all['segmentType'] == segmentType) & (segment_summary_all['segmentTypeGroup'] == segmentTypeGroup), 'mape'] = mapeSegment\n",
    "\n",
    "    return segment_summary_all      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "def performance_measurement_for_segment():\n",
    "    # performance measurement\n",
    "    segment_summary_all_train = evaluate_performance_by_segment(predictions[predictions['sample'] == 'train'])\n",
    "    segment_summary_all_test = evaluate_performance_by_segment(predictions[predictions['sample'] == 'test'])\n",
    "    segment_summary_all_oot = evaluate_performance_by_segment(predictions[predictions['sample'] == 'oot'])\n",
    "    segment_summary_all = evaluate_performance_by_segment(predictions)\n",
    "\n",
    "    return segment_summary_all_train, segment_summary_all_test,segment_summary_all_oot, segment_summary_all"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####################\n",
    "### Save Results ###\n",
    "####################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_results(path):\n",
    "    print(\"\\nSaving Results...\")\n",
    "    # create folder\n",
    "    if not os.path.exists(path):\n",
    "        os.mkdir(path)\n",
    "    \n",
    "    # save grid search results\n",
    "    if grid_search_:\n",
    "        xgb_gs_train.to_excel(path+\"fullgrid.xlsx\" )\n",
    "    \n",
    "    # save predictions\n",
    "#    predictions_calb.to_csv(path+\"predictions_calb.csv\" )\n",
    "    \n",
    "    # save performance results\n",
    "    perf_sum.to_excel(path+\"performance_summary.xlsx\" )\n",
    "    \n",
    "    # save comparison plot - actual vs predicted\n",
    "    with matplotlib.backends.backend_pdf.PdfPages(path+'Performance_Details.pdf') as pdf:\n",
    "        for i in [fig_train_score, fig_train_row, fig_test_score, fig_test_row, fig_oot_score, fig_oot_row]:\n",
    "            pdf.savefig(i)\n",
    "        \n",
    "    # save importance features' information\n",
    "    importance_xgb.to_excel(path+\"importance_xgb.xlsx\")   \n",
    "    \n",
    "    \n",
    "    # save model performance summary data\n",
    "    perf_dict = {'train_score': train_sum_by_pred_score, 'train_count': train_sum_by_pred_row,\n",
    "                 'test_score': test_sum_by_pred_score, 'test_count': test_sum_by_pred_row,\n",
    "                 'oot_score': oot_sum_by_pred_score, 'oot_count': oot_sum_by_pred_row}\n",
    "    \n",
    "    with pd.ExcelWriter(path+'Performance_Details_data.xlsx') as writer:\n",
    "        for name_ in perf_dict:\n",
    "            perf_dict[name_].to_excel(writer, sheet_name=name_)\n",
    "    \n",
    "    del perf_dict\n",
    "    gc.collect()\n",
    "    \n",
    "    # save gini performance by various segment breakdown\n",
    "    #segment_summary_all.to_excel(path+\"segment_breakdown.xlsx\")\n",
    "    \n",
    "    # save model performance summary data\n",
    "    #perf_dict = {'summary_all_train': segment_summary_all_train,\n",
    "    #             'summary_all_test': segment_summary_all_test,\n",
    "    #             'summary_all_oot': segment_summary_all_oot,\n",
    "    #             'summary_all': segment_summary_all}\n",
    "    \n",
    "    #with pd.ExcelWriter(path+'Performance_Details_for_segment_data.xlsx') as writer:\n",
    "    #    for name_ in perf_dict:\n",
    "    #        perf_dict[name_].to_excel(writer, sheet_name=name_)\n",
    "    \n",
    "    #del perf_dict\n",
    "    #gc.collect()  \n",
    "    \n",
    "    \n",
    "    # save xgb model object\n",
    "    pickle.dump(bst, open(path+\"model_xgb.pkl\", 'wb'))\n",
    "    # load the xgb model from disk\n",
    "    #loaded_model = pickle.load(open(path+filename, 'rb'))\n",
    "    \n",
    "    # save pmml object\n",
    "    #sklearn2pmml(pipeline, path + str(\"xgboost_model.pmml\"), with_repr=True)\n",
    "    \n",
    "    #save pmml predictions\n",
    "    #pred_all.to_excel(path + str('predictions_with_pmml.xlsx'))\n",
    "    \n",
    "    # save pmml object\n",
    "    #sklearn2pmml(pipeline_ren, path + str(\"xgboost_model_renamed.pmml\"), with_repr=True)\n",
    "    \n",
    "    #save pmml predictions\n",
    "    #pred_all_ren.to_excel(path + str('predictions_with_pmml_renamed.xlsx'))\n",
    "    \n",
    "    # save character encoding matrix\n",
    "    #encoding_matrix.to_excel(path+\"encoding_matrix.xlsx\")\n",
    "    #encoding_matrix.to_pickle(path+\"encoding_matrix.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "def execute_preprocess(df, max_threshold):\n",
    "    '''\n",
    "    execute all functions defined for preprocessing\n",
    "    '''\n",
    "    na_report = create_na_report(df)\n",
    "    num_cols, char_cols = create_dtype_report(df)\n",
    "    new_df = eliminate_missing(df, na_report, max_threshold)\n",
    "    new_df = eliminate_single_unique(new_df)\n",
    "    #enc_df = create_encoding_matrix(new_df)\n",
    "    #new_df = encode_from_matrix(new_df, enc_df)\n",
    "    \n",
    "    #save enc_df for later use in prediction code\n",
    "    gc.collect()\n",
    "    print(\"\\nPreprocess is done...\")\n",
    "    return new_df#, enc_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def split_df_by_index_pmml(df, key_var, train_index, test_index, oot_index):\n",
    "    # first get TARGET arrays\n",
    "    y_train, y_test, y_oot = df['target'].iloc[train_index], df['target'].iloc[test_index], df['target'].iloc[oot_index]\n",
    "    \n",
    "    # get key values apart including TARGET\n",
    "    cols_ = df.columns[df.columns.isin(list(key_var))] #including TARGET\n",
    "    df_key_var = df[cols_]  #save key values\n",
    "    df_n = df.drop(cols_, axis=1)\n",
    "    \n",
    "    # set dataframes\n",
    "    X_train, X_test, X_oot = df_n.iloc[train_index], df_n.iloc[test_index], df_n.iloc[oot_index]\n",
    "    train_keys, test_keys, oot_keys = df_key_var.iloc[train_index], df_key_var.iloc[test_index], df_key_var.iloc[oot_index]\n",
    "\n",
    "    del df_n\n",
    "    gc.collect()\n",
    "    return X_train, X_test, X_oot, y_train, y_test, y_oot, train_keys, test_keys, oot_keys\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Modelling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "conn = pyodbc.connect('Driver={SQL Server};' \n",
    "                      'Server=DEVELOPMENT-01;'\n",
    "                      'Database=Erc_Project;'\n",
    "                      'Trusted_Connection=yes;')\n",
    "cursor = conn.cursor()\n",
    "data = pd.read_sql_query((\"\"\" SELECT a.contactcode,a.refdate,a.target\n",
    "\n",
    ",BusinessUnitCode\n",
    ",AssetManagementCode\n",
    ",ReachStatusCode\n",
    ",FinancialStatusCode\n",
    ",FirstReachDate_monthdiff\n",
    ",FollowingStartDate_monthdiff\n",
    ",Sensibility\n",
    ",IsTCCitizen\n",
    ",ContactTypeCode\n",
    ",Age\n",
    ",GenderCode\n",
    ",MaritalStatusCode\n",
    ",CityCode\n",
    ",ExistingFlag\n",
    ",kefil_cnt\n",
    ",min_acqdate_open_diff\n",
    ",gsmsms_cnt\n",
    ",CustomerDifficulty\n",
    ",restate_haciz_cnt\n",
    ",restate_aktif_TenderPrice\n",
    ",restate_haciz_TenderPrice\n",
    ",restate_ipotek_TenderPrice\n",
    ",restate_pasif_TenderPrice\n",
    ",restate_aktif_MarketValue\n",
    ",restate_haciz_MarketValue\n",
    ",restate_pasif_MarketValue\n",
    ",ticari_arac_anlasma_cnt\n",
    ",ticari_arac_icra_tenderprice\n",
    ",otomobil_icra_tenderprice\n",
    ",ticari_arac_diger_tenderprice\n",
    ",otomobil_diger_tenderprice\n",
    ",ticari_arac_icra_NetPaymentAmount\n",
    ",ticari_arac_anlasma_MarketingValue\n",
    ",otomobil_anlasma_MarketingValue\n",
    ",ticari_arac_icra_MarketingValue\n",
    ",ticari_arac_deger_yok_MarketingValue\n",
    ",otomobil_deger_yok_MarketingValue\n",
    ",motor_deger_yok_MarketingValue\n",
    ",diger_deger_yok_MarketingValue\n",
    ",ticari_arac_yakalama_MarketingValue\n",
    ",otomobil_yakalama_MarketingValue\n",
    ",diger_yakalama_MarketingValue\n",
    ",ticari_arac_diger_MarketingValue\n",
    ",otomobil_diger_MarketingValue\n",
    ",diger_MarketingValue\n",
    ",tah_icra_odemesi\n",
    ",tah_icra_odemesi_avg_l3m\n",
    ",tah_icra_satis_avg_l3m\n",
    ",tah_diger_avg_l3m\n",
    ",outbound_call_cnt_max_l3m\n",
    ",tah_odeme_teklifi_min_l3m\n",
    ",tah_total_min_l3m\n",
    ",outbound_call_cnt_max_l6m\n",
    ",gsmsms_cnt_max_l6m\n",
    ",tah_icra_odemesi_max_l6m\n",
    ",tcknsms_cnt_min_l6m\n",
    ",IntAmount_UPB_rat_avg_l12m\n",
    ",sum_ExpenseAmount_avg_l12m\n",
    ",tah_total_avg_l12m\n",
    ",Current_UPB_Debt_rat_min_l12m\n",
    ",tah_odeme_teklifi_min_l12m\n",
    ",Current_UPB_Debt_rat_max_l12m\n",
    ",outbound_call_cnt_max_l12m\n",
    ",gsmsms_cnt_max_l12m\n",
    ",tcknsms_cnt_max_l12m\n",
    ",tah_icra_odemesi_max_l12m\n",
    ",tah_diger_max_l12m\n",
    ",min_cutoffdate_diff\n",
    ",acq_tot_debt\n",
    ",min_noticedate_diff\n",
    ",bankcode_cnt\n",
    ",AcqPaymentTRY\n",
    ",loantype1_cnt\n",
    ",loantype2_cnt\n",
    ",loantype3_cnt\n",
    ",loantype4_cnt\n",
    ",loantype5_cnt\n",
    ",loantype6_cnt\n",
    ",loantype7_cnt\n",
    ",loantype8_cnt\n",
    ",mirasci_flg\n",
    ",call_duration\n",
    ",first_call_diff\n",
    ",KKBbankaTSmonth\n",
    ",MaxÜrünKapatmaTarihi\n",
    ",xDevirBakiye\n",
    ",MaxVYSKapatmaTarihi_diff\n",
    ",ToplamMemzuc\n",
    ",ProtocolAmount\n",
    ",ActivatedProtocolAmount\n",
    ",BrokeProtocolAmount\n",
    ",CanceledProtocolAmount\n",
    ",LiveProtocolAmount\n",
    ",TotalProtocolAmount\n",
    ",masraf_tutari\n",
    ",meslek_L\n",
    ",tenure_open\n",
    ",BrokeProtocolAmount_min_l3m\n",
    ",ProtocolAmount_min_l6m\n",
    ",ActivatedProtocolAmount_min_l6m\n",
    ",BrokeProtocolAmount_min_l6m\n",
    ",CanceledProtocolAmount_min_l6m\n",
    ",LiveProtocolAmount_avg_l6m\n",
    ",LiveProtocolCount_max_l6m\n",
    ",FulledTotalProtocolAmount_max_l6m\n",
    ",TotalProtocolAmount_max_l6m\n",
    ",BrokeProtocolAmount_min_l12m\n",
    ",TotalProtocolAmount_min_l12m\n",
    ",ProtocolAmount_avg_l12m\n",
    ",CanceledProtocolAmount_avg_l12m\n",
    ",LiveProtocolCount_max_l12m\n",
    ",ActivatedProtocolAmount_max_l12m\n",
    ",LiveProtocolAmount_max_l12m\n",
    ",Sensibility_avg_rat_l1m_l3m\n",
    ",Current_UPB_Debt_rat_avg_rat_l1m_l3m\n",
    ",ExpenseAmount_Debt_rat_avg_rat_l1m_l3m\n",
    ",Inbound_call_20_cnt_avg_rat_l1m_l3m\n",
    ",Inbound_call_cnt_avg_rat_l1m_l3m\n",
    ",Manuel_call_20_cnt_avg_rat_l1m_l3m\n",
    ",Manuel_call_cnt_avg_rat_l1m_l3m\n",
    ",SalaryConfiscationFlag_avg_rat_l1m_l3m\n",
    ",tcknsms_cnt_avg_rat_l1m_l6m\n",
    ",ActivatedProtocolCount_avg_rat_l1m_l6m\n",
    ",BrokeProtocolCount_avg_rat_l1m_l6m\n",
    ",outbound_call_20_cnt_avg_rat_l3m_l6m\n",
    ",Inbound_call_20_cnt_avg_rat_l3m_l6m\n",
    ",Inbound_call_cnt_avg_rat_l3m_l6m\n",
    ",Manuel_call_20_cnt_avg_rat_l3m_l6m\n",
    ",Manuel_call_cnt_avg_rat_l3m_l6m\n",
    ",tah_icra_satis_avg_rat_l3m_l6m\n",
    ",ActivatedProtocolCount_avg_rat_l3m_l6m\n",
    ",BrokeProtocolCount_avg_rat_l3m_l6m\n",
    ",LiveProtocolAmount_avg_rat_l3m_l6m\n",
    ",ActivatedProtocolCount_avg_rat_l1m_l12m\n",
    ",FullyPaidProtocolCount_avg_rat_l1m_l12m\n",
    ",LiveProtocolCount_avg_rat_l1m_l12m\n",
    ",ProtocolAmount_avg_rat_l1m_l12m\n",
    ",BrokeProtocolAmount_avg_rat_l1m_l12m\n",
    ",outbound_call_20_cnt_avg_rat_l3m_l12m\n",
    ",ActivatedProtocolCount_avg_rat_l3m_l12m\n",
    ",CanceledProtocolCount_avg_rat_l3m_l12m\n",
    ",FullyPaidProtocolCount_avg_rat_l3m_l12m\n",
    ",ProtocolAmount_avg_rat_l3m_l12m\n",
    ",CanceledProtocolAmount_avg_rat_l3m_l12m\n",
    ",TotalProtocolAmount_avg_rat_l3m_l12m\n",
    ",Inbound_call_20_cnt_avg_rat_l6m_l12m\n",
    ",Inbound_call_cnt_avg_rat_l6m_l12m\n",
    ",Manuel_call_20_cnt_avg_rat_l6m_l12m\n",
    ",Manuel_call_cnt_avg_rat_l6m_l12m\n",
    ",tah_icra_odemesi_Debt_rat_avg_rat_l6m_l12m\n",
    ",tah_icra_satis_avg_rat_l6m_l12m\n",
    ",tah_diger_Debt_rat_avg_rat_l6m_l12m\n",
    ",BrokeProtocolCount_avg_rat_l6m_l12m\n",
    ",CanceledProtocolCount_avg_rat_l6m_l12m\n",
    ",FullyPaidProtocolCount_avg_rat_l6m_l12m\n",
    ",LiveProtocolCount_avg_rat_l6m_l12m\n",
    ",NewProtocolCount_avg_rat_l6m_l12m\n",
    ",outbound_call_20_cnt_min_rat_l1m_l3m\n",
    ",Inbound_call_20_cnt_min_rat_l1m_l3m\n",
    ",Inbound_call_cnt_min_rat_l1m_l3m\n",
    ",Manuel_call_20_cnt_min_rat_l1m_l3m\n",
    ",Manuel_call_cnt_min_rat_l1m_l3m\n",
    ",tah_diger_min_rat_l1m_l3m\n",
    ",ActivatedProtocolCount_min_rat_l1m_l3m\n",
    ",BrokeProtocolCount_min_rat_l1m_l3m\n",
    ",Manuel_call_20_cnt_min_rat_l1m_l6m\n",
    ",Manuel_call_cnt_min_rat_l1m_l6m\n",
    ",ActivatedProtocolCount_min_rat_l1m_l6m\n",
    ",BrokeProtocolCount_min_rat_l1m_l6m\n",
    ",FullyPaidProtocolCount_min_rat_l1m_l6m\n",
    ",outbound_call_20_cnt_min_rat_l3m_l6m\n",
    ",Inbound_call_20_cnt_min_rat_l3m_l6m\n",
    ",Inbound_call_cnt_min_rat_l3m_l6m\n",
    ",Manuel_call_20_cnt_min_rat_l3m_l6m\n",
    ",Manuel_call_cnt_min_rat_l3m_l6m\n",
    ",ActivatedProtocolCount_min_rat_l3m_l6m\n",
    ",BrokeProtocolCount_min_rat_l3m_l6m\n",
    ",outbound_call_20_cnt_min_rat_l1m_l12m\n",
    ",Manuel_call_20_cnt_min_rat_l1m_l12m\n",
    ",Manuel_call_cnt_min_rat_l1m_l12m\n",
    ",tah_icra_odemesi_Debt_rat_min_rat_l1m_l12m\n",
    ",CanceledProtocolCount_min_rat_l1m_l12m\n",
    ",FullyPaidProtocolCount_min_rat_l1m_l12m\n",
    ",LiveProtocolCount_min_rat_l1m_l12m\n",
    ",TotalProtocolAmount_min_rat_l1m_l12m\n",
    ",Inbound_call_20_cnt_min_rat_l3m_l12m\n",
    ",Inbound_call_cnt_min_rat_l3m_l12m\n",
    ",Manuel_call_20_cnt_min_rat_l3m_l12m\n",
    ",Manuel_call_cnt_min_rat_l3m_l12m\n",
    ",tah_icra_satis_min_rat_l3m_l12m\n",
    ",tah_odeme_teklifi_min_rat_l3m_l12m\n",
    ",CanceledProtocolCount_min_rat_l3m_l12m\n",
    ",sum_IntAmount_min_rat_l6m_l12m\n",
    ",IntAmount_UPB_rat_min_rat_l6m_l12m\n",
    ",outbound_call_cnt_min_rat_l6m_l12m\n",
    ",CanceledProtocolCount_min_rat_l6m_l12m\n",
    ",LiveProtocolCount_min_rat_l6m_l12m\n",
    ",Current_UPB_Debt_rat_max_rat_l1m_l3m\n",
    ",Inbound_call_20_cnt_max_rat_l1m_l3m\n",
    ",Manuel_call_20_cnt_max_rat_l1m_l3m\n",
    ",Manuel_call_cnt_max_rat_l1m_l3m\n",
    ",tah_odeme_teklifi_UPB_rat_max_rat_l1m_l3m\n",
    ",LiveProtocolAmount_max_rat_l1m_l3m\n",
    ",Current_UPB_Debt_rat_max_rat_l1m_l6m\n",
    ",BrokeProtocolCount_max_rat_l1m_l6m\n",
    ",outbound_call_20_cnt_max_rat_l3m_l6m\n",
    ",Inbound_call_20_cnt_max_rat_l3m_l6m\n",
    ",Inbound_call_cnt_max_rat_l3m_l6m\n",
    ",Manuel_call_20_cnt_max_rat_l3m_l6m\n",
    ",Manuel_call_cnt_max_rat_l3m_l6m\n",
    ",tah_total_max_rat_l3m_l6m\n",
    ",Current_UPB_Debt_rat_max_rat_l1m_l12m\n",
    ",tah_diger_max_rat_l1m_l12m\n",
    ",ActivatedProtocolCount_max_rat_l1m_l12m\n",
    ",CanceledProtocolCount_max_rat_l1m_l12m\n",
    ",FullyPaidProtocolCount_max_rat_l1m_l12m\n",
    ",LiveProtocolCount_max_rat_l1m_l12m\n",
    ",NewProtocolCount_max_rat_l1m_l12m\n",
    ",BrokeProtocolCount_max_rat_l3m_l12m\n",
    ",FullyPaidProtocolCount_max_rat_l3m_l12m\n",
    ",Current_UPB_Debt_rat_max_rat_l6m_l12m\n",
    ",ExpenseAmount_Debt_rat_max_rat_l6m_l12m\n",
    ",Inbound_call_20_cnt_max_rat_l6m_l12m\n",
    ",Inbound_call_cnt_max_rat_l6m_l12m\n",
    ",Manuel_call_20_cnt_max_rat_l6m_l12m\n",
    ",Manuel_call_cnt_max_rat_l6m_l12m\n",
    ",tah_diger_max_rat_l6m_l12m\n",
    ",ActivatedProtocolCount_max_rat_l6m_l12m\n",
    ",BrokeProtocolCount_max_rat_l6m_l12m\n",
    ",CanceledProtocolCount_max_rat_l6m_l12m\n",
    ",FullyPaidProtocolCount_max_rat_l6m_l12m\n",
    ",LiveProtocolCount_max_rat_l6m_l12m\n",
    ",NewProtocolCount_max_rat_l6m_l12m\n",
    "\n",
    "\n",
    " FROM [Erc_Project].[exp].[ticari_vars_target] a\n",
    "    left join [Erc_Project].[exp].[ticari_ratio_vars_avg] b on a.contactcode=b.contactcode and a.refdate=b.refdate \n",
    "    left join [Erc_Project].[exp].[ticari_ratio_vars_min] c on a.contactcode=c.contactcode and a.refdate=c.refdate\n",
    "    left join [Erc_Project].[exp].[ticari_ratio_vars_max] d on a.contactcode=d.contactcode and a.refdate=d.refdate\n",
    "\"\"\"), conn)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#model_vars=pd.read_excel('ticari_model_variables_'+str(datetime.today().strftime(\"%Y_%m_%d\"))+'.xlsx')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#model_vars.loc[len(model_vars.index)]=['121','refdate']\n",
    "#model_vars.loc[len(model_vars.index)]=['122','target']\n",
    "#model_vars=model_vars.drop(model_vars.index[0])\n",
    "#model_vars"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#data=data[model_vars[0]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.fillna(-999,inplace=True)\n",
    "data=data.replace([np.nan],-999)\n",
    "data=data.replace([np.inf,-np.inf],-999)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Dataset has 56647 entries and 236 features\n",
      "0 columns eliminated due to missing values\n",
      "0 columns eliminated due to single unique values, namely: []\n",
      "\n",
      "Preprocess is done...\n",
      "\n",
      "Shape of dataframes; train: (35629,), test: (8908,), oot: (12110,)\n",
      "\n",
      "Percentage of dataframes; train: 62.90%, test: 15.73%, oot: 21.38%\n",
      "\n",
      "Shape of dataframes; train: (35629,), test: (8908,), oot: (12110,)\n",
      "\n",
      "Percentage of dataframes; train: 62.90%, test: 15.73%, oot: 21.38%\n",
      "\n",
      "iteration: 1\n",
      "[0]\ttrain-mae:0.49527\teval-mae:0.49529\n",
      "[50]\ttrain-mae:0.30999\teval-mae:0.31112\n",
      "[100]\ttrain-mae:0.19756\teval-mae:0.19958\n",
      "[150]\ttrain-mae:0.12927\teval-mae:0.13202\n",
      "[200]\ttrain-mae:0.08768\teval-mae:0.09111\n",
      "[250]\ttrain-mae:0.06243\teval-mae:0.06629\n",
      "[300]\ttrain-mae:0.04702\teval-mae:0.05133\n",
      "[350]\ttrain-mae:0.03768\teval-mae:0.04227\n",
      "[400]\ttrain-mae:0.03195\teval-mae:0.03684\n",
      "[450]\ttrain-mae:0.02844\teval-mae:0.03359\n",
      "[499]\ttrain-mae:0.02629\teval-mae:0.03166\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 2\n",
      "[0]\ttrain-rmse:0.49527\teval-rmse:0.49529\n",
      "[50]\ttrain-rmse:0.31293\teval-rmse:0.31473\n",
      "[100]\ttrain-rmse:0.20858\teval-rmse:0.21321\n",
      "[150]\ttrain-rmse:0.15241\teval-rmse:0.16073\n",
      "[200]\ttrain-rmse:0.12449\teval-rmse:0.13662\n",
      "[250]\ttrain-rmse:0.11186\teval-rmse:0.12645\n",
      "[300]\ttrain-rmse:0.10604\teval-rmse:0.12280\n",
      "[350]\ttrain-rmse:0.10345\teval-rmse:0.12144\n",
      "[400]\ttrain-rmse:0.10203\teval-rmse:0.12099\n",
      "[450]\ttrain-rmse:0.10116\teval-rmse:0.12088\n",
      "[499]\ttrain-rmse:0.10049\teval-rmse:0.12086\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 3\n",
      "[0]\ttrain-mae:0.49527\teval-mae:0.49529\n",
      "[50]\ttrain-mae:0.31026\teval-mae:0.31121\n",
      "[100]\ttrain-mae:0.19798\teval-mae:0.19974\n",
      "[150]\ttrain-mae:0.12980\teval-mae:0.13219\n",
      "[200]\ttrain-mae:0.08826\teval-mae:0.09127\n",
      "[250]\ttrain-mae:0.06300\teval-mae:0.06643\n",
      "[300]\ttrain-mae:0.04750\teval-mae:0.05143\n",
      "[350]\ttrain-mae:0.03805\teval-mae:0.04237\n",
      "[400]\ttrain-mae:0.03229\teval-mae:0.03688\n",
      "[450]\ttrain-mae:0.02869\teval-mae:0.03356\n",
      "[499]\ttrain-mae:0.02640\teval-mae:0.03158\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 4\n",
      "[0]\ttrain-rmse:0.49528\teval-rmse:0.49529\n",
      "[50]\ttrain-rmse:0.31331\teval-rmse:0.31481\n",
      "[100]\ttrain-rmse:0.20931\teval-rmse:0.21332\n",
      "[150]\ttrain-rmse:0.15359\teval-rmse:0.16075\n",
      "[200]\ttrain-rmse:0.12603\teval-rmse:0.13659\n",
      "[250]\ttrain-rmse:0.11344\teval-rmse:0.12643\n",
      "[300]\ttrain-rmse:0.10744\teval-rmse:0.12272\n",
      "[350]\ttrain-rmse:0.10451\teval-rmse:0.12135\n",
      "[400]\ttrain-rmse:0.10303\teval-rmse:0.12079\n",
      "[450]\ttrain-rmse:0.10185\teval-rmse:0.12057\n",
      "[499]\ttrain-rmse:0.10080\teval-rmse:0.12049\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 5\n",
      "[0]\ttrain-mae:0.49528\teval-mae:0.49529\n",
      "[50]\ttrain-mae:0.31045\teval-mae:0.31126\n",
      "[100]\ttrain-mae:0.19833\teval-mae:0.19984\n",
      "[150]\ttrain-mae:0.13022\teval-mae:0.13236\n",
      "[200]\ttrain-mae:0.08877\teval-mae:0.09145\n",
      "[250]\ttrain-mae:0.06349\teval-mae:0.06663\n",
      "[300]\ttrain-mae:0.04787\teval-mae:0.05161\n",
      "[350]\ttrain-mae:0.03832\teval-mae:0.04251\n",
      "[400]\ttrain-mae:0.03250\teval-mae:0.03701\n",
      "[450]\ttrain-mae:0.02888\teval-mae:0.03370\n",
      "[499]\ttrain-mae:0.02660\teval-mae:0.03171\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 6\n",
      "[0]\ttrain-rmse:0.49528\teval-rmse:0.49529\n",
      "[50]\ttrain-rmse:0.31355\teval-rmse:0.31484\n",
      "[100]\ttrain-rmse:0.20994\teval-rmse:0.21339\n",
      "[150]\ttrain-rmse:0.15457\teval-rmse:0.16089\n",
      "[200]\ttrain-rmse:0.12736\teval-rmse:0.13669\n",
      "[250]\ttrain-rmse:0.11485\teval-rmse:0.12661\n",
      "[300]\ttrain-rmse:0.10855\teval-rmse:0.12288\n",
      "[350]\ttrain-rmse:0.10533\teval-rmse:0.12155\n",
      "[400]\ttrain-rmse:0.10362\teval-rmse:0.12101\n",
      "[450]\ttrain-rmse:0.10247\teval-rmse:0.12083\n",
      "[499]\ttrain-rmse:0.10142\teval-rmse:0.12078\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 7\n",
      "[0]\ttrain-mae:0.49530\teval-mae:0.49532\n",
      "[50]\ttrain-mae:0.31264\teval-mae:0.31262\n",
      "[100]\ttrain-mae:0.20277\teval-mae:0.20271\n",
      "[150]\ttrain-mae:0.13628\teval-mae:0.13621\n",
      "[200]\ttrain-mae:0.09607\teval-mae:0.09598\n",
      "[250]\ttrain-mae:0.07173\teval-mae:0.07163\n",
      "[300]\ttrain-mae:0.05701\teval-mae:0.05691\n",
      "[350]\ttrain-mae:0.04810\teval-mae:0.04800\n",
      "[400]\ttrain-mae:0.04271\teval-mae:0.04261\n",
      "[450]\ttrain-mae:0.03945\teval-mae:0.03934\n",
      "[499]\ttrain-mae:0.03751\teval-mae:0.03740\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 8\n",
      "[0]\ttrain-rmse:0.49532\teval-rmse:0.49532\n",
      "[50]\ttrain-rmse:0.31644\teval-rmse:0.31643\n",
      "[100]\ttrain-rmse:0.21754\teval-rmse:0.21747\n",
      "[150]\ttrain-rmse:0.16730\teval-rmse:0.16719\n",
      "[200]\ttrain-rmse:0.14461\teval-rmse:0.14446\n",
      "[250]\ttrain-rmse:0.13536\teval-rmse:0.13519\n",
      "[300]\ttrain-rmse:0.13181\teval-rmse:0.13162\n",
      "[350]\ttrain-rmse:0.13048\teval-rmse:0.13029\n",
      "[400]\ttrain-rmse:0.13000\teval-rmse:0.12980\n",
      "[450]\ttrain-rmse:0.12982\teval-rmse:0.12962\n",
      "[499]\ttrain-rmse:0.12975\teval-rmse:0.12955\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 9\n",
      "[0]\ttrain-mae:0.49530\teval-mae:0.49532\n",
      "[50]\ttrain-mae:0.31264\teval-mae:0.31262\n",
      "[100]\ttrain-mae:0.20277\teval-mae:0.20272\n",
      "[150]\ttrain-mae:0.13629\teval-mae:0.13622\n",
      "[200]\ttrain-mae:0.09607\teval-mae:0.09599\n",
      "[250]\ttrain-mae:0.07175\teval-mae:0.07165\n",
      "[300]\ttrain-mae:0.05702\teval-mae:0.05692\n",
      "[350]\ttrain-mae:0.04811\teval-mae:0.04801\n",
      "[400]\ttrain-mae:0.04272\teval-mae:0.04262\n",
      "[450]\ttrain-mae:0.03947\teval-mae:0.03936\n",
      "[499]\ttrain-mae:0.03752\teval-mae:0.03742\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 10\n",
      "[0]\ttrain-rmse:0.49532\teval-rmse:0.49532\n",
      "[50]\ttrain-rmse:0.31645\teval-rmse:0.31643\n",
      "[100]\ttrain-rmse:0.21754\teval-rmse:0.21749\n",
      "[150]\ttrain-rmse:0.16731\teval-rmse:0.16720\n",
      "[200]\ttrain-rmse:0.14462\teval-rmse:0.14447\n",
      "[250]\ttrain-rmse:0.13537\teval-rmse:0.13520\n",
      "[300]\ttrain-rmse:0.13182\teval-rmse:0.13164\n",
      "[350]\ttrain-rmse:0.13049\teval-rmse:0.13030\n",
      "[400]\ttrain-rmse:0.13001\teval-rmse:0.12981\n",
      "[450]\ttrain-rmse:0.12983\teval-rmse:0.12963\n",
      "[499]\ttrain-rmse:0.12976\teval-rmse:0.12956\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 11\n",
      "[0]\ttrain-mae:0.49530\teval-mae:0.49532\n",
      "[50]\ttrain-mae:0.31266\teval-mae:0.31263\n",
      "[100]\ttrain-mae:0.20279\teval-mae:0.20272\n",
      "[150]\ttrain-mae:0.13631\teval-mae:0.13623\n",
      "[200]\ttrain-mae:0.09609\teval-mae:0.09600\n",
      "[250]\ttrain-mae:0.07176\teval-mae:0.07166\n",
      "[300]\ttrain-mae:0.05703\teval-mae:0.05693\n",
      "[350]\ttrain-mae:0.04813\teval-mae:0.04802\n",
      "[400]\ttrain-mae:0.04274\teval-mae:0.04263\n",
      "[450]\ttrain-mae:0.03948\teval-mae:0.03937\n",
      "[499]\ttrain-mae:0.03754\teval-mae:0.03743\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 12\n",
      "[0]\ttrain-rmse:0.49532\teval-rmse:0.49532\n",
      "[50]\ttrain-rmse:0.31646\teval-rmse:0.31644\n",
      "[100]\ttrain-rmse:0.21756\teval-rmse:0.21749\n",
      "[150]\ttrain-rmse:0.16732\teval-rmse:0.16721\n",
      "[200]\ttrain-rmse:0.14463\teval-rmse:0.14449\n",
      "[250]\ttrain-rmse:0.13538\teval-rmse:0.13521\n",
      "[300]\ttrain-rmse:0.13183\teval-rmse:0.13165\n",
      "[350]\ttrain-rmse:0.13050\teval-rmse:0.13031\n",
      "[400]\ttrain-rmse:0.13001\teval-rmse:0.12982\n",
      "[450]\ttrain-rmse:0.12984\teval-rmse:0.12964\n",
      "[499]\ttrain-rmse:0.12977\teval-rmse:0.12957\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 13\n",
      "[0]\ttrain-mae:0.49530\teval-mae:0.49532\n",
      "[50]\ttrain-mae:0.31335\teval-mae:0.31330\n",
      "[100]\ttrain-mae:0.20347\teval-mae:0.20339\n",
      "[150]\ttrain-mae:0.13699\teval-mae:0.13689\n",
      "[200]\ttrain-mae:0.09677\teval-mae:0.09666\n",
      "[250]\ttrain-mae:0.07244\teval-mae:0.07232\n",
      "[300]\ttrain-mae:0.05771\teval-mae:0.05759\n",
      "[350]\ttrain-mae:0.04881\teval-mae:0.04868\n",
      "[400]\ttrain-mae:0.04342\teval-mae:0.04329\n",
      "[450]\ttrain-mae:0.04016\teval-mae:0.04003\n",
      "[499]\ttrain-mae:0.03821\teval-mae:0.03809\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:105: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "iteration: 14\n",
      "[0]\ttrain-rmse:0.49532\teval-rmse:0.49532\n",
      "[50]\ttrain-rmse:0.31735\teval-rmse:0.31730\n",
      "[100]\ttrain-rmse:0.21886\teval-rmse:0.21871\n"
     ]
    }
   ],
   "source": [
    "\n",
    "if __name__ == \"__main__\":\n",
    "    \"\"\"main parameters should be set before each run\"\"\"\n",
    "\n",
    "    grid_search_= True\n",
    "    save_results_= True\n",
    "    random_search = False\n",
    "    \n",
    "    \n",
    "    segment = 'ticari' #kkbsiz,kkbli_yeni,kkbli_mevcut_sm, kkbli_mevcut_ou, kkbli_mevcut_emk,kkbli_mevcut_ku\n",
    "    \n",
    "\n",
    "    # key variables\n",
    "    key_var = [\"contactcode\", \"refdate\", \"target\"] # for management of data issues\n",
    "    key_vars = [\"contactcode\", \"refdate\", \"target\"] # for preparing data.table() for model estimation\n",
    "    \n",
    "    # define maximum acceptable percentage of missing information\n",
    "    max_threshold = 1\n",
    "    \n",
    "    # splitting percentage of development data into train nand test\n",
    "    splitting_perc = 0.2\n",
    "   \n",
    "    min_date = '2021-01-01' #oot date\n",
    "\n",
    "    # specify seed for random selection - to keep track of the chosen seed, for results' reproduction\n",
    "    seed = 1234\n",
    "    \n",
    "    ## diff_threshold for grid search\n",
    "    diff_threshold = 1000\n",
    "\n",
    "\n",
    "    trial = 'v1'\n",
    "    t_string = str(trial)\n",
    "    today = date.today()\n",
    "    path = 'C://Users//p-cemre.kassara//Desktop/DV f2 py template/{}-{}-{}/'\n",
    "    path = path.format(segment, t_string, today)\n",
    "    \n",
    "        ##datayi okutma##\n",
    "    #directory = Path(\"D:/gtp-files/Final/\")\n",
    "\n",
    "    \n",
    "    #data_directory = directory / \"Data_Files\"\n",
    "    \n",
    "    #data okuma:\n",
    "    \n",
    "    #data=pd.read_csv('Ticari.csv')\n",
    "\n",
    "    \n",
    "    df= data.copy()\n",
    "            \n",
    "    \n",
    "    df=df.reset_index(drop=True)\n",
    "\n",
    "\n",
    "\n",
    "    ###############\n",
    "    ###Modelling###\n",
    "    ###############\n",
    "    \n",
    "    \n",
    "    #code execution:\n",
    "    df = execute_preprocess(df, max_threshold)\n",
    "    df = df.reset_index(drop = True)\n",
    "    train_index, test_index, oot_index=set_split_index(df, splitting_perc, seed, min_date)\n",
    "   \n",
    "    if grid_search_:\n",
    "        grid_params = {\n",
    "        # Parameters that we are going to tune.\n",
    "        'eta' : [0.01,0.05, 0.1],  #0.01,0.05, #0.1\n",
    "        'max_depth': [9,12,15,18], #[9,15],#[9,12,15], #np.arange(3, 8, 2).tolist(), #3,5  #6\n",
    "        'min_child_weight': [20,50,100,150,200], #[50,100], #250,500,750],\n",
    "        'subsample': [1],\n",
    "        'colsample_bytree': [1],\n",
    "        'gamma' : [0,40,60,80,100], #40,60,80,100\n",
    "        'reg_alpha' : [0.1,0.5,0.9], #[0.9,0.5,0.1], #0.05,0.1\n",
    "        # Other parameters  \n",
    "        'objective': ['reg:squarederror'],\n",
    "        #eval_metric değiştirilecek:\n",
    "        'eval_metric' : ['mae','rmse'], #mae #mse # Use mean abs error rather than rmse (lower impact of outliers)\n",
    "        'booster' : ['gbtree'] #dart \n",
    "        }\n",
    "        xgb_gs_train, best_gs_iter, params = execute_grid_search(df, key_var, grid_params, splitting_perc, min_date, seed, random_search, diff_threshold, 432)    \n",
    "    else:\n",
    "        params = set_params(seed)\n",
    " \n",
    "    bst, predictions, perf_sum, importance_xgb = execute_final_model(df, key_var, params, splitting_perc, seed, num_boost_round=500, early_stopping_rounds=20, verbose_eval=10)\n",
    "    train_sum_by_pred_score, train_sum_by_pred_row, fig_train_score, fig_train_row, test_sum_by_pred_score, test_sum_by_pred_row, fig_test_score, fig_test_row, oot_sum_by_pred_score, oot_sum_by_pred_row, fig_oot_score, fig_oot_row = performance_measurement()    \n",
    "    #segment_summary_all = evaluate_performance_by_segment()    \n",
    "    #segment_summary_all_train, segment_summary_all_test,segment_summary_all_oot, segment_summary_all = performance_measurement_for_segment()\n",
    "    \n",
    "    print(perf_sum[[\"sample\",\"mae\"]])\n",
    "    if save_results_:\n",
    "        save_results(path)\n",
    "    print('Done with model_building.py')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_out=predictions[predictions['sample']=='train']\n",
    "train_out[\"pred_binary\"]=np.where(train_out['predicted']>0.03,1,0)\n",
    "train_out.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_out=predictions[predictions['sample']=='test']\n",
    "test_out[\"pred_binary\"]=np.where(test_out['predicted']>0.03,1,0)\n",
    "test_out.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "oot_out=predictions[predictions['sample']=='test']\n",
    "oot_out[\"pred_binary\"]=np.where(oot_out['predicted']>0.03,1,0)\n",
    "oot_out.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import roc_auc_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "train_conf_matrix=confusion_matrix(train_out[\"actual\"],train_out[\"pred_binary\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_roc_auc=roc_auc_score(train_out[\"actual\"],train_out[\"pred_binary\"])\n",
    "test_roc_auc=roc_auc_score(test_out[\"actual\"],test_out[\"pred_binary\"])\n",
    "oot_roc_auc=roc_auc_score(oot_out[\"actual\"],oot_out[\"pred_binary\"])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_gini=2*train_roc_auc-1\n",
    "test_gini=2*test_roc_auc-1\n",
    "oot_gini=2*oot_roc_auc-1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"gini train: \", train_gini)\n",
    "print(\"gini test: \", test_gini)\n",
    "print(\"gini oot: \", oot_gini)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# aşağısı silinecek"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#powercurve için:\n",
    "#pmml"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, X_oot, y_train, y_test, y_oot, train_keys, test_keys, oot_keys = split_df_by_index_pmml(df, key_var, train_index, test_index, oot_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_X = X_train.copy()\n",
    "df_y = y_train.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "regressor = XGBRegressor(n_estimators = (bst.best_iteration +1),\n",
    "                              #early_stopping_rounds=early_stopping_rounds,\n",
    "                              learning_rate = params['eta'],\n",
    "                              max_depth = params['max_depth'],\n",
    "                              min_child_weight = params['min_child_weight'],\n",
    "                              subsample = params['subsample'],\n",
    "                              colsample_bytree = params['colsample_bytree'],\n",
    "                              gamma = params['gamma'],\n",
    "                              reg_alpha = params['reg_alpha'],\n",
    "                              objective = params['objective'],\n",
    "                              booster = params['booster'],\n",
    "                              eval_metric = params['eval_metric'],\n",
    "                              seed = params['seed'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline = PMMLPipeline([\n",
    "      (\"regressor\", regressor)\n",
    "    ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline.fit(df_X, df_y.ravel())\n",
    "pipeline.verify(df_X,zeroThreshold = 1e-6, precision = 1e-6)\n",
    "pipeline = make_pmml_pipeline(pipeline, active_fields = df_X.columns.values, target_fields = df_y.name )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predict_pmml_train = pipeline.predict(X_train)\n",
    "predict_pmml_test = pipeline.predict(X_test)\n",
    "predict_pmml_oot = pipeline.predict(X_oot)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predict_pmml_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#predictions[predictions['sample']=='train']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pmml_train_df = pd.DataFrame({'sample': 'train', \n",
    "                              'actual': y_train, \n",
    "                              'predicted_pmml' : predict_pmml_train,\n",
    "                              'RT_CUSTOMER_ID': train_keys[\"RT_CUSTOMER_ID\"], \n",
    "                              #'REPORT_DATE': train_keys[\"REPORT_DATE\"],\n",
    "                              'APP_ID': train_keys[\"APP_ID\"]\n",
    "                              })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pmml_test_df = pd.DataFrame({'sample': 'test', \n",
    "                             'actual': y_test, \n",
    "                             'predicted_pmml' : predict_pmml_test,\n",
    "                             'RT_CUSTOMER_ID': test_keys[\"RT_CUSTOMER_ID\"], \n",
    "                             #'REPORT_DATE': test_keys[\"REPORT_DATE\"],\n",
    "                             'APP_ID': test_keys[\"APP_ID\"]\n",
    "                              })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pmml_oot_df = pd.DataFrame({'sample': 'oot', \n",
    "                            'actual': y_oot, \n",
    "                            'predicted_pmml' : predict_pmml_oot,\n",
    "                            'RT_CUSTOMER_ID': oot_keys[\"RT_CUSTOMER_ID\"], \n",
    "                             #'REPORT_DATE': oot_keys[\"REPORT_DATE\"],\n",
    "                             'APP_ID': oot_keys[\"APP_ID\"]\n",
    "                              })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_pmml = pmml_train_df.append([pmml_test_df, pmml_oot_df])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_all =pd.merge(predictions , predictions_pmml, left_on=['APP_ID'], right_on=['APP_ID'], how='left' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_all"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "if save_results_:\n",
    "    save_results(path)\n",
    "    print('Done with model_building.py')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "####################\n",
    "###Analizler Özet###\n",
    "####################\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_all = pd.read_csv(r'D:\\gtp-files\\Final\\3.Hedef Değişken Belirleme\\FAZ1\\KKBLI_YENI_v1\\KKBli_Yeni_target_final.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_v1=pd.merge(df,data_all[['APP_ID','KKB_FLAG','ANA_GRUP_FLAG','APP_SH_DOC_INCOME','RT_MODEL_EST_INCOME',\n",
    "                                 'APP_SH_DECL_INCOME','APP_SALARY_AMT_LM','SKY_FLAG_URUN',\n",
    "                                 'AGE_GRUP','YENISALARYCUSTFLAG','income_seg','APP_FINAL_DECISION',\n",
    "                                 'RT_WORKING_TYPE_DESC', 'RT_SECTOR_TYPE_DESC', 'RT_OCCUPATION', \n",
    "                                 'DOC_INCOME_FINAL','target_w_inf','inflation']],\n",
    "               on=['APP_ID'],how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_v1['DOC_INCOME_FLAG'] = np.where(df_v1['APP_SH_DOC_INCOME'] > 0, 1,0)\n",
    "df_v1['DECL_INCOME_FLAG'] = np.where(df_v1['APP_SH_DECL_INCOME'] > 0, 1,0)\n",
    "df_v1['APP_SALARY_AMT_LM_FLAG'] = np.where(df_v1['APP_SALARY_AMT_LM'] > 0, 1,0)\n",
    "df_v1['RT_KKB_CC_O_TOT_LIM_FLAG'] = np.where(df_v1['RT_KKB_CC_O_TOT_LIM'] > 0, 1,0)\n",
    "df_v1['RT_KKB_CC_MAX_O_C_LIM_L1Y_FLAG'] = np.where(df_v1['RT_KKB_CC_MAX_O_C_LIM_L1Y'] > 0, 1,0)\n",
    "df_v1['RT_KKB_INST_O_TOT_INST_AMT_FLAG'] = np.where(df_v1['RT_KKB_INST_O_TOT_INST_AMT'] > 0, 1,0)\n",
    "df_v1['RT_MODEL_EST_INCOME_FLAG']=np.where(df_v1['RT_MODEL_EST_INCOME']>0,1,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cx_Oracle as cxo \n",
    "import sqlalchemy\n",
    "import timeit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v=pd.merge(predictions,df_v1,on=['APP_ID','APP_DATE','RT_CUSTOMER_ID'],how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "oracle_db=sqlalchemy.create_engine(\"oracle+cx_oracle://SASGELIRTAHMIN:DDzVB4597_PHZK239+@dbtexa.vakifbank.intra:1854/?service_name=DWHTEST\")\n",
    "connection=oracle_db.connect() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sql=\"SELECT APP_ID,S314 FROM SPSSDM.DM_BRY_SEGMENT_KKBLI_YENI_S314\" \n",
    "df_chunk=pd.read_sql(sql,connection,chunksize=1000000) \n",
    "chunklist= [] \n",
    "for chunk in tqdm(df_chunk,total=20):\n",
    "    chunklist.append(chunk) \n",
    "gc.collect()\n",
    "\n",
    "start_time=timeit.default_timer() \n",
    "df_sm=pd.concat(chunklist,ignore_index=True)\n",
    "print(timeit.default_timer()-start_time) \n",
    "\n",
    "del chunklist, df_chunk, chunk \n",
    "gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_sm.columns = map(str.upper,df_sm.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_sm['S314_flag']=np.where(df_sm['S314']>0,1,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1=pd.merge(predictions_v,df_sm[['APP_ID','S314','S314_flag']],on=['APP_ID'],how='left') #,'APP_DATE','RT_CUSTOMER_ID'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['S314'] = predictions_v1['S314']*predictions_v1['inflation']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['actual_grp'] = np.where(predictions_v1['actual'] ==0, '0',\n",
    "                                     np.where(predictions_v1['actual'] < 2000, '<2000',\n",
    "                                              np.where(predictions_v1['actual'] < 3000, '<3000',\n",
    "                                                       np.where(predictions_v1['actual'] < 5000, '<5000',\n",
    "                                                                np.where(predictions_v1['actual'] < 7000, '<7000',\n",
    "                                                                         np.where(predictions_v1['actual'] < 10000, '<10000',\n",
    "                                                                                  np.where(predictions_v1['actual'] < 15000, '<15000',\n",
    "                                                                                           np.where(predictions_v1['actual'] < 20000, '<20000',\n",
    "                                                                                                    np.where(predictions_v1['actual'] < 50000, '<50000',\n",
    "                                                                                                             np.where(predictions_v1['actual'] >= 50000, '>=50000','Unknown'))))))))))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['pred_grp'] = np.where(predictions_v1['predicted'] ==0, '0',\n",
    "                                     np.where(predictions_v1['predicted'] < 2000, '<2000',\n",
    "                                              np.where(predictions_v1['predicted'] < 3000, '<3000',\n",
    "                                                       np.where(predictions_v1['predicted'] < 5000, '<5000',\n",
    "                                                                np.where(predictions_v1['predicted'] < 7000, '<7000',\n",
    "                                                                         np.where(predictions_v1['predicted'] < 10000, '<10000',\n",
    "                                                                                  np.where(predictions_v1['predicted'] < 15000, '<15000',\n",
    "                                                                                           np.where(predictions_v1['predicted'] < 20000, '<20000',\n",
    "                                                                                                    np.where(predictions_v1['predicted'] < 50000, '<50000',\n",
    "                                                                                                             np.where(predictions_v1['predicted'] >= 50000, '>=50000','Unknown'))))))))))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['model_grp'] = np.where(predictions_v1['RT_MODEL_EST_INCOME'] ==0, '0',\n",
    "                                     np.where(predictions_v1['RT_MODEL_EST_INCOME'] < 2000, '<2000',\n",
    "                                              np.where(predictions_v1['RT_MODEL_EST_INCOME'] < 3000, '<3000',\n",
    "                                                       np.where(predictions_v1['RT_MODEL_EST_INCOME'] < 5000, '<5000',\n",
    "                                                                np.where(predictions_v1['RT_MODEL_EST_INCOME'] < 7000, '<7000',\n",
    "                                                                         np.where(predictions_v1['RT_MODEL_EST_INCOME'] < 10000, '<10000',\n",
    "                                                                                  np.where(predictions_v1['RT_MODEL_EST_INCOME'] < 15000, '<15000',\n",
    "                                                                                           np.where(predictions_v1['RT_MODEL_EST_INCOME'] < 20000, '<20000',\n",
    "                                                                                                    np.where(predictions_v1['RT_MODEL_EST_INCOME'] < 50000, '<50000',\n",
    "                                                                                                             np.where(predictions_v1['RT_MODEL_EST_INCOME'] >= 50000, '>=50000','Unknown'))))))))))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['S314_grp'] = np.where(predictions_v1['S314'] ==0, '0',\n",
    "                                     np.where(predictions_v1['S314'] < 2000, '<2000',\n",
    "                                              np.where(predictions_v1['S314'] < 3000, '<3000',\n",
    "                                                       np.where(predictions_v1['S314'] < 5000, '<5000',\n",
    "                                                                np.where(predictions_v1['S314'] < 7000, '<7000',\n",
    "                                                                         np.where(predictions_v1['S314'] < 10000, '<10000',\n",
    "                                                                                  np.where(predictions_v1['S314'] < 15000, '<15000',\n",
    "                                                                                           np.where(predictions_v1['S314'] < 20000, '<20000',\n",
    "                                                                                                    np.where(predictions_v1['S314'] < 50000, '<50000',\n",
    "                                                                                                             np.where(predictions_v1['S314'] >= 50000, '>=50000','Unknown'))))))))))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['act_pred']=predictions_v1['actual']-predictions_v1['predicted']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['act_pred_grp'] = np.where(predictions_v1['act_pred'] <= -20000, '<=-20000',\n",
    "                             np.where(predictions_v1['act_pred'] < -15000, '<-15000',\n",
    "                             np.where(predictions_v1['act_pred'] < -10000, '<-10000',\n",
    "                             np.where(predictions_v1['act_pred'] < -7000, '<-7000',\n",
    "                             np.where(predictions_v1['act_pred'] < -5000, '<-5000',\n",
    "                             np.where(predictions_v1['act_pred'] < -3000, '<-3000',\n",
    "                             np.where(predictions_v1['act_pred'] < -2000, '<-2000',\n",
    "                             np.where(predictions_v1['act_pred'] < -1000, '<-1000',\n",
    "                             np.where(predictions_v1['act_pred'] < -500, '<-500',\n",
    "                             np.where(predictions_v1['act_pred'] < 0, '<0',\n",
    "                             np.where(predictions_v1['act_pred'] ==0, '0',\n",
    "                             np.where(predictions_v1['act_pred'] < 500, '<500', \n",
    "                             np.where(predictions_v1['act_pred'] < 1000, '<1000',         \n",
    "                             np.where(predictions_v1['act_pred'] < 2000, '<2000',\n",
    "                             np.where(predictions_v1['act_pred'] < 3000, '<3000',\n",
    "                             np.where(predictions_v1['act_pred'] < 5000, '<5000',\n",
    "                             np.where(predictions_v1['act_pred'] < 7000, '<7000',\n",
    "                             np.where(predictions_v1['act_pred'] < 10000, '<10000',\n",
    "                             np.where(predictions_v1['act_pred'] < 15000, '<15000',\n",
    "                             np.where(predictions_v1['act_pred'] < 20000, '<20000',\n",
    "                             np.where(predictions_v1['act_pred'] >= 20000, '>=20000',\n",
    "                                      'Unknown')))))))))))))))))))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['model_pred']=predictions_v1['RT_MODEL_EST_INCOME']-predictions_v1['predicted']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['model_pred_grp'] = np.where(predictions_v1['model_pred'] <= -20000, '<=-20000',\n",
    "                             np.where(predictions_v1['model_pred'] < -15000, '<-15000',\n",
    "                             np.where(predictions_v1['model_pred'] < -10000, '<-10000',\n",
    "                             np.where(predictions_v1['model_pred'] < -7000, '<-7000',\n",
    "                             np.where(predictions_v1['model_pred'] < -5000, '<-5000',\n",
    "                             np.where(predictions_v1['model_pred'] < -3000, '<-3000',\n",
    "                             np.where(predictions_v1['model_pred'] < -2000, '<-2000',\n",
    "                             np.where(predictions_v1['model_pred'] < -1000, '<-1000',\n",
    "                             np.where(predictions_v1['model_pred'] < -500, '<-500',\n",
    "                             np.where(predictions_v1['model_pred'] < 0, '<0',\n",
    "                             np.where(predictions_v1['model_pred'] ==0, '0',\n",
    "                             np.where(predictions_v1['model_pred'] < 500, '<500', \n",
    "                             np.where(predictions_v1['model_pred'] < 1000, '<1000',         \n",
    "                             np.where(predictions_v1['model_pred'] < 2000, '<2000',\n",
    "                             np.where(predictions_v1['model_pred'] < 3000, '<3000',\n",
    "                             np.where(predictions_v1['model_pred'] < 5000, '<5000',\n",
    "                             np.where(predictions_v1['model_pred'] < 7000, '<7000',\n",
    "                             np.where(predictions_v1['model_pred'] < 10000, '<10000',\n",
    "                             np.where(predictions_v1['model_pred'] < 15000, '<15000',\n",
    "                             np.where(predictions_v1['model_pred'] < 20000, '<20000',\n",
    "                             np.where(predictions_v1['model_pred'] >= 20000, '>=20000',\n",
    "                                      'Unknown')))))))))))))))))))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['error'] = abs((predictions_v1['predicted'] - predictions_v1['actual'])/predictions_v1['actual'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['error_grp'] = np.where(predictions_v1['error'] < 0.15, '0-15',\n",
    "                              np.where(predictions_v1['error'] < 0.30, '15-30',\n",
    "                              np.where(predictions_v1['error'] < 0.50, '30-50',\n",
    "                              np.where(predictions_v1['error'] < 0.75, '50-75',\n",
    "                              np.where(predictions_v1['error'] >= 0.75, '75+',\n",
    "                              'Unknown')))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['error_model'] = abs((predictions_v1['predicted'] - predictions_v1['RT_MODEL_EST_INCOME'])/predictions_v1['RT_MODEL_EST_INCOME'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['error_model_grp'] = np.where(predictions_v1['error_model'] < 0.15, '0-15',\n",
    "                              np.where(predictions_v1['error_model'] < 0.30, '15-30',\n",
    "                              np.where(predictions_v1['error_model'] < 0.50, '30-50',\n",
    "                              np.where(predictions_v1['error_model'] < 0.75, '50-75',\n",
    "                              np.where(predictions_v1['error_model'] >= 0.75, '75+',\n",
    "                              'Unknown')))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['count']=1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1['count'].sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1=predictions_v1.fillna(-9999)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#deneme \n",
    "# tablo oluşturma\n",
    "predictions_v2= pd.pivot_table(predictions_v1,index=['sample','actual_grp','pred_grp','model_grp','act_pred_grp','model_pred_grp',\n",
    "                                                     'error_grp','error_model_grp',\n",
    "                                                     'ANA_GRUP_FLAG', 'RT_WORKING_TYPE_DESC', \n",
    "                                                     'RT_SECTOR_TYPE_DESC', 'RT_OCCUPATION',  \n",
    "                       'MESLEK_SEGMENT', 'KKB_FLAG', 'S314_flag','S314_grp',\n",
    "                       'APP_SALARY_AMT_LM_FLAG', 'APP_SALARYCUST_FLAG','RT_MODEL_EST_INCOME_FLAG',\n",
    "                       'RT_KKB_CC_O_TOT_LIM_FLAG','RT_KKB_CC_MAX_O_C_LIM_L1Y_FLAG',\n",
    "                       'M144','DECL_INCOME_FLAG','RT_KKB_INST_O_TOT_INST_AMT_FLAG',\n",
    "                       'SKY_FLAG_URUN','AGE_GRUP','YENISALARYCUSTFLAG','income_seg','APP_FINAL_DECISION',\n",
    "                       'DOC_INCOME_FLAG'],\n",
    "                            aggfunc={'APP_SH_DECL_INCOME' : np.sum,\n",
    "                                     'APP_SH_DOC_INCOME' : np.sum, \n",
    "                                     'RT_MODEL_EST_INCOME' : np.sum,\n",
    "                                     'S314':np.sum,\n",
    "                                     'APP_SALARY_AMT_LM' :np.sum,\n",
    "                                     'DOC_INCOME_FINAL' :np.sum , 'target_w_inf' : np.sum,\n",
    "                                     'count' : np.sum, \n",
    "                                     'RT_KKB_CC_O_TOT_LIM' : np.sum, 'RT_KKB_CC_MAX_O_C_LIM_L1Y' : np.sum,\n",
    "                                     'RT_KKB_INST_O_TOT_INST_AMT': np.sum,\n",
    "                                     'actual':np.sum,\n",
    "                                     'predicted':np.sum\n",
    "                                     })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v2.reset_index().to_excel(path + str(\"Analysis_perf.xlsx\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_v1.to_csv(path + str(\"AllData_KKBli_Yeni.csv\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##BURADAN SONRASI ÇALIŞTIRILMAYACAK"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##ENCODING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "encoding_matrix['variable'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_enc = pd.merge(predictions_v1,data[['APP_ID','MESLEK_SEGMENT', 'RT_CUST_SEGMENT', 'APP_BANK_REGION_CODE',\n",
    "       'M144']],on='APP_ID',how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_enc=predictions_enc.fillna('-9999')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "writer = ExcelWriter(path + str('Encoding_List.xlsx'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in ('MESLEK_SEGMENT','RT_CUST_SEGMENT','APP_BANK_REGION_CODE','M144'):\n",
    "    predictions_enc[[str(i)+str('_x'),str(i)+str('_y'),'APP_ID']].groupby([str(i)+str('_x'),str(i)+str('_y')]).count().reset_index().to_excel(writer, sheet_name = str(i),index=False)\n",
    "writer.save()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
