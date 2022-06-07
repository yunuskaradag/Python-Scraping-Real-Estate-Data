{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import numpy as np \n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\sgunduz.experian\\Anaconda3\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3418: DtypeWarning: Columns (20,55,90,198,234,255,308,309,310,311,323,324) have mixed types.Specify dtype option on import or set low_memory=False.\n",
      "  exec(code_obj, self.user_global_ns, self.user_ns)\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv(r'D:\\gtp-files\\Final\\3.Hedef Değişken Belirleme\\FAZ1\\KKBLI_YENI_v1\\KKBli_Yeni_target_final.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Unnamed: 0             1076695\n",
       "ANA_GRUP_FLAG          1076695\n",
       "ASGARI_UCRET           1076695\n",
       "DECL_INCOME_FLAG       1076695\n",
       "FRAUD_FLAG             1076695\n",
       "                        ...   \n",
       "target                 1076695\n",
       "count                  1076695\n",
       "inflation              1076695\n",
       "target_w_inf           1076695\n",
       "outlier_income_flag    1076695\n",
       "Length: 416, dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|████████████████████████████████████████| 179/179 [00:02<00:00, 61.90it/s]\n"
     ]
    }
   ],
   "source": [
    "##INFLATION CONVERSION TO ALL LIMIT, RISK, CASH ETC. COLUMNS\n",
    "inflation = pd.read_excel(r'D:\\gtp-files\\Final\\4.Shortlist Analizleri\\Inf_Varlist.xlsx')\n",
    "\n",
    "for i in tqdm(range(len(inflation))):\n",
    "    var=inflation['ALANLAR'].iloc[i]\n",
    "    data[var]=data[var]*data['inflation']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-6-8abda2241688>:99: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead.  To get a de-fragmented frame, use `newframe = frame.copy()`\n",
      "  data['VAR98']=data['RT_SHARED_PERC_FM3']/data['RT_SHARED_PERC_FM1']\n",
      "<ipython-input-6-8abda2241688>:100: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead.  To get a de-fragmented frame, use `newframe = frame.copy()`\n",
      "  data['VAR99']=data['RT_SHARED_PERC_FM3']/data['RT_SHARED_PERC_FM2']\n",
      "<ipython-input-6-8abda2241688>:101: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead.  To get a de-fragmented frame, use `newframe = frame.copy()`\n",
      "  data['VAR100']=data['RT_CRE_TENURE']/data['RT_CUST_TENURE']\n"
     ]
    }
   ],
   "source": [
    "##CALCULATED COLUMNS\n",
    "data['VAR1']=data['RT_MAX_LIM_SKY_NOTGUARANTEE_CC']/data['RT_MAX_LIM_SKY_NOTGUARANTEE']\n",
    "data['VAR2']=data['RT_MAX_LIM_SKY_NOTGUARANTEE_OD']/data['RT_MAX_LIM_SKY_NOTGUARANTEE']\n",
    "data['VAR3']=data['RT_MAX_LIM_SKY_NOTGUARANTEE_GP']/data['RT_MAX_LIM_SKY_NOTGUARANTEE']\n",
    "data['VAR4']=data['RT_KKB_OPND_CRE_CNT_LM']/data['RT_KKB_OPND_CRE_CNT_L3M']\n",
    "data['VAR5']=data['RT_KKB_OPND_CRE_CNT_LM']/data['RT_KKB_OPND_CRE_CNT_L6M']\n",
    "data['VAR6']=data['RT_KKB_OPND_CRE_CNT_LM']/data['RT_KKB_OPND_CRE_CNT_L12M']\n",
    "data['VAR7']=data['RT_KKB_OPND_CRE_CNT_L3M']/data['RT_KKB_OPND_CRE_CNT_L6M']\n",
    "data['VAR8']=data['RT_KKB_OPND_CRE_CNT_L3M']/data['RT_KKB_OPND_CRE_CNT_L12M']\n",
    "data['VAR9']=data['RT_KKB_OPND_CRE_CNT_L6M']/data['RT_KKB_OPND_CRE_CNT_L12M']\n",
    "data['VAR10']=data['RT_KKB_APP_CNT_LM']/data['RT_KKB_APP_CNT_L3M']\n",
    "data['VAR11']=data['RT_KKB_APP_CNT_LM']/data['RT_KKB_APP_CNT_L6M']\n",
    "data['VAR12']=data['RT_KKB_APP_CNT_LM']/data['RT_KKB_APP_CNT_L12M']\n",
    "data['VAR13']=data['RT_KKB_APP_CNT_L3M']/data['RT_KKB_APP_CNT_L6M']\n",
    "data['VAR14']=data['RT_KKB_APP_CNT_L3M']/data['RT_KKB_APP_CNT_L12M']\n",
    "data['VAR15']=data['RT_KKB_APP_CNT_L6M']/data['RT_KKB_APP_CNT_L12M']\n",
    "data['VAR16']=data['RT_KKB_CC_O_TOT_RISK']/data['RT_KKB_CC_O_TOT_LIM']\n",
    "data['VAR17']=data['RT_KKB_CC_UNPAID_RISK']/data['RT_KKB_CC_O_TOT_RISK']\n",
    "data['VAR18']=data['RT_KKB_INST_O_TOT_CL_AMT']/data['RT_KKB_INST_O_TOT_INST_AMT']\n",
    "data['VAR19']=data['RT_KKB_INST_O_TOT_GPL_AMT']/data['RT_KKB_INST_O_TOT_INST_AMT']\n",
    "data['VAR20']=data['RT_KKB_INST_O_TOT_HL_AMT']/data['RT_KKB_INST_O_TOT_INST_AMT']\n",
    "data['VAR21']=data['RT_KKB_INST_UNPAID_RISK']/data['RT_KKB_INST_O_TOT_RISK']\n",
    "data['VAR22']=data['RT_KKB_O_PL_CNT']/data['RT_KKB_O_CRE_CNT']\n",
    "data['VAR23']=data['RT_KKB_OD_O_TOT_RISK']/data['RT_KKB_OD_O_TOT_LIM']\n",
    "data['VAR24']=data['RT_KKB_OD_O_TOT_RISK']/data['RT_KKB_O_CRE_CNT']\n",
    "data['VAR25']=data['RT_KKB_O_CC_CNT']/data['RT_KKB_O_CRE_CNT']\n",
    "data['VAR26']=data['RT_KKB_O_HL_CNT']/data['RT_KKB_O_CRE_CNT']\n",
    "data['VAR27']=data['RT_KKB_O_OD_CNT']/data['RT_KKB_O_CRE_CNT']\n",
    "data['VAR28']=data['RT_KKB_O_PL_CNT']/data['RT_KKB_TOT_UNPAID_RISK']\n",
    "data['VAR29']=data['RT_KKB_CC_UNPAID_RISK']/data['RT_KKB_TOT_UNPAID_RISK']\n",
    "data['VAR30']=data['RT_KKB_INST_UNPAID_RISK']/data['RT_KKB_INST_O_TOT_RISK']\n",
    "data['VAR31']=data['RT_KKB_CC_2ND_MAX_O_C_LIM_L1Y']/data['RT_KKB_CC_MAX_O_C_LIM_L1Y']\n",
    "data['VAR32']=data['RT_KKB_HL_O_RISK']/data['RT_KKB_INST_O_TOT_RISK']\n",
    "data['VAR33']=data['RT_RM_TOT_DISTRNT_CNT_LM']/data['RT_RM_TOT_DISTRNT_CNT_L3M']\n",
    "data['VAR34']=data['RT_RM_TOT_DISTRNT_CNT_LM']/data['RT_RM_TOT_DISTRNT_CNT_L6M']\n",
    "data['VAR35']=data['RT_RM_TOT_DISTRNT_CNT_LM']/data['RT_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR36']=data['RT_RM_TOT_DISTRNT_CNT_L3M']/data['RT_RM_TOT_DISTRNT_CNT_L6M']\n",
    "data['VAR37']=data['RT_RM_TOT_DISTRNT_CNT_L3M']/data['RT_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR38']=data['RT_RM_TOT_DISTRNT_CNT_L6M']/data['RT_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR39']=data['FM1_RM_TOT_DISTRNT_CNT_LM']/data['FM1_RM_TOT_DISTRNT_CNT_L3M']\n",
    "data['VAR40']=data['FM1_RM_TOT_DISTRNT_CNT_LM']/data['FM1_RM_TOT_DISTRNT_CNT_L6M']\n",
    "data['VAR41']=data['FM1_RM_TOT_DISTRNT_CNT_LM']/data['FM1_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR42']=data['FM1_RM_TOT_DISTRNT_CNT_L3M']/data['FM1_RM_TOT_DISTRNT_CNT_L6M']\n",
    "data['VAR43']=data['FM1_RM_TOT_DISTRNT_CNT_L3M']/data['FM1_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR44']=data['FM1_RM_TOT_DISTRNT_CNT_L6M']/data['FM1_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR45']=data['FM2_RM_TOT_DISTRNT_CNT_LM']/data['FM2_RM_TOT_DISTRNT_CNT_L3M']\n",
    "data['VAR46']=data['FM2_RM_TOT_DISTRNT_CNT_LM']/data['FM2_RM_TOT_DISTRNT_CNT_L6M']\n",
    "data['VAR47']=data['FM2_RM_TOT_DISTRNT_CNT_LM']/data['FM2_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR48']=data['FM2_RM_TOT_DISTRNT_CNT_L3M']/data['FM2_RM_TOT_DISTRNT_CNT_L6M']\n",
    "data['VAR49']=data['FM2_RM_TOT_DISTRNT_CNT_L3M']/data['FM2_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR50']=data['FM2_RM_TOT_DISTRNT_CNT_L6M']/data['FM2_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR51']=data['FM3_RM_TOT_DISTRNT_CNT_LM']/data['FM3_RM_TOT_DISTRNT_CNT_L3M']\n",
    "data['VAR52']=data['FM3_RM_TOT_DISTRNT_CNT_LM']/data['FM3_RM_TOT_DISTRNT_CNT_L6M']\n",
    "data['VAR53']=data['FM3_RM_TOT_DISTRNT_CNT_LM']/data['FM3_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR54']=data['FM3_RM_TOT_DISTRNT_CNT_L3M']/data['FM3_RM_TOT_DISTRNT_CNT_L6M']\n",
    "data['VAR55']=data['FM3_RM_TOT_DISTRNT_CNT_L3M']/data['FM3_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR56']=data['FM3_RM_TOT_DISTRNT_CNT_L6M']/data['FM3_RM_TOT_DISTRNT_CNT_L12M']\n",
    "data['VAR57']=data['FM2_KKBC_TOT_RISK']/data['FM2_KKBC_TOT_LIM']\n",
    "data['VAR58']=data['FM2_KKBC_TOT_CASH_RISK']/data['FM2_KKBC_TOT_CASH_LIM']\n",
    "data['VAR59']=data['FM2_KKBC_TOT_NC_RISK']/data['FM2_KKBC_TOT_NC_LIM']\n",
    "data['VAR60']=data['FM3_KKBC_TOT_RISK']/data['FM3_KKBC_TOT_LIM']\n",
    "data['VAR61']=data['FM3_KKBC_TOT_CASH_RISK']/data['FM3_KKBC_TOT_CASH_LIM']\n",
    "data['VAR62']=data['FM3_KKBC_TOT_NC_RISK']/data['FM3_KKBC_TOT_NC_LIM']\n",
    "data['VAR63']=data['FM1_KKBC_TOT_RISK']/data['FM1_KKBC_TOT_LIM']\n",
    "data['VAR64']=data['FM1_KKBC_TOT_CASH_RISK']/data['FM1_KKBC_TOT_CASH_LIM']\n",
    "data['VAR65']=data['FM1_KKBC_TOT_NC_RISK']/data['FM1_KKBC_TOT_NC_LIM']\n",
    "data['VAR66']=data['RT_KKBC_TOT_RISK']/data['RT_KKBC_TOT_LIM']\n",
    "data['VAR67']=data['RT_KKBC_TOT_CASH_RISK']/data['RT_KKBC_TOT_CASH_LIM']\n",
    "data['VAR68']=data['RT_KKBC_TOT_NC_RISK']/data['RT_KKBC_TOT_NC_LIM']\n",
    "data['VAR69']=data['FM3_MMZC_TOT_CASH_RISK_LM']/data['FM3_MMZC_TOT_CASH_LIM_LM']\n",
    "data['VAR70']=data['FM3_MMZC_TOT_CASH_RISK_FX_LM']/data['FM3_MMZC_TOT_CASH_LIM_FX_LM']\n",
    "data['VAR71']=data['FM3_MMZC_TOT_NC_RISK_LM']/data['FM3_MMZC_TOT_NC_LIM_LM']\n",
    "data['VAR72']=data['FM3_MMZC_TOT_FACTOR_RISK_LM']/data['FM3_MMZC_TOT_FACTOR_LIM_LM']\n",
    "data['VAR73']=data['FM3_MMZC_TOT_LEASING_RISK_LM']/data['FM3_MMZC_TOT_LEASING_LIM_LM']\n",
    "data['VAR74']=data['FM3_MMZC_B_TOT_CASH_RISK_LM']/data['FM3_MMZC_B_TOT_CASH_LIM_LM']\n",
    "data['VAR75']=data['FM3_MMZC_B_TOT_NC_RISK_LM']/data['FM3_MMZC_B_TOT_NC_LIM_LM']\n",
    "data['VAR76']=data['FM2_MMZC_TOT_CASH_RISK_LM']/data['FM2_MMZC_TOT_CASH_LIM_LM']\n",
    "data['VAR77']=data['FM2_MMZC_TOT_CASH_RISK_FX_LM']/data['FM2_MMZC_TOT_CASH_LIM_FX_LM']\n",
    "data['VAR78']=data['FM2_MMZC_TOT_NC_RISK_LM']/data['FM2_MMZC_TOT_NC_LIM_LM']\n",
    "data['VAR79']=data['FM2_MMZC_TOT_FACTOR_RISK_LM']/data['FM2_MMZC_TOT_FACTOR_LIM_LM']\n",
    "data['VAR80']=data['FM2_MMZC_TOT_LEASING_RISK_LM']/data['FM2_MMZC_TOT_LEASING_LIM_LM']\n",
    "data['VAR81']=data['FM2_MMZC_B_TOT_CASH_RISK_LM']/data['FM2_MMZC_B_TOT_CASH_LIM_LM']\n",
    "data['VAR82']=data['FM2_MMZC_B_TOT_NC_RISK_LM']/data['FM2_MMZC_B_TOT_NC_LIM_LM']\n",
    "data['VAR83']=data['FM1_MMZC_TOT_CASH_RISK_LM']/data['FM1_MMZC_TOT_CASH_LIM_LM']\n",
    "data['VAR84']=data['FM1_MMZC_TOT_CASH_RISK_FX_LM']/data['FM1_MMZC_TOT_CASH_LIM_FX_LM']\n",
    "data['VAR85']=data['FM1_MMZC_TOT_NC_RISK_LM']/data['FM1_MMZC_TOT_NC_LIM_LM']\n",
    "data['VAR86']=data['FM1_MMZC_TOT_FACTOR_RISK_LM']/data['FM1_MMZC_TOT_FACTOR_LIM_LM']\n",
    "data['VAR87']=data['FM1_MMZC_TOT_LEASING_RISK_LM']/data['FM1_MMZC_TOT_LEASING_LIM_LM']\n",
    "data['VAR88']=data['FM1_MMZC_B_TOT_CASH_RISK_LM']/data['FM1_MMZC_B_TOT_CASH_LIM_LM']\n",
    "data['VAR89']=data['FM1_MMZC_B_TOT_NC_RISK_LM']/data['FM1_MMZC_B_TOT_NC_LIM_LM']\n",
    "data['VAR90']=data['RT_MMZC_TOT_CASH_RISK_LM']/data['RT_MMZC_TOT_CASH_LIM_LM']\n",
    "data['VAR91']=data['RT_MMZC_TOT_CASH_RISK_FX_LM']/data['RT_MMZC_TOT_CASH_LIM_FX_LM']\n",
    "data['VAR92']=data['RT_MMZC_TOT_NC_RISK_LM']/data['RT_MMZC_TOT_NC_LIM_LM']\n",
    "data['VAR93']=data['RT_MMZC_TOT_FACTOR_RISK_LM']/data['RT_MMZC_TOT_FACTOR_LIM_LM']\n",
    "data['VAR94']=data['RT_MMZC_TOT_LEASING_RISK_LM']/data['RT_MMZC_TOT_LEASING_LIM_LM']\n",
    "data['VAR95']=data['RT_MMZC_B_TOT_CASH_RISK_LM']/data['RT_MMZC_B_TOT_CASH_LIM_LM']\n",
    "data['VAR96']=data['RT_MMZC_B_TOT_NC_RISK_LM']/data['RT_MMZC_B_TOT_NC_LIM_LM']\n",
    "data['VAR97']=data['RT_SHARED_PERC_FM2']/data['RT_SHARED_PERC_FM1']\n",
    "data['VAR98']=data['RT_SHARED_PERC_FM3']/data['RT_SHARED_PERC_FM1']\n",
    "data['VAR99']=data['RT_SHARED_PERC_FM3']/data['RT_SHARED_PERC_FM2']\n",
    "data['VAR100']=data['RT_CRE_TENURE']/data['RT_CUST_TENURE']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "##SHORTLIST \n",
    "fields1=pd.read_excel(r'D:\\gtp-files\\Final\\4.Shortlist Analizleri\\Shortlist_Final_v6_Faz1_Final.xlsx', sheet_name = \"shortlist\" )\n",
    "fields_l1 = fields1['ColumnName'][fields1['Faz1_Model_Vars'] == 1]\n",
    "fields2=pd.read_excel(r'D:\\gtp-files\\Final\\4.Shortlist Analizleri\\Shortlist_Final_v6_Faz1_Final.xlsx', sheet_name = \"keyvars\" )\n",
    "fields_l2 = fields2['ColumnName']\n",
    "\n",
    "fields = fields_l1.append(fields_l2)\n",
    "\n",
    "\n",
    "\n",
    "#SHORTLIST DATA SELECTION\n",
    "\n",
    "data= data[fields]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0          TIK31675225\n",
       "1          TIK36636091\n",
       "2          TIK27476984\n",
       "3          TIK29631273\n",
       "4          KMH28175308\n",
       "              ...     \n",
       "1076690    LMT10124387\n",
       "1076691     LMT9762784\n",
       "1076692    TIK32464605\n",
       "1076693    TIK30429224\n",
       "1076694     LMT9357680\n",
       "Name: APP_ID, Length: 1076695, dtype: object"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['APP_ID']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "##LABEL ENCODING\n",
    "#manuel encoding excelde elle sayı ataması yapılır:\n",
    "df_meslek = pd.read_excel(r'D:\\gtp-files\\Final\\2.Segmentasyon\\Encoding_List_Faz1.xlsx',sheet_name='MESLEK_SEGMENT')\n",
    "#df_m144 = pd.read_excel(r'D:\\gtp-files\\Final\\2.Segmentasyon\\Encoding_List_Faz1.xlsx',sheet_name='M144')\n",
    "df_cust_seg = pd.read_excel(r'D:\\gtp-files\\Final\\2.Segmentasyon\\Encoding_List_Faz1.xlsx',sheet_name='RT_CUST_SEGMENT')\n",
    "df_bank_reg = pd.read_excel(r'D:\\gtp-files\\Final\\2.Segmentasyon\\Encoding_List_Faz1.xlsx',sheet_name='APP_BANK_REGION_CODE')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.merge(data,df_meslek,on='MESLEK_SEGMENT',how='left')\n",
    "#data = pd.merge(data,df_m144,on='M144',how='left')\n",
    "data = pd.merge(data,df_cust_seg,on='RT_CUST_SEGMENT',how='left')\n",
    "data = pd.merge(data,df_bank_reg,on='APP_BANK_REGION_CODE',how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.drop(columns=['MESLEK_SEGMENT','RT_CUST_SEGMENT','APP_BANK_REGION_CODE'], inplace=True)\n",
    "\n",
    "data.rename(columns={'MESLEK_SEGMENT_L': 'MESLEK_SEGMENT',\n",
    "                     #'M144_L':'M144',\n",
    "                    'RT_CUST_SEGMENT_L':'RT_CUST_SEGMENT',\n",
    "                  'APP_BANK_REGION_CODE_L':'APP_BANK_REGION_CODE'}, inplace = True) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RT_WORKING_TYPE_DESC        1031123\n",
       "RT_SECTOR_TYPE_DESC         1006512\n",
       "RT_OCCUPATION               1076171\n",
       "MIN_SALARY_FLAG             1076695\n",
       "FM1_MMZC_TOT_CASH_LIM_LM        924\n",
       "                             ...   \n",
       "APP_DATE                    1076695\n",
       "target_w_inf                1076695\n",
       "MESLEK_SEGMENT              1076695\n",
       "RT_CUST_SEGMENT             1076695\n",
       "APP_BANK_REGION_CODE        1076695\n",
       "Length: 151, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "##FILL NULL WITH 9999\n",
    "data.fillna(-9999,inplace=True)\n",
    "data=data.replace([np.nan],-9999)\n",
    "data=data.replace([np.inf,-np.inf],-9999)\n",
    "data=data.replace([np.inf,-np.inf],-9999)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "##ONE HOT ENCODING\n",
    "\n",
    "##CHANGE FLOAT64 TO INT\n",
    "data[['RT_OCCUPATION','RT_SECTOR_TYPE_DESC','RT_WORKING_TYPE_DESC','RT_CUSTOMER_TYPE',\n",
    "                                        'RT_EMPLOYMENT_TYPE']] = data[['RT_OCCUPATION','RT_SECTOR_TYPE_DESC','RT_WORKING_TYPE_DESC','RT_CUSTOMER_TYPE',\n",
    "                                        'RT_EMPLOYMENT_TYPE']].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 1       957858\n",
       "-9999     45572\n",
       " 4        41230\n",
       " 2        13257\n",
       " 0        10951\n",
       " 3         7827\n",
       "Name: RT_WORKING_TYPE_DESC, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['RT_WORKING_TYPE_DESC'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#0 gelenleri -9999 doldurma:\n",
    "\n",
    "data['RT_WORKING_TYPE_DESC'] = np.where(data['RT_WORKING_TYPE_DESC']==0,-9999,data['RT_WORKING_TYPE_DESC'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 1       957858\n",
       "-9999     56523\n",
       " 4        41230\n",
       " 2        13257\n",
       " 3         7827\n",
       "Name: RT_WORKING_TYPE_DESC, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['RT_WORKING_TYPE_DESC'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#one-hot encoding:\n",
    "data = pd.get_dummies(data, columns=['RT_OCCUPATION','RT_SECTOR_TYPE_DESC','RT_WORKING_TYPE_DESC','RT_CUSTOMER_TYPE',\n",
    "                                        'RT_EMPLOYMENT_TYPE']) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "#kolon ismi düzeltme:\n",
    "data.rename(columns={'target_w_inf': 'TARGET',\n",
    "                        'RT_SECTOR_TYPE_DESC_-9999':'RT_SECTOR_TYPE_DESC_9999',\n",
    "                        'RT_OCCUPATION_-9999':'RT_OCCUPATION_9999', \n",
    "                        'RT_WORKING_TYPE_DESC_-9999':'RT_WORKING_TYPE_DESC_9999',\n",
    "                        'RT_EMPLOYMENT_TYPE_-9999': 'RT_EMPLOYMENT_TYPE_9999',\n",
    "                        'RT_CUSTOMER_TYPE_-9999':'RT_CUSTOMER_TYPE_9999'}, inplace = True) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['MIN_SALARY_FLAG', 'FM1_MMZC_TOT_CASH_LIM_LM',\n",
       "       'FM1_MMZC_TOT_ST_CASH_RISK_LM', 'FM1_MMZC_TOT_RESTRUCT_RISK_LM',\n",
       "       'FM1_MMZC_TOT_LIQ_RISK_LM', 'FM1_MMZC_TOT_PRBLMTC_RISK_LM',\n",
       "       'FM1_MMZC_TOT_FACTOR_LIM_LM', 'FM1_MMZC_TOT_LEASING_LIM_LM',\n",
       "       'FM1_MMZC_B_TOT_NC_LIM_LM', 'FM1_MMZC_B_TOT_ST_CASHRISK_LM',\n",
       "       ...\n",
       "       'RT_EMPLOYMENT_TYPE_43', 'RT_EMPLOYMENT_TYPE_44',\n",
       "       'RT_EMPLOYMENT_TYPE_45', 'RT_EMPLOYMENT_TYPE_46',\n",
       "       'RT_EMPLOYMENT_TYPE_47', 'RT_EMPLOYMENT_TYPE_49',\n",
       "       'RT_EMPLOYMENT_TYPE_50', 'RT_EMPLOYMENT_TYPE_51',\n",
       "       'RT_EMPLOYMENT_TYPE_52', 'RT_EMPLOYMENT_TYPE_55'],\n",
       "      dtype='object', length=293)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#modelleme için datayı csvye çıkma:\n",
    "data.to_csv(r'D:\\gtp-files\\Final\\3.Hedef Değişken Belirleme\\FAZ1\\KKBLI_YENI_v1\\KKBli_Yeni_target_final_new_sl.csv')"
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
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
