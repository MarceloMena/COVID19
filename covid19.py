import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn import svm

def growth_curve_LR(df, country, DFN = 0 ):
    df = df.drop(['Province/State', 'Lat', 'Long'], axis = 1)
    #df = df.set_index('Country/Region')
    test1 = df.loc[df['Country/Region'] == country].sum()
    test1 = test1.to_frame()
    #print('cases:',test1)
    test1 = test1.rename(columns={0: "cases"})
    #test1 = test1.drop(['Country/Region', 'Lat', 'Long'], axis = 0)
    test1_clean = test1[test1.cases > 0]
    test1_clean = test1_clean.reset_index(drop = True)
    #print('cases:',test1_clean)
    y_cases = (test1_clean.cases.astype(float))
    x_days = test1_clean.index.to_numpy().reshape(len(test1_clean.index.to_numpy()),1)
    y_cases_log = np.log(test1_clean.cases.astype(float)).to_numpy().reshape(len(test1_clean.index.to_numpy()),1)
    y_cases_log.shape
    regr = LinearRegression()
    regr.fit(x_days, y_cases_log)
    #print('Theta 0: ', regr.intercept_[0])
    #print('Theta 1: ', regr.coef_[0][0])
    #print(f'Days since the first case {len(test1_clean)}')
    stop = len(x_days) + DFN
    x_days_predict = np.linspace(0,stop-1,stop)
    x_days_predict = x_days_predict.reshape(len(x_days_predict),1)
    plt.scatter(x_days,y_cases, s = 50)
    plt.plot(x_days_predict, np.exp(regr.predict(x_days_predict)), color = 'orange', linewidth = 3)
    plt.title(f'COVID in {country}')
    plt.xlabel('days since first case')
    plt.ylabel('number of cases')
    plt.show()
    predict = np.exp(regr.predict(x_days_predict))
    predict[(len(predict)-1)]
    print(f'There might be {predict[(len(predict)-1)]}number of cases in {DFN} days')
    return regr.intercept_[0],regr.coef_[0][0], predict

def growth_curve_SVM(df, country, DFN = 0, degree = 3):
    test1 = df.loc[df['Country/Region'] == country].sum()
    test1 = test1.to_frame()
    #print('cases:',test1)
    test1 = test1.rename(columns={0: "cases"})
    test1 = test1.drop(['Country/Region', 'Lat', 'Long'], axis = 0)
    test1_clean = test1[test1.cases > 0]
    test1_clean = test1_clean.reset_index(drop = True)
    #print('cases:',test1_clean)
    y_cases = (test1_clean.cases.astype(float))
    x_days = test1_clean.index.to_numpy().reshape(len(test1_clean.index.to_numpy()),1)
    y_cases_log = np.log(test1_clean.cases.astype(float)).to_numpy().reshape(len(test1_clean.index.to_numpy()),1)
    y_cases_log.shape
    svm_regr = svm.SVR(kernel='linear', degree = degree)
    svm_regr.fit(x_days, y_cases_log)
    print(f'Days since the first case {len(test1_clean)}')
    stop = len(x_days) + DFN
    x_days_predict = np.linspace(0,stop-1,stop)
    x_days_predict = x_days_predict.reshape(len(x_days_predict),1)
    plt.scatter(x_days,y_cases, s = 50)
    plt.plot(x_days_predict, np.exp(svm_regr.predict(x_days_predict)), color = 'orange', linewidth = 3)
    plt.title(f'COVID in {country}')
    plt.xlabel('days since first case')
    plt.ylabel('number of cases')
    plt.show()
    predict = np.exp(svm_regr.predict(x_days_predict))
    predict[(len(predict)-1)]
    print(f'There might be {predict[(len(predict)-1)]}number of cases in {DFN} days')
    return svm_regr.coef_,svm_regr.intercept_, predict

def data_cleaner(df, country, min_cases=1):
    """
    Cleans all the data and give you the information only in the days since there are confirmed y_cases

    df: Kaggle Dataframe 'covid_19_data.csv'
    cuntry: string with the name of the country that you want to study

    """

    df = df.drop(['Province/State', 'Lat', 'Long'], axis = 1)
    df.set_index('Country/Region')
    per_country = df.loc[df['Country/Region'] == country].sum()
    per_country = per_country.to_frame()
    per_country = per_country.drop(['Country/Region'], axis = 0)
    per_country = per_country.rename(columns={0: "cases"})
    per_country_clean = per_country[per_country.cases >= min_cases]
    per_country_clean = per_country_clean.reset_index(drop = True)

    return per_country_clean

def log_converter(df):
    """
    Transform the number of cases to logaritmic and creates a new growth_curve_LR

    df: Accpets only the output from data_cleaner
    """
    df['log cases']  = np.log(df.cases.astype(float))
    return df

def data_reshaper(df):
    data_matrix = []
    data_matrix.append(df.index.to_numpy().reshape(len(df.to_numpy()),1))
    for i in df.columns:
        data_matrix.append(df[i].to_numpy().reshape(len(df.to_numpy()),1))
    return data_matrix

def log_svm_prediction(df, DFN = 0, degree = 3):
    days = df[0]
    log_cases = df[2]
    svm_regr = svm.SVR(kernel='poly', degree = degree)
    svm_regr.fit(days, log_cases.ravel())
    stop = len(days) + DFN
    x_days_predict = np.linspace(0,stop-1,stop)
    x_days_predict = x_days_predict.reshape(len(x_days_predict),1)
    log_predict = svm_regr.predict(x_days_predict)
    log_predict_now =  log_predict[:len(days)]
    predict = np.exp(svm_regr.predict(x_days_predict))
    score = svm_regr.score(days, log_cases,log_predict_now)
    return predict[stop-1], x_days_predict, log_predict, score

def predictor(df, country, DFN = 0, plot = True, degree = 1, pres = 0.9, min_days = 2, min_cases = 1):

    country_data = data_cleaner(df,country, min_cases)
    days_infection = len(country_data)
    if days_infection >=min_days:
        country_data_log = log_converter(country_data)
        data_matrix = data_reshaper(country_data_log)
        try:
            prediction_day, x_days_predict, log_predict, score = log_svm_prediction(data_matrix, DFN, degree = degree)
            #print(score)

            prediction = np.exp(log_predict)
            pred_df = pd.DataFrame(data = [prediction, log_predict]).T
            pred_df = pred_df.rename(columns = {0: "predicted cases", 1: "predicted log cases"})
            cases_country = pd.concat([country_data_log, pred_df], axis =1)

            if score > pres and plot == True:
                plotter(cases_country, data_matrix, prediction_day, score, country, DFN)
        except:
            prediction = 0
            score = 0
            cases_country = []
    else:
        prediction = 0
        score = 0
        cases_country = []

    return prediction, score, cases_country, days_infection

def plotter(cases_country, data_matrix, prediction_day, score, country, DFN):
    plt.figure(figsize = [12, 6])

    plt.subplot(2,1,1)
    plt.scatter(cases_country.index, cases_country['cases'], s = 50)
    plt.plot(cases_country.index, cases_country['predicted cases'], color = 'orange', linewidth = 3)

    plt.title(f'Evolution and prediction of cases of COVID-19 in {country}',fontsize = 15)
    plt.yscale('log')
    plt.ylabel('Number of cases \n [log scale]',fontsize = 12)
    plt.grid()

    plt.subplot(2,1,2)

    plt.scatter(cases_country.index, cases_country['cases'], s = 50)
    plt.plot(cases_country.index, cases_country['predicted cases'], color = 'orange', linewidth = 3)
    plt.xlabel('Days sice the first confiermed case',fontsize = 12)
    plt.ylabel('Number of cases',fontsize = 12)
    plt.grid()

    plt.show()

    print(f'Days since first confirmed case was {len(data_matrix[0])} days ago')
    print(f'Number of predicted cases in {DFN} days is {prediction_day}')
    print('The score of this prediction is:', score)
    return

def data_analizer_by_params(df, DFN, list_countries, degree = 1, min_cases = 10, min_days = 2, min_score = 0):
    """
    Data Analizer for 'time_series_covid_19_confirmed.csv' as a Dataframe that
    as output give us 4 DataFrames of the prediction and real cases of the
    countries that fullfil all the parameters

    Parameters
    ----------

    df : Dataframe from de 'time_series_covid_19_confirmed.csv'
    DFN: Days from now. the number of days that are going to be proficted in to
    the days_future
    degree : degree of the SVM regressor 'poly' that is going to be used
    min_cases : minimun of cases that a country needs to have in order to make
    the predictions
    min_days : minumun of days that the country needs to have above the minimun
    of cases to make the predictions, it is recommended that this number is not
    less than 2
    min_score = minimum score that the prediction needs to have to be valid

    """

    country = df['Country/Region'].unique()

    days = range(df.shape[1]-4)
    days_future = range(df.shape[1]-4+DFN)

    cases_real = pd.DataFrame(columns=country, index = days)
    cases_real_log = pd.DataFrame(columns=country, index = days)
    cases_projections = pd.DataFrame(columns=country, index = days_future)
    cases_projections_log = pd.DataFrame(columns=country, index = days_future)
    cases_projections_future_only = pd.DataFrame(columns=country, index = days_future)
    n=0
    m=[]
    for i in list_countries:
        prediction, score, data_country, days_infection= predictor(df, i,
        DFN = DFN, plot = False, degree = degree, min_cases = min_cases,
        min_days = min_days, pres = min_score)
        try:
            cases_real[i] = data_country.cases
            cases_real_log[i] = data_country['log cases']
            cases_projections[i] = data_country['predicted cases']
            cases_projections_log[i] = data_country['predicted log cases']

            rever_pred_cases = data_country['predicted cases'].iloc[::-1].reset_index()
            cases_projections_future_only[i] = rever_pred_cases['predicted cases']
            cases_projections_future_only = cases_projections.drop('index')

            #m.append(i)
            #print(f'{i} has {len(cases_real[i])} days since the first cornfirmed case')
        except:
            n+=1
    #print('The countries that fullfil the requirements are:', m)

    cols = cases_real.sum()!=0
    countries_selected_real = cases_real[cols[cols].index]
    countries_selected_real_log = cases_real_log[cols[cols].index]
    countries_selected_projections = cases_projections[cols[cols].index]
    countries_selected_projections_log = cases_projections_log[cols[cols].index]

    countries_selected_projections_future_only = cases_projections_future_only[cols[cols].index]
    countries_selected_projections_future_only = countries_selected_projections_future_only[::-1].reset_index().drop('index', axis = 1)
    return countries_selected_real, countries_selected_projections, countries_selected_projections_future_only
