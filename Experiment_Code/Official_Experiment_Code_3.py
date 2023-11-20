import numpy as np
import pandas as pd
import Official_Experiment_Function_3 as rf
import warnings

import matplotlib.pyplot as plt

"""Pre_config"""
warnings.simplefilter(action='ignore', category=FutureWarning)

allow_clean=False
if 'SPY_PYTHONPATH' in rf.os.environ:
    allow_clean = True
    """
    Check Last line for clean variables
    """

"""Pre_config END"""



"""
Creating time-series of relevant dates
"""
time_series=pd . date_range ( start ='31-10-2013 ',end ='30-09-2023 ', freq ='M') #Date series

"""
Retrieve Subset equity and Interest rate
"""
sheet=-1
col=""
while sheet < 0 or sheet>1:
    sheet=int(input("Which Risk Free rate you want? press 0 for Euribor or 1 for Bund: "))

match sheet:
    case 0:
         sheet = "EURIBOR_3_M"
         col="BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ"
    case 1:
        sheet = "BUND"
        col="RF GERMANY GVT BMK BID YLD 3M - RED. YIELD"      


EuroStoxx = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EUROSTOXX600").iloc[: , 1:]
Subset_Stock_Selected= pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset").iloc[: , 1:]
Interest = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name=sheet)

"""
Clean the data 
Obtain monthly rates from annualized interest rates time-series
"""
Subset_Stock_Selected.columns = Subset_Stock_Selected.columns.str.replace(" - TOT RETURN IND","")
RFREE=np.array(Interest[[col]]/12)

"""
a) Computing the logarithmic excess returns for:
    1) EuroStoxx600 Total Return Index
    2) Sample of stocks

b) Inserting them in a dataframe
"""
stock_names = list(Subset_Stock_Selected.columns)

df_factors = pd.DataFrame(data=
                         np.subtract(np.array(100 *( np . log ( EuroStoxx ) 
                                     -np . log ( EuroStoxx . shift (1) )) ),
                                   RFREE),
                      columns = ['Market']).iloc[1:,]

df_stocks = pd.DataFrame( 
                         np.subtract(np.array(100 *( np . log ( Subset_Stock_Selected ) 
                                     -np . log ( Subset_Stock_Selected . shift (1) )) ),
                                   RFREE),
                      columns = stock_names)
df_stocks = df_stocks.iloc[1:,]
df_stocks.index = df_factors.index = time_series


"""
Here in the argument title the content of the string: "sheet" is added
"""

l = rf.plotscatter(df_factors,df_stocks,"Excess Returns vs Eurostoxx -"+sheet,
            "Market Excess Returns",
            "",sheet,
            ("2_Excess_return", "Excess_return")
            )

"""
Here we estimate the CAPM model for each stock.
    -CAPM_summary is a dataframe in which we stored the alphas, betas plus their
    respective p-values and the R-Squared of each estimation.
    -CAPM_list is a list containing the complete output of each linear regression
"""

CAPM_summary, CAPM_list = rf.OLS(df_stocks,df_factors)
CAPM_summary.loc['Mean'] = CAPM_summary.mean()
CAPM_summary = CAPM_summary.sort_values('p-value_alpha')

CAPM_summary.to_excel("3_CAPM_stocks.xlsx")


"""
Here we create a (2,3) plot with:
    row 1: the 3 scatter plots showing the best linear approximation
    row 2: the 3 scatter plots showing the worst linear approximation
"""
CAPM_summary = CAPM_summary.sort_values('R-Squared')

rf.m_scatter(CAPM_summary, df_factors, df_stocks,
             ("2_scatter_comparison","2_scatter_comparison"))

"""
P-values analysis
"""
CAPM_summary = CAPM_summary.sort_values('p-value_alpha')
rf.plotbar(CAPM_summary['p-value_alpha'],("3_p_value_plots"))

CAPM_summary = CAPM_summary.sort_values('p-value_beta: Market')
rf.plotbar(CAPM_summary['p-value_beta: Market'],"3_p_value_plots")

"""
F-TEST Comparison
"""

        
F_test_p_values = rf.f_test_retrieval(CAPM_list)

F_test_p_values.loc['Mean'] = F_test_p_values.mean()
F_test_p_values = F_test_p_values.sort_values('F-Test_p-value')

rf.plotbar(F_test_p_values['F-Test_p-value'],"3_F_test_p_value_plots")
    

"""
THIS USES THE PREVIOUSLY DEFINED CODE TO PLOT THE OLS FITTED LINE IN THE 
SCATTER PLOTS
"""

rf.plotCAPM(df_stocks,
            df_factors,
            stock_names,
            CAPM_summary.loc[CAPM_summary.index != 'Mean'],
            ("2_testCAPM","CAPM",sheet))

"""
Computes excess returns of a equally-weighted portfolio for each month
"""
array_stocks = np.array(df_stocks.T)
Excess_equi_valued = np.array(sum(array_stocks)/len(array_stocks)).reshape(-1,1)

df_portfolios = pd.DataFrame(data = Excess_equi_valued, 
                            columns = ['Portfolio - EW'],
                            index = time_series)


"""
Runs a CAPM using as independent variable the excess return of said portfolio
"""

CAPM_EW_Portfolio_Summary, CAPM_EW_Portfolio = rf.OLS(
                             df_portfolios,
                             df_factors)

"""
Comparison between:
    1) the average value of the parameters obtained for the single stocks
    2) the value of the parameters obtained using the equally-weighted portfolio
"""

comparison_stocks_EW_portfolio = pd.concat([CAPM_summary.loc['Mean',:], 
                        CAPM_EW_Portfolio_Summary.iloc[0,:]],
                       axis = 1).T


rf.comparison_scatter(df_stocks,df_portfolios['Portfolio - EW'],
            df_factors['Market'],
            "Excess Returns vs Eurostoxx -"+sheet,
            "Market Excess Returns",
            "Excess Returns",
            ("3_Comparison_Scatter",sheet),
            CAPM_EW_Portfolio_Summary
            )

rf.comparison_scatter(df_stocks,df_portfolios['Portfolio - EW'],
            df_factors['Market'],
            "Comparison between the scatter plots",
            "Market Excess Returns(" +sheet+")",
            "Excess Returns(" +sheet+")",
            ("3_Comparison_Scatter",sheet)
            )

comparison_stocks_EW_portfolio.T.to_excel("3_Comparison.xlsx")


"""
RESET TEST
"""

a = rf.RESET_test(CAPM_list)
b = rf.RESET_test(CAPM_EW_Portfolio)
diag_CAPM_RESET = pd.concat([a,b], axis = 0)

diag_CAPM_RESET.loc['Mean'] = diag_CAPM_RESET.mean()
diag_CAPM_RESET = diag_CAPM_RESET.sort_values('p-value')

diag_CAPM_RESET.to_excel("4_RESET_test.xlsx")
rf.plotbar(diag_CAPM_RESET['p-value'],"4_p_value_RESET")



"""
WHITE TEST
"""

a = rf.h_test(CAPM_list)
b = rf.h_test(CAPM_EW_Portfolio)
diag_CAPM_het_WHITE = pd.concat([a,b], axis = 0)

diag_CAPM_het_WHITE.loc['Mean'] = diag_CAPM_het_WHITE.mean()
diag_CAPM_het_WHITE = diag_CAPM_het_WHITE.sort_values('p-value')

diag_CAPM_het_WHITE.to_excel("4_WHITE_test.xlsx")
rf.plotbar(diag_CAPM_het_WHITE['p-value'],"4_p_value_WHITE")


"""
DURBIN-WATSON TEST
"""

a = rf.Durbin_Watson_test(CAPM_list)
b = rf.Durbin_Watson_test(CAPM_EW_Portfolio)
diag_CAPM_serialcor_DW = pd.concat([a,b], axis = 0)

diag_CAPM_serialcor_DW.loc['Mean'] = diag_CAPM_serialcor_DW.mean()
diag_CAPM_serialcor_DW = diag_CAPM_serialcor_DW.sort_values('Test-statistic')

diag_CAPM_serialcor_DW.to_excel("4_DW_test.xlsx")
rf.plotbar(diag_CAPM_serialcor_DW['Test-statistic'],"4_p_value_DW",
           ten_value = 1.8)

"""
BREUSCH-GODFREY TEST
"""

a = rf.Breusch_Godfrey_test(CAPM_list)
b = rf.Breusch_Godfrey_test(CAPM_EW_Portfolio)
diag_CAPM_serialcor_BG = pd.concat([a,b], axis = 0)

diag_CAPM_serialcor_BG.loc['Mean'] = diag_CAPM_serialcor_BG.mean()
diag_CAPM_serialcor_BG = diag_CAPM_serialcor_BG.sort_values('p-value')

diag_CAPM_serialcor_BG.to_excel("4_DW_test.xlsx")
rf.plotbar(diag_CAPM_serialcor_BG['p-value'],"4_p_value_BG")

"""
FAMA-FRENCH download and cleaning
"""

FF = pd . read_excel('fama_french.xlsx',sheet_name="four")
FF.columns = FF.iloc[0,:]
FF = FF.drop([0])
FF = FF.iloc[-120:,:]
FF.index = time_series

temp = pd.read_excel('fama_french.xlsx', sheet_name = 'mom')
temp.columns = temp.iloc[0,:]
temp = temp.drop([0])
temp = temp.iloc[-120:,:]
temp.index = time_series

FF = pd.concat([FF,temp], axis = 1)


l = ['SMB', 'CMA', 'RMW', 'HML', 'MOM']
FF = FF.loc[:,l]

FF = FF.astype(float)

df_factors = pd.concat([df_factors, FF], axis = 1)

"""
FAMA-FRENCH ESTIMATION
"""
FF_summary, FF_list = rf.OLS(df_stocks,df_factors)
FF_summary.loc['Mean'] = FF_summary.mean()

FF_summary = FF_summary.sort_values('p-value_alpha')
FF_summary.to_excel("5_FF_stocks.xlsx")

"""
Alpha p-values comparison
"""

CAPM_summary = CAPM_summary.sort_index()
FF_summary = FF_summary.sort_index()

"""
Comparison with barplot
"""

rf.comparison_barplot(FF_summary, CAPM_summary)

"""
Check for multicollinearity
"""

factor_corr_matrix = df_factors.corr()

"""
GETS Procedure
"""

GETS_summary, bic_list = rf.GETS(FF_summary, df_factors, df_stocks)


FF_summary_port, FF_list_port = rf.OLS(df_portfolios,df_factors)

"""
ELIMINATING THE CORRELATED COVARIATES FIRST
"""
l = ['MOM', 'HML']

df_factors_2 =df_factors.drop(l, axis = 1)

FF_summary_2, FF_list = rf.OLS(df_stocks,df_factors_2)
FF_summary_2.loc['Mean'] = FF_summary_2.mean()

GETS_summary2, bic_list2 = rf.GETS(FF_summary_2, df_factors_2, df_stocks)

"""
Check why the only relevant variable appears to be the market
"""

plt.figure()

plt.plot(time_series, df_factors.loc[:, 'Market'], label = 'Market')
plt.plot(time_series, df_portfolios.iloc[:,0], label = 'Portfolio')
plt.legend()

plt.close()


