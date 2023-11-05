import numpy as np
import pandas as pd
import Regre_Function_fm_3 as rf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt



t=pd . date_range ( start ='15-09-2013 ',end ='15-09-2023 ', freq ='M') #Date series

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


EuroStoxx = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EUROSTOXX600")
Subset_Stock_Selected= pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset")
Interest = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name=sheet)

"""
Clean the data 
Obtain monthly rates from annualized interest rates time-series
"""

Subset_Stock_Selected.columns = Subset_Stock_Selected.columns.str.replace(" - TOT RETURN IND","")
EuroStoxx = EuroStoxx.loc[:, EuroStoxx.columns != 'Name']#Delete column of date
Subset_Stock_Selected=Subset_Stock_Selected.loc[:, Subset_Stock_Selected.columns != 'Name'] #Delete column of date
0
RFREE=np.array(Interest[[col]]/12)


"""
a) Computing the logarithmic excess returns for:
    1) EuroStoxx600 Total Return Index
    2) Sample of stocks

b) Inserting them in a dataframe
"""
stock_names = list(Subset_Stock_Selected.columns)
print(type(np.subtract(100*np.log(EuroStoxx) - np . log ( EuroStoxx . shift (1) ),RFREE)) )

df_factors = pd.DataFrame(data = 
                         np.subtract(np.array(100 *( np . log ( EuroStoxx ) 
                                     -np . log ( EuroStoxx . shift (1) )) ),
                                   RFREE),
                      columns = ['Market'])
df_factors = df_factors.iloc[1:,]
df_factors.index = t

del EuroStoxx


df_stocks = pd.DataFrame(data = 
                         np.subtract(np.array(100 *( np . log ( Subset_Stock_Selected ) 
                                     -np . log ( Subset_Stock_Selected . shift (1) )) ),
                                   RFREE),
                      columns = stock_names)
df_stocks = df_stocks.iloc[1:,]
df_stocks.index = t

del Subset_Stock_Selected, RFREE,Interest,col

"""
Here in the argument title the content of the string: "sheet" is added
"""

rf.plotscatter(df_factors,df_stocks,"Excess Returns vs Eurostoxx -"+sheet,
            "Market Excess Returns",
            "",sheet,
            "Excess_return"
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

"""
P-values analysis
"""

rf.plotbar(CAPM_summary['p-value_alpha'])

rf.plotbar(CAPM_summary['p-value_beta: Market'])


"""
THIS USES THE PREVIOUSLY DEFINED CODE TO PLOT THE OLS FITTED LINE IN THE 
SCATTER PLOTS
"""

rf.plotCAPM(df_stocks,
            df_factors,
            stock_names,
            CAPM_summary.loc[CAPM_summary.index != 'Mean'],
            sheet)

"""
Computes excess returns of a equally-weighted portfolio for each month
"""
array_stocks = np.array(df_stocks.T)
Excess_equi_valued = np.array(sum(array_stocks)/len(array_stocks)).reshape(-1,1)

df_portfolios = pd.DataFrame(data = Excess_equi_valued, 
                            columns = ['Portfolio - EW'],
                            index = t)
del Excess_equi_valued

"""
Runs a CAPM using as independent variable the excess return of said portfolio
"""

CAPM_EW_Portfolio_Summary, CAPM_EW_Portfolio = rf.OLS(
                             df_portfolios['Portfolio - EW'],
                             df_factors['Market'])

"""
Comparison between:
    1) the average value of the parameters obtained for the single stocks
    2) the value of the parameters obtained using the equally-weighted portfolio
"""

comparison_stocks_EW_portfolio = pd.concat([CAPM_summary.loc['Mean',:], 
                        CAPM_EW_Portfolio_Summary.iloc[0,:]],
                       axis = 1).T

rf.comparison_scatter_2(df_stocks,df_portfolios['Portfolio - EW'],
            df_factors['Market'],
            CAPM_EW_Portfolio_Summary,
            "Excess Returns vs Eurostoxx -"+sheet,
            "Market Excess Returns",
            "Excess Returns",sheet,
            "Comparison_Scatter"
            )

rf.comparison_scatter(df_stocks,df_portfolios['Portfolio - EW'],
            df_factors['Market'],
            "Comparison between the scatter plots",
            "Market Excess Returns(" +sheet+")",
            "Excess Returns(" +sheet+")",sheet,
            "Comparison_Scatter"
            )

