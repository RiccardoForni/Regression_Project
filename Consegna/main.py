import numpy as np
import pandas as pd
import Regression_function as rf
import Regression_Plotting as rz


rp = rz.Plotting(True)
"""
Retrieve Subset equity
"""
EuroStoxx = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EUROSTOXX600")
Subset_Stock_Selected= pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset")
"""
Creating time-series from data
"""
time_series=pd . date_range ( start =EuroStoxx.iloc[1,0].strftime("%d-%m-%Y"),end ='30-09-2023 ', freq ='M') #Date series
"""
Clean the data 
"""
EuroStoxx = EuroStoxx.iloc[: , 1:]
Subset_Stock_Selected = Subset_Stock_Selected.iloc[: , 1:]
Subset_Stock_Selected.columns = Subset_Stock_Selected.columns.str.replace(" - TOT RETURN IND","")



sheets={"EURIBOR_3_M":"BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ","BUND":"RF GERMANY GVT BMK BID YLD 3M - RED. YIELD"}
for sheet,col in sheets.items():  
    print("eseguo:" +sheet)

    """
    Retrieve Subset Interest rate
    """
    Interest = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name=sheet)

    """
    Obtain monthly rates from annualized interest rates time-series
    """
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

    l = rp.plotscatter(df_factors,df_stocks,"Excess Returns vs Eurostoxx -"+sheet,
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

    rp.m_scatter(CAPM_summary, df_factors, df_stocks,
                ("2_scatter_comparison","2_scatter_comparison"))

    """
    P-values analysis
    """
    CAPM_summary = CAPM_summary.sort_values('p-value_alpha')
    rp.plotbar(CAPM_summary['p-value_alpha'],("3_p_value_plots"))

    CAPM_summary = CAPM_summary.sort_values('p-value_beta: Market')
    rp.plotbar(CAPM_summary['p-value_beta: Market'],"3_p_value_plots")

    """
    F-TEST Comparison
    """

            
    F_test_p_values = rf.f_test_retrieval(CAPM_list)

    F_test_p_values.loc['Mean'] = F_test_p_values.mean()
    F_test_p_values = F_test_p_values.sort_values('F-Test_p-value')

    rp.plotbar(F_test_p_values['F-Test_p-value'],"3_F_test_p_value_plots")
        

    """
    THIS USES THE PREVIOUSLY DEFINED CODE TO PLOT THE OLS FITTED LINE IN THE 
    SCATTER PLOTS
    """

    rp.plotCAPM(df_stocks,
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


    rp.comparison_scatter(df_stocks,df_portfolios['Portfolio - EW'],
                df_factors['Market'],
                "Excess Returns vs Eurostoxx -"+sheet,
                "Market Excess Returns",
                "Excess Returns",
                ("3_Comparison_Scatter",sheet),
                CAPM_EW_Portfolio_Summary
                )

    rp.comparison_scatter(df_stocks,df_portfolios['Portfolio - EW'],
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
    rp.plotbar(diag_CAPM_RESET['p-value'],"4_p_value_RESET")



    """
    WHITE TEST
    """

    a = rf.h_test(CAPM_list)
    b = rf.h_test(CAPM_EW_Portfolio)
    diag_CAPM_het_WHITE = pd.concat([a,b], axis = 0)

    diag_CAPM_het_WHITE.loc['Mean'] = diag_CAPM_het_WHITE.mean()
    diag_CAPM_het_WHITE = diag_CAPM_het_WHITE.sort_values('p-value')

    diag_CAPM_het_WHITE.to_excel("4_WHITE_test.xlsx")
    rp.plotbar(diag_CAPM_het_WHITE['p-value'],"4_p_value_WHITE")


    """
    DURBIN-WATSON TEST
    """

    a = rf.Durbin_Watson_test(CAPM_list)
    b = rf.Durbin_Watson_test(CAPM_EW_Portfolio)
    diag_CAPM_serialcor_DW = pd.concat([a,b], axis = 0)

    diag_CAPM_serialcor_DW.loc['Mean'] = diag_CAPM_serialcor_DW.mean()
    diag_CAPM_serialcor_DW = diag_CAPM_serialcor_DW.sort_values('Test-statistic')

    diag_CAPM_serialcor_DW.to_excel("4_DW_test.xlsx")
    rp.plotbar(diag_CAPM_serialcor_DW['Test-statistic'],"4_p_value_DW",
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
    rp.plotbar(diag_CAPM_serialcor_BG['p-value'],"4_p_value_BG")

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

    rp.comparison_barplot(FF_summary, CAPM_summary)

    """
    Check for multicollinearity
    """

    factor_corr_matrix = df_factors.corr()

    """
    GETS Procedure using bic as a criterion to discriminate between models
    """

    GETS_summary, bic_list = rf.GETS_ABIC(FF_summary, df_factors, df_stocks,'c')

    GETS_summary = GETS_summary.sort_index()
    rp.comparison_barplot(GETS_summary, CAPM_summary)

    """
    ELIMINATING THE CORRELATED COVARIATES FIRST THEN DOING THE GETS PROCEDURE USING
    BIC AS THE INFORMATION CRITERION
    """
    l = ['MOM', 'HML']

    df_factors_2 =df_factors.drop(l, axis = 1)

    FF_summary_2, FF_list = rf.OLS(df_stocks,df_factors_2)
    FF_summary_2.loc['Mean'] = FF_summary_2.mean()

    GETS_summary2, bic_list2 = rf.GETS_ABIC(FF_summary_2, df_factors_2, df_stocks,'b')

    GETS_summary2 = GETS_summary2.sort_index()
    #rp.comparison_barplot(GETS_summary2, CAPM_summary)

    """
    GETS PROCEDURE USING THE AIC AS A CRITERION TO DISCRIMINATE BETWEEN MODELS
    """

    GETS_summary_aic, aic_list = rf.GETS_ABIC(FF_summary, df_factors, df_stocks,'a')

    """
    Running the fama-french model on the equally-weighted portfolio
    """

    FF_summary_port, FF_list_port = rf.OLS(df_portfolios,df_factors)
    GETS_summary_port, bic_list = rf.GETS_ABIC(FF_summary_port, df_factors, df_portfolios,'b')
    GETS_summary_port = GETS_summary_port.sort_index()

    rp.comparison_barplot(GETS_summary_port, CAPM_summary)

    """
    Ad-hoc GETS
    """
    """
    GETS_ad_hoc_summary = rf.ad_hoc_GETS(FF_summary, df_factors, df_stocks)
    """
        
    res = rf.ad_hoc_GETS(FF_summary, df_factors, df_stocks)

    df = pd.DataFrame( columns = FF_summary.columns)

    for i in range(len(res)):
        df = pd.concat([df, res[i]], axis = 0)

    rp.comparison_barplot(df, CAPM_summary.loc[CAPM_summary.index != 'Mean', :])

           
        
    """
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    6. CHOW TEST
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    """

    """
    CAPM
    """

    p_val_df = rf.CHOW_TEST(df_stocks, df_factors)

    """    
    Plots
    """


    
    d4 = rf.CAPM_break_dates(p_val_df, CAPM_summary ,df_stocks, df_factors)


    """
    COMPARING THE BREAK dates_CAPM
    """

    dates_CAPM = pd.DataFrame(columns = d4.keys())

    for i in d4.keys():
            
        if d4[i].shape[0] > 1:
            
            for j in range(d4[i].shape[0]-1):
                dates_CAPM.loc[j,i] = d4[i].loc[j, 'end_date']
                
    dates_CAPM = dates_CAPM.fillna(np.nan).replace([np.nan], [None])

    l = []

    for i in range(dates_CAPM.values.shape[0]):
        
        for j in range(dates_CAPM.values.shape[1]):
            
            if dates_CAPM.values[i][j] != None:
        
                l.append((dates_CAPM.values[i][j], dates_CAPM.columns[j]))

    from matplotlib.dates import date2num

    df_bd_CAPM = pd.DataFrame(l, columns=['Date', 'Name'])
    df_bd_CAPM = df_bd_CAPM.sort_values('Date')



    dates_CAPM, names = zip(*l)
    """
    Plots
    """
    rp.fama_french_plotting(df_bd_CAPM)
    
        


    """
    CHOW TEST FAMA FRENCH
    """

    p_val_df_FF = rf.CHOW_TEST_FF(df_stocks, df_factors)

    """    
    Plots
    """
    rp.chow_test_plotting(p_val_df)

    
    """
    THIS WILL YIELD ALL THE MODELS WITHOUT IRRELEVANT VARIABLES FOR EACH STOCK AND FOR
    EACH BREAK DATE
    """

    d =  rf.break_dates_optimization(p_val_df_FF, FF_summary, 
                                df_stocks, df_factors)

    

    """
    COMPARISON BETWEEN THE AVERAGE VALUES OF R-SQUARED
    """

    df_final = pd.DataFrame(index = df_stocks.columns, columns = FF_summary.columns)
    for i in d.keys():
        
        df_final.loc[i,:] = list(d[i].iloc[:,:-2].mean())
        
    df_final.sort_index(inplace=True)
    CAPM_summary.sort_index(inplace = True)

            
    rp.comparison_barplot(df_final, CAPM_summary.loc[CAPM_summary.index != 'Mean', :])

    """
    Maximum localized improvement of the R-Squared
    """

    df_final2 = pd.DataFrame(index = df_stocks.columns, columns = FF_summary.columns)
    for i in d.keys():
        
        d[i] = d[i].sort_values('R-Squared', ascending = False)
        df_final2.loc[i,:] = list(d[i].iloc[0,:-2])
        
    df_final2.sort_index(inplace=True)
    CAPM_summary.sort_index(inplace = True)


    rp.comparison_barplot(df_final2, CAPM_summary.loc[CAPM_summary.index != 'Mean', :])

    """
    COMPARISON MAX VALUES OF THE P-VALUE OF ALPHA COEFFICIENTS
    """
    df_final2 = pd.DataFrame(index = df_stocks.columns, columns = FF_summary.columns)
    for i in d.keys():
        
        d[i] = d[i].sort_values('p-value_alpha')
        df_final2.loc[i,:] = list(d[i].iloc[0,:-2])
        
    df_final2.sort_index(inplace=True)
    CAPM_summary.sort_index(inplace = True)


    rp.comparison_barplot(df_final2, CAPM_summary.loc[CAPM_summary.index != 'Mean', :])


    """
    COMPARING THE BREAK DATES FOR THE FAMA FRENCH MODEL
    """

    dates = pd.DataFrame(columns = d.keys())

    for i in d.keys():
            
        if d[i].shape[0] > 1:
            
            for j in range(d[i].shape[0]-1):
                dates.loc[j,i] = d[i].loc[j, 'end_date']
                
    dates = dates.fillna(np.nan).replace([np.nan], [None])

    l = []

    for i in range(dates.values.shape[0]):
        
        for j in range(dates.values.shape[1]):
            
            if dates.values[i][j] != None:
        
                l.append((dates.values[i][j], dates.columns[j]))



    df = pd.DataFrame(l, columns=['Date', 'Name'])
    df = df.sort_values('Date')



    dates, names = zip(*l)

    rp.fama_french_plotting(df)


    """
    ---------------------------------------------------------------------------------------
    PUNTO 7
    --------------------------------------------------------------------------------------
    """


    """
    CAPM
    """


    beg = int(0)
    end = int(21)

    stop = df_stocks.shape[0]

    d3 = {}

    l_col = []
    l_col = l_col + list(CAPM_summary.columns) 

    """
    CREATING THE LIST OF PARAMETERS FOR WHICH WE WANT THE PLOT WITH CONFIDENCE INTERVAL
    """

    l_conf = ['Alpha','Market']

    for i in l_conf:

        l_col = l_col+ [i+ '_LBound', i+ '_UBound'] 

    l_col = l_col + ['end_date']

    for i in df_stocks.columns:
        
        d3[i] = pd.DataFrame(columns = l_col)

    """
    ESTIMATING THE CAPM MODELS FOR EACH STOCK WITH A ROLLING WINDOW OF 5 YEARS
    """

    j = 0
        
    while end <= stop:

        roll_df_stocks = df_stocks.iloc[beg:end, :]
        
        roll_df_factors = df_factors.iloc[beg:end, :]
        
        
        roll_CAPM_summary, roll_CAPM_list = rf.OLS(roll_df_stocks,roll_df_factors['Market'], 
                                                hac = True,
                                                conf_int = [True, l_conf])
        
        for i in d3.keys():
            
            d3[i] = pd.concat([d3[i],roll_CAPM_summary.loc[i,:].to_frame().T],
                            ignore_index= True)
            d3[i].iloc[j,-1] = roll_df_stocks.index[-1]
            
            
        beg += 1
        end += 1   
        j += 1      

    for i in d3.keys():
        
        d3[i] = d3[i].set_index('end_date')

    """
    CHECK TO SEE THAT THE CONFIDENCE INTERVALS ARE SYMMETRIC
    """    
        
    for i in d3.keys():
        
        for j in l_conf: 
            t = (d3[i]['beta: Market'] - d3[i][j+ '_LBound']) + (d3[i]['beta: Market'] - d3[i][j+'_UBound'])
            check = sum(t)
            if i == 'ASML HOLDING':
                check_2 = t
            
            
        
    """
    PLOT OF PARAMETERS THAT ADMIT CONFIDENCE INTERVALS
    """

    df_bd_CAPM_2 = df_bd_CAPM.set_index('Name')

    list_to_plot = list(set(df_bd_CAPM_2.index))
    rp.plotting_CAPM_7(list_to_plot,d3,df_bd_CAPM_2,l_conf)
