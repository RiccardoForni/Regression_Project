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



sheets={"EURIBOR_3_M":"BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ"}
#, "BUND":"RF GERMANY GVT BMK BID YLD 3M - RED. YIELD"
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
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    4. CAPM ESTIMATION
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    """

    """
    Here we estimate the CAPM model for each stock.
        -CAPM_summary is a dataframe in which we stored the alphas, betas plus their
        respective p-values and the R-Squared of each estimation.
        -CAPM_list is a list containing the complete output of each linear regression
    """

    CAPM_summary, CAPM_list = rf.OLS(df_stocks,df_factors, hac = True)
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
    P-values analysis of the coefficients 
    """
    CAPM_summary = CAPM_summary.sort_values('p-value_alpha')
    rp.plotbar(CAPM_summary['p-value_alpha'],("3_p_value_plots"), 
               obj ='p-value of Alpha coefficient')

    CAPM_summary = CAPM_summary.sort_values('p-value_beta: Market')
    rp.plotbar(CAPM_summary['p-value_beta: Market'],"3_p_value_plots", 
               obj = "p-value of Beta coefficient")

    """
    F-TEST comparison among the stock sample
    """
            
    F_test_p_values = rf.f_test_retrieval(CAPM_list)

    F_test_p_values.loc['Mean'] = F_test_p_values.mean()
    F_test_p_values = F_test_p_values.sort_values('F-Test_p-value')

    rp.plotbar(F_test_p_values['F-Test_p-value'],"3_F_test_p_value_plots",
               obj = "p-value of the F-tests")
        

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
    Runs a CAPM using as independent variable the excess return of the above portfolio
    """

    CAPM_EW_Portfolio_Summary, CAPM_EW_Portfolio = rf.OLS(
                                df_portfolios,
                                df_factors, hac = True)

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
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    4. DIAGNOSTIC TESTS
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    """

    """
    RESET TEST
    """

    a = rf.RESET_test(CAPM_list)
    b = rf.RESET_test(CAPM_EW_Portfolio)
    diag_CAPM_RESET = pd.concat([a,b], axis = 0)

    diag_CAPM_RESET.loc['Mean'] = diag_CAPM_RESET.mean()
    diag_CAPM_RESET = diag_CAPM_RESET.sort_values('p-value')

    diag_CAPM_RESET.to_excel("4_RESET_test.xlsx")
    rp.plotbar(diag_CAPM_RESET['p-value'],"4_p_value_RESET",
               obj = 'p-value of the RESET test')



    """
    WHITE TEST
    """

    a = rf.h_test(CAPM_list)
    b = rf.h_test(CAPM_EW_Portfolio)
    diag_CAPM_het_WHITE = pd.concat([a,b], axis = 0)

    diag_CAPM_het_WHITE.loc['Mean'] = diag_CAPM_het_WHITE.mean()
    diag_CAPM_het_WHITE = diag_CAPM_het_WHITE.sort_values('p-value')

    diag_CAPM_het_WHITE.to_excel("4_WHITE_test.xlsx")
    rp.plotbar(diag_CAPM_het_WHITE['p-value'],"4_p_value_WHITE",
               obj = 'p-value of the White test')


    """
    DURBIN-WATSON TEST
    """

    a = rf.Durbin_Watson_test(CAPM_list)
    b = rf.Durbin_Watson_test(CAPM_EW_Portfolio)
    diag_CAPM_serialcor_DW = pd.concat([a,b], axis = 0)

    diag_CAPM_serialcor_DW.loc['Mean'] = diag_CAPM_serialcor_DW.mean()
    diag_CAPM_serialcor_DW = diag_CAPM_serialcor_DW.sort_values('Test-statistic')

    diag_CAPM_serialcor_DW.to_excel("4_DW_test.xlsx")
    
    """
    Plots where the null hypothesis is positive autocorrelation
        -The values for LBound and UBound are the critical values obtained by 
         tables for samples with 100 observations and one regressor.
    """
    rp.plotbar_DW(diag_CAPM_serialcor_DW['Test-statistic'],"4_p_value_DW",
            Lbound = 1.522, Ubound = 1.562,
            obj = 'p-value of the Durbin-Watson test',conf = 0.01)

    rp.plotbar_DW(diag_CAPM_serialcor_DW['Test-statistic'],"4_p_value_DW",
            Lbound = 1.654, Ubound = 1.694,
            obj = 'p-value of the Durbin-Watson test',conf = 0.05)
    
    
    """
    Plots where the null hypothesis is negative autocorrelation
    """
    
    rp.plotbar_DW(diag_CAPM_serialcor_DW['Test-statistic'],"4_p_value_DW",
            Lbound = 1.522, Ubound = 1.562,
            obj = 'p-value of the Durbin-Watson test',conf = 0.01, 
            pos_autocorr= False)

    rp.plotbar_DW(diag_CAPM_serialcor_DW['Test-statistic'],"4_p_value_DW",
            Lbound = 1.654, Ubound = 1.694,
            obj = 'p-value of the Durbin-Watson test',conf = 0.05,
            pos_autocorr = False)


    """
    BREUSCH-GODFREY TEST
    """
    for i in range(3):
        a = rf.Breusch_Godfrey_test(CAPM_list, n = i +1)
        b = rf.Breusch_Godfrey_test(CAPM_EW_Portfolio, n = i+1)
        diag_CAPM_serialcor_BG = pd.concat([a,b], axis = 0)
    
        diag_CAPM_serialcor_BG.loc['Mean'] = diag_CAPM_serialcor_BG.mean()
        diag_CAPM_serialcor_BG = diag_CAPM_serialcor_BG.sort_values('p-value')
    
        diag_CAPM_serialcor_BG.to_excel("4_DW_test.xlsx")
        rp.plotbar(diag_CAPM_serialcor_BG['p-value'],"4_p_value_BG",
                   obj = 'p-value of the Bresuch-Godfrey test for {} lags'.format(i + 1))
        
    """
    Statistical significance of parameters using robust standard errors vs 
    non-robust standard errors
    """
    
    #Variable useful for better input window
    separator = "=" * 60
    
    CAPM_summary_stder, CAPM_list_stder = rf.OLS(df_stocks,df_factors)
    CAPM_summary_stder.loc['Mean'] = CAPM_summary_stder.mean()

    CAPM_summary_stder = CAPM_summary_stder.sort_index()
    CAPM_summary = CAPM_summary.sort_index()
    
    print(separator)
    print('\nComparison between the significance of parameters between the CAPM'+
          ' using robust standard errors vs non-robust standard errors\n')
    print(separator)
    
    rp.comparison_barplot(CAPM_summary_stder, CAPM_summary, label1 = 'Non-robust standard erorrs',
                          label2 = 'Robust standard erorrs')


    """
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    5. FAMA-FRENCH MODEL
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    """

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
    FF_summary, FF_list = rf.OLS(df_stocks,df_factors, hac = True)
    FF_summary.loc['Mean'] = FF_summary.mean()

    FF_summary = FF_summary.sort_values('p-value_alpha')
    FF_summary.to_excel("5_FF_stocks.xlsx")

    """
    Comparison with barplot between the CAPM and the Fama-French
    """

    CAPM_summary = CAPM_summary.sort_index()
    FF_summary = FF_summary.sort_index()
    
    print(separator)
    print("\nComparison between CAPM and FAMA-FRENCH with 5 factors\n")
    print(separator)
    
    rp.comparison_barplot(FF_summary, CAPM_summary)

    """
    Check for multicollinearity
    """

    factor_corr_matrix = df_factors.corr()



    """
    Running the fama-french model on the equally-weighted portfolio
    """

    FF_summary_port, FF_list_port = rf.OLS(df_portfolios,df_factors, hac = True)
    GETS_summary_port, bic_list = rf.GETS_ABIC(FF_summary_port, df_factors, df_portfolios,'b')
    GETS_summary_port = GETS_summary_port.sort_index()

    print(separator)
    print("\nComparison between CAPM and FAMA-FRENCH with 5 factors for the equally-weighted portfolio")
    print("\nIrrelevant variables removed by comparison between average value of BIC\n")
    print(separator)
    
    rp.comparison_barplot(GETS_summary_port, CAPM_summary)

    """
    Ad-hoc GETS
    """
    
    """
    GETS_ad_hoc_summary = rf.ad_hoc_GETS(FF_summary, df_factors, df_stocks)
    """
        
    res = rf.ad_hoc_GETS(FF_summary, df_factors, df_stocks)

    GETS_ad_hoc_summary= pd.DataFrame( columns = FF_summary.columns)

    for i in range(len(res)):
        GETS_ad_hoc_summary= pd.concat([GETS_ad_hoc_summary, res[i]], axis = 0)

    print(separator)
    print("\nComparison between CAPM and FAMA-FRENCH with 5 factors")
    print("\nIrrelevant variables removed by comparison between value of BIC for each stock\n")
    print(separator)
    
    rp.comparison_barplot(GETS_ad_hoc_summary, CAPM_summary.loc[CAPM_summary.index != 'Mean', :])
  
    
    """
    Plot to show which factors are relevant for each stock
    """
    
    rp.factor_plot(GETS_ad_hoc_summary, df_factors)
         
        
    """
    Comparison between General Fama-French and Specific Fama-French
    """    
    
    print(separator)
    print("\nComparison between the 6 factors Fama-French and Fama-French resulting from GETS procedure")
    print(separator)
    
    rp.comparison_barplot(GETS_ad_hoc_summary, FF_summary.loc[FF_summary.index != 'Mean',:], 
                          label1 = 'Specific Fama-French',
                          label2 = 'General Fama-French',
                          legend_pos= 'lower center')
    
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
    
    """
    Finding the p-values for each break date starting with the estimation using:
        
        1st subsample composed of the first 20% of the sample
        2nd subsample composed of the last 80% of the sample
        
    Then increasing the size of the first subsample by the subsequent observation and
    dropping that observation from the 2nd subsample.
    """
    
    p_val_df = rf.CHOW_TEST(df_stocks, df_factors)
    
    """    
    Plots of the obtained p-values
    """
    
    rp.chow_test_plotting(p_val_df, model = 'CAPM')

    """
    Finding all the possible breaks in the relationship between the dependent variable
    and the covariates
    """
    
    d4 = rf.CAPM_break_dates(p_val_df, CAPM_summary ,df_stocks, df_factors)


    """
    Comparing the break dates obtained for the CAPM
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


    df_bd_CAPM = pd.DataFrame(l, columns=['Date', 'Name'])
    df_bd_CAPM = df_bd_CAPM.sort_values('Date')



    dates_CAPM, names = zip(*l)

    rp.fama_french_plotting(df_bd_CAPM, model = 'CAPM')
    
      


    """
    FAMA FRENCH
    """
    
    """
    Obtaining the p-values for the Chow test in the same manner as those obtained
    for the CAPM model
    """

    p_val_df_FF = rf.CHOW_TEST_FF(df_stocks, df_factors)

    """    
    Plots
    """
    rp.chow_test_plotting(p_val_df_FF, model = 'Fama-French')

    
    """
    THIS WILL YIELD ALL THE MODELS WITHOUT IRRELEVANT VARIABLES FOR EACH STOCK AND FOR
    EACH BREAK DATE
    """

    d =  rf.break_dates_optimization(p_val_df_FF, FF_summary, 
                                df_stocks, df_factors)

    

    """
    COMPARISON BETWEEN THE AVERAGE VALUES OF R-SQUARED:
        -taking the resulting model for each break and average the value of the R-Squared
        -comparing it with the R-Squared value obtained, for the stock, with the CAPM
    """

    df_final = pd.DataFrame(index = df_stocks.columns, columns = FF_summary.columns)
    for i in d.keys():
        
        df_final.loc[i,:] = list(d[i].iloc[:,:-2].mean())
        
    df_final.sort_index(inplace=True)
    CAPM_summary.sort_index(inplace = True)

    print(separator)
    print("\nComparison between the CAPM and FAMA-FRENCH with 5 factors")
    print("\nIrrelevant variables removed by comparison between value of BIC for each stock.", end ='')
    print("\nThis is done for each of intervals determined by the breaks in the parameters' values.")
    print("\nSelect R-SQUARED to have the comparison with the average value of the R-SQUARED for each stock\n")
    print(separator)    
    
    rp.comparison_barplot(df_final, CAPM_summary.loc[CAPM_summary.index != 'Mean', :])

    """
    Maximum localized improvement of the R-Squared:
        -taking the resulting model for each break and taking the maximum value of the 
         R-Squared among them.
        -comparing it with the R-Squared value obtained, for the stock, with the CAPM
    """

    df_final2 = pd.DataFrame(index = df_stocks.columns, columns = FF_summary.columns)
    for i in d.keys():
        
        d[i] = d[i].sort_values('R-Squared', ascending = False)
        df_final2.loc[i,:] = list(d[i].iloc[0,:-2])
        
    df_final2.sort_index(inplace=True)
    CAPM_summary.sort_index(inplace = True)

    print(separator)
    print("\nComparison between the CAPM and FAMA-FRENCH with 5 factors")
    print("\nIrrelevant variables removed by comparison between value of BIC for each stock", end = '')
    print("\nThis is done for each of intervals determined by the breaks in the parameters' values")
    print("\nSelect R-SQUARED to have the comparison with the maximum value of the R-SQUARED for each stock\n")
    print(separator)
    
    rp.comparison_barplot(df_final2, CAPM_summary.loc[CAPM_summary.index != 'Mean', :])

    """
    COMPARISON MINIMUM VALUES OF THE P-VALUE OF ALPHA COEFFICIENTS:
        -taking the resulting model for each break and taking the minimum value of the 
         p-value of the Alpha coefficient among them.
        -comparing it with the p-value of the Alpha coefficient obtained, for the stock, 
        with the CAPM
    """
    df_final2 = pd.DataFrame(index = df_stocks.columns, columns = FF_summary.columns)
    for i in d.keys():
        
        d[i] = d[i].sort_values('p-value_alpha')
        df_final2.loc[i,:] = list(d[i].iloc[0,:-2])
        
    df_final2.sort_index(inplace=True)
    CAPM_summary.sort_index(inplace = True)

    print(separator)
    print("\nComparison between the CAPM and FAMA-FRENCH with 5 factors")
    print("\nIrrelevant variables removed by comparison between value of BIC for each stock", end = '')
    print("\nThis is done for each of intervals determined by the breaks in the parameters' values")
    print("\nSelect p-value of the Alpha to have the comparison with the minimum value of the p-value of the alpha parameter for each stock\n")
    print(separator)
    
    rp.comparison_barplot(df_final2, CAPM_summary.loc[CAPM_summary.index != 'Mean', :])

    import matplotlib.pyplot as plt
    
    #Getting the ones for which there is no improvement
    l_no_improvement = []
    for i in df_final2.index:
        
        if df_final2.loc[i, 'R-Squared'] == CAPM_summary.loc[i, 'R-Squared']:
            l_no_improvement.append(i)
        
    df_to_plot = pd.DataFrame(columns = d[l_no_improvement[0]].columns, index = l_no_improvement)
    
    #Plotting the number of models
    n_models = pd.DataFrame(index = d.keys(), columns = ['N_models'])
    for i in d.keys():
        n_models.loc[i,:] = int(d[i].shape[0])
        
    plt.plot()
    
    plt.bar(n_models.index, n_models['N_models'])
    plt.xticks(rotation=90)
    plt.title('Number of models')
    
    plt.show()
       
    #Show the factors for those models
    for i in l_no_improvement:
        
        df_to_plot.loc[i,:] = d[i].values
        
        
    rp.factor_plot(df_to_plot, df_factors)

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

    rp.fama_french_plotting(df, model = 'Fama-French')


    """
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    7. CAPM ROLLING WINDOW ESTIMATION
    -----------------------------------------------------------------------------------------
    -----------------------------------------------------------------------------------------
    """



    """
    CAPM
    """
    dictCapmRolling= rf.CAPM_in_rolling_windows(df_stocks,df_factors,df_bd_CAPM,CAPM_summary)
    for elem in dictCapmRolling:
        rp.plotting_CAPM_7(dictCapmRolling[elem[0]],dictCapmRolling[elem[1]],
                           dictCapmRolling[elem[2]],dictCapmRolling[elem[3]], dictCapmRolling[elem[4]])


    rp.plotEstimator(rf.ad_hoc_fun(GETS_ad_hoc_summary,df_factors))

    """
    -------------------------------------------------------------------------------------
    COMPARISON OF RESIDUALS CAPM VS FF
    -------------------------------------------------------------------------------------
    """

    """
    Correlation among residuals for the CAPM model
    """
    resid_autocorr_CAPM=rf.resid_autocorr_calculator(df_stocks,CAPM_list)

    """
    Correlation among residuals for the Fama-French model with 6 factors
    """
    resid_autocorr_FF=rf.resid_autocorr_calculator(df_stocks,FF_list)
    


    print(separator)
    print("""\nComparison between the correlation among the residuals and their\n
        lagged values between the CAPM model and the Fama-French model\n""")
    print(separator)

    rp.comparison_barplot(resid_autocorr_FF, resid_autocorr_CAPM)
    
    rp.histoplotting(resid_autocorr_CAPM,resid_autocorr_FF,n=50)
