import numpy as np
import pandas as pd
import Riccardo_Official_Experiment_Function as rf
import Riccardo_Official_Experiment_Plotting as rz
import sys

rp = rz.Plotting(True if sys.argv[1] == 'T' else False)
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

    #rp.comparison_barplot(FF_summary, CAPM_summary)

    """
    Check for multicollinearity
    """

    factor_corr_matrix = df_factors.corr()

    """
    GETS Procedure using bic as a criterion to discriminate between models
    """

    GETS_summary, bic_list = rf.GETS_ABIC(FF_summary, df_factors, df_stocks,'b')

    GETS_summary = GETS_summary.sort_index()
    #rp.comparison_barplot(GETS_summary, CAPM_summary)

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

    rf.comparison_barplot(df, CAPM_summary.loc[CAPM_summary.index != 'Mean', :])

    # %% CHOW test               
    """
    6. CHOW TEST
    """
    import pdb; pdb.set_trace()
    p_val_df = rf.CHOW_TEST(df_stocks, df_factors)
    rp.shish(p_val_df)
    rp.shish2(p_val_df)

    l_check = []

    for i in p_val_df.columns:
        
        l_check.append(max(p_val_df.loc[:,i]))




