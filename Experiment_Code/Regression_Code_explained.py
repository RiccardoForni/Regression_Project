import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import statsmodels . api as sm
import os

"""
- setx = market return
- set y = stock returns
- title = title of the graph
- xlabel = name of the x-axis
- ylabel = name of the y-axis
- sigla = unique name (either ER for euribor or ERB for bunds) to differentiate
          between plots of excess returns of stocks computed with Euribor
          or with the bund yield.
- Subset = dataframe with the data of total index returns
- string_to_save = name of the folder in which the graph will be saved in the 
                   working directory
"""

def plotscatter(setx,sety,title,xlabel,ylabel,sigla,Subset,string_to_save):
    
    #Creates a new folder in which the plots will be saved
    cwd = os.getcwd()
    folder = cwd + "/"+string_to_save
    
    #Checks whether the folder already exists 
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    #Creates a map object called my_int
    """
    My explanation is that it's an object containing all the names of the stocks
    which using the command:next(myint) goes through all the names
    """
    myint=iter(Subset.columns)
    for e in sety:
        str=next(myint)
        plt.figure()
        plt.scatter(setx,e)
        plt.title(title)
        plt . xlabel (xlabel)
        plt . ylabel (str+ylabel)
        plt.savefig(folder+"/"+sigla+"-"+str+".png")
        plt.close()
      
"""
- Stock_Risk_Free = excess stock returns
- Market = market returns 
- Subset = dataframe in whcih the data on total index returns are stored
"""



def OLS_Pvalue(Stock_Risk_Free,Market,Subset):
    Res = []
    P = {}
    #X is a matrix containing our explanatory variables: a constant and 
    #the market return
    X = np . column_stack (( np . ones_like ( Market ) , Market ))
    #Same object as the function before
    myint=iter(Subset.columns)
    for e in Stock_Risk_Free:
        #Stores in a list the object which contain all the results from the 
        #Linear regression
        Res.append(sm . OLS ( e[1:] , X[1:]  ). fit ())
        
        """
        P is a dictionary whose index will be given by the name of the stock
        and the value will be the p-value associated with the last object of
        the list, which is the regression done just the command before
        """
        
        P[next(myint)]=Res[-1].pvalues[0]
        """
        with open('summary'+str(i)+'.txt', 'w') as fh:
            fh.write(Res1[-1].summary().as_text())
        
        i+=1
        """
        
        #The sorted command rearranges the dictionary by using the values.
     
    return Res,sorted(P.items(), key=lambda x:x[1])



t=pd . date_range ( start ='15-09-2013 ',end ='15-09-2023 ', freq ='M') #Date series

"""
Retrieve Subset equity and Interest rate
"""
EuroStoxx = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EUROSTOXX600")
Subset_Stock_Selected= pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset")
Interest = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EURIBOR_3_M")
Interest_BUND = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="BUND")


#Makes the names without TOT RETURN INDEX
Subset_Stock_Selected.columns = Subset_Stock_Selected.columns.str.replace("- TOT RETURN IND","")

#Delete the time columns from the dataframes containing the stock and market returns

EuroStoxx = EuroStoxx.loc[:, EuroStoxx.columns != 'Name']#Delete column of date
Subset_Stock_Selected=Subset_Stock_Selected.loc[:, Subset_Stock_Selected.columns != 'Name'] #Delete column of date


#Annualize the riskfree interest rates
RFREE=np.array(Interest[["BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ"]]/12)
BUNDRISK=np.array(Interest_BUND[['RF GERMANY GVT BMK BID YLD 3M - RED. YIELD']]/12)
"""
Clear from Date
and
calculate Log operation 
"""
#Instead of dataframes, the monthly returns, without time are stored in
#an array object

Market =  np.array(100 *( np . log ( EuroStoxx ) -np . log ( EuroStoxx . shift (1) )) )
Equities = np.array(100 *( np . log ( Subset_Stock_Selected ) -np . log ( Subset_Stock_Selected . shift (1) ))) #Log operation
"""
Calculate for each equity: Equity-interest rate

The result is a matrix [Number of equity][Value calculated]
"""
#HERE THE MONTHLY EXTRA-RETURNS OF STOCKS ARE COMPUTED

rStock = np.subtract(Equities,RFREE).T #Subtract operation filled inside an array 
rStock_Bond =np.subtract(Equities,BUNDRISK).T

#Here the monthly extra-returns of the market are computed
#ERROR: We should not use it 

rMarket =np.subtract(Market,RFREE)
rMarket_Bond =np.subtract(Market,BUNDRISK)

#SEE UP IN THE FIRST FUNCTION

plotscatter(Market,rStock,"Excess Returns vs Eurostoxx - 3M Euribor",
            "Time - Monthly - 30-09-2013 - 30-09-2023",
            "","ER",Subset_Stock_Selected,
            "Excess_return"
            )

plotscatter(Market,rStock_Bond,"Excess Returns vs Eurostoxx - 3M BUND",
            "Time - Monthly - 30-09-2013 - 30-09-2023",
            "","ERB",Subset_Stock_Selected,
            "Excess_return"
            )

#Go back up

Res,D_sort = OLS_Pvalue(rStock,rMarket,Subset_Stock_Selected)
Res_bund,D_sort= OLS_Pvalue(rStock_Bond,rMarket,Subset_Stock_Selected)
for e in D_sort:
    plt.bar(e[0],e[1])
plt.savefig("TEST.png")



