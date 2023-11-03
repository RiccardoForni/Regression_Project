import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import statsmodels . api as sm
import os

def plotscatter(setx,sety,title,xlabel,ylabel,sigla,Subset,string_to_save):
    cwd = os.getcwd()
    folder = cwd + "/"+string_to_save

    if not os.path.exists(folder):
        os.mkdir(folder)
        
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
      


def OLS(Stock_Risk_Free,Market):
    Res= []
    X = np . column_stack (( np . ones_like ( Market ) , Market ))
    for e in Stock_Risk_Free:
        Res.append(pd.read_html(sm . OLS ( e[1:] , X[1:]  ). fit ().summary().tables[1].as_html(),header=0,index_col=0)[0])
        
        """
        
        with open('summary'+str(i)+'.txt', 'w') as fh:
            fh.write(Res1[-1].summary().as_html())
        
        i+=1
        """
    return Res

def ReorderByOLSParam(Stocks,Subset,Row_interess,Coloum_interess):
    """
    Function return stock reorder by OLS result
    Row_interess -> choose row of OLS Summary between
    Const=0, X1=1

    Coloum_interess-> choose coloum of OLS summary between
    coef=0      std err=1    t=2     P>|t|=3     0.025=4   0.975=5  
    """ 
    P = {}
    myint=iter(Subset.columns)
    for e in Stocks:
            P[next(myint)]=e.iloc[Row_interess][Coloum_interess]
    return sorted(P.items(), key=lambda x:x[1])

t=pd . date_range ( start ='15-09-2013 ',end ='15-09-2023 ', freq ='M') #Date series

"""
Retrieve Subset equity and Interest rate
"""
EuroStoxx = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EUROSTOXX600")
Subset_Stock_Selected= pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset")
Interest = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EURIBOR_3_M")
Interest_BUND = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="BUND")



Subset_Stock_Selected.columns = Subset_Stock_Selected.columns.str.replace("- TOT RETURN IND","")
EuroStoxx = EuroStoxx.loc[:, EuroStoxx.columns != 'Name']#Delete column of date
Subset_Stock_Selected=Subset_Stock_Selected.loc[:, Subset_Stock_Selected.columns != 'Name'] #Delete column of date

RFREE=np.array(Interest[["BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ"]]/12)
BUNDRISK=np.array(Interest_BUND[['RF GERMANY GVT BMK BID YLD 3M - RED. YIELD']]/12)
"""
Clear from Date
and
calculate Log operation 
"""
Market =  np.array(100 *( np . log ( EuroStoxx ) -np . log ( EuroStoxx . shift (1) )) )
Equities = np.array(100 *( np . log ( Subset_Stock_Selected ) -np . log ( Subset_Stock_Selected . shift (1) ))) #Log operation
"""
Calculate for each equity: Equity-interest rate

The result is a matrix [Number of equity][Value calculated]
"""
rStock = np.subtract(Equities,RFREE).T #Subtract operation filled inside an array 
rStock_Bond =np.subtract(Equities,BUNDRISK).T

rMarket =np.subtract(Market,RFREE)
rMarket_Bond =np.subtract(Market,BUNDRISK)
"""
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

"""
Res_Euribor= OLS(rStock,rMarket)
Res_Bund= OLS(rStock_Bond,rMarket)

P_sort=ReorderByOLSParam(Res_Euribor,Subset_Stock_Selected,0,3)

for e in P_sort:
    plt.bar(e[0],e[1])
    plt.xticks(rotation=90)
plt.savefig("TEST.png")
