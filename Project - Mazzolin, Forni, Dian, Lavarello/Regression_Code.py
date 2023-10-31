import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import statsmodels . api as sm
import os

def plotscatter(setx,sety,title,xlabel,ylabel,sigla,Subset,string_to_save,string_to_cancel):
    cwd = os.getcwd()
    folder = cwd + "/img/"+string_to_save+"/" 

    if not os.path.exists(folder):
        os.mkdir(folder)
    i=0
    for e in sety.T:
        str=Subset.columns[i].replace(string_to_cancel,"")
        plt.figure()
        plt.scatter(setx,e)
        plt.title(title)
        plt . xlabel (xlabel)
        plt . ylabel (str+ylabel)
        plt.savefig(folder+sigla+"-"+str+".png")
        plt.close()
        i+=1


def OLS_Pvalue(Stock_Risk_Free,Market,Subset):
    Res = []
    P = {}
    P_sort=[]
    X = np . column_stack (( np . ones_like ( Market ) , Market ))
    i=0
    for e in Stock_Risk_Free.T:
        Res.append(sm . OLS ( e[1:] , X[1:]  ). fit ())
        P.update({Res[-1].pvalues[0]:Subset.columns[i]})
        P_sort.append(Res[-1].pvalues[0])
        i+=1
        """
        with open('summary'+str(i)+'.txt', 'w') as fh:
            fh.write(Res1[-1].summary().as_text())
        
        i+=1
        """
    P_sort=np.sort(np.array(P_sort))
    P_value_ordered = []
    for e in P_sort:
        P_value_ordered.append([e,P[e]])


    return Res,P_value_ordered



t=pd . date_range ( start ='15-09-2013 ',end ='15-09-2023 ', freq ='M') #Date series

"""
Retrieve Subset equity and Interest rate
"""
EuroStoxx = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EUROSTOXX600")
Subset_Stock_Selected= pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset")
Interest = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EURIBOR_3_M")
Interest_BUND = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="BUND")

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
rStock = np.subtract(Equities,RFREE) #Subtract operation filled inside an array 
rStock_Bond =np.subtract(Equities,BUNDRISK)

rMarket =np.subtract(Market,RFREE)
rMarket_Bond =np.subtract(Market,BUNDRISK)
DIM_DATA=np . size ( Market.shape[0]) #Constant dimension
N_stock= rStock.shape[1]

plotscatter(Market,rStock,"Excess Returns vs Eurostoxx - 3M Euribor",
            "Time - Monthly - 30-09-2013 - 30-09-2023",
            "","ER",Subset_Stock_Selected,
            "Excess_return"," - TOT RETURN IND"
            )

plotscatter(Market,rStock_Bond,"Excess Returns vs Eurostoxx - 3M BUND",
            "Time - Monthly - 30-09-2013 - 30-09-2023",
            "","ERB",Subset_Stock_Selected,
            "Excess_return"," - TOT RETURN IND"
            )


Res,P_Value = OLS_Pvalue(rStock,rMarket,Subset_Stock_Selected)
Res_bund,P_Value_bund= OLS_Pvalue(rStock_Bond,rMarket,Subset_Stock_Selected)


for e,j in zip(P_Value,P_Value_bund):
    print(e[1]==j[1])

