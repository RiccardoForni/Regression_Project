import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""import scipy as sp"""
Data = []
t=pd . date_range ( start ='15-09-2013 ',end ='15-09-2023 ', freq ='M') #Date series

"""
Retrieve Subset equity and Interest rate
"""
Subset= pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="Subset")
Interest = pd . read_excel('DataEuroStock_Tecnology.xlsx',sheet_name="EURIBOR_3_M")


"""
Calculate interest rate
"""
RFREE=Interest[["BD INTEREST RATES - EURIBOR RATE - 3 MONTH NADJ"]]/12
RF = np.array(RFREE)

"""
Clear from Date
and
calculate Log operation 
"""
Subset_Clear=Subset.loc[:, Subset.columns != 'Name'] #Delete column of date
Subset_Log_Operation = 100 *( np . log ( Subset_Clear ) -np . log ( Subset_Clear . shift (1) )) #Log operation

"""
Retrieve equity 
"""
for e in Subset_Log_Operation:
    Data.append(Subset_Log_Operation.loc[:, Subset_Log_Operation.columns == e]) # Retrieve equity

Equity = np.array(Data) #Convert Data da simple array an numpy array for subtract operation


"""
Calculate for each equity: Equity-interest rate

The result is a matrix [Number of equity][Value calculated]
"""
result = []
for E in Equity:
    result.append(np.subtract(E,RFREE)) #Subtract operation filled inside an array 

n=np . size ( Equity[0]) #Constant dimension
i=0
for e in np.array(Subset_Clear).T:
    str=Subset_Clear.columns[i].replace(" - TOT RETURN IND","")
    plt.figure()
    plt.plot(t , e[1:n],t,RF[1:n])
    plt . xlabel ('Time - Monthly - 30-09-2013 - 30-09-2023 ')
    plt . ylabel (str+' Monthly Total Ret and Risk Free')
    plt.savefig("img/Total_Return/TR-"+str+".png")
    if i == 20 :
        plt.close()
    i+=1
plt.close()
i=0
for e in result:
    str=Subset_Clear.columns[i].replace(" - TOT RETURN IND","")
    plt.figure()
    plt.plot(t , e[1:n],)
    plt . xlabel ('Time - Monthly - 30-09-2013 - 30-09-2023 ')
    plt . ylabel (str+' Excess Returns')
    plt.savefig("img/Excess_Return/ER-"+str+".png")
    if i == 20 :
        plt.close()
    i+=1
plt.close()
    


