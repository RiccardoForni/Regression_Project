"""
i=0
for e in np.array(Subset_Clear).T:
    str=Subset_Clear.columns[i].replace(" - TOT RETURN IND","")
    plt.figure()
    plt.plot(t , (e[1:n]/e[1]))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt . xlabel ('Time - Monthly - 30-09-2013 - 30-09-2023 ')
    plt . ylabel (str+' Monthly Total Ret')
    plt.savefig("img/Total_Return/TR-"+str+".png")
    if i == 15 :
        plt.close('all')
    i+=1
plt.close('all')
"""

"""
beta = []
alpha = []
Euro1D = ArrayEuroS.squeeze()
for Stock in result :
    b_,a_= np.polyfit(Euro1D[1:n],Stock[1:n], 1)
    beta.append(b_)
    print(b_)
    alpha.append(a_)
    
beta = np.array(beta).squeeze()
alpha = np.array(alpha).squeeze()
"""

"""
i = 0
for e in result:

    str=Subset_Clear.columns[i].replace(" - TOT RETURN IND","")
    plt.figure()
    plt.plot(ArrayEuroS[1:n], beta[i]*ArrayEuroS[1:n]+alpha[i])
    plt.scatter(ArrayEuroS[1:n],e[1:n])
    plt . xlabel ('Eurostoxx')
    plt . ylabel (str)
    plt.savefig("img/testCAPM/CAPM-"+str+".png")
    i+=1

plt.close('all')
"""