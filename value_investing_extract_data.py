import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
# from format import format


def getelementinlist(list,element):
    try:
        return list[element]
    except:
        return '-'

def getfinancialreportingdf(ticker):

    urlfinancials = 'https://www.marketwatch.com/investing/stock/'+ticker+'/financials'
    urlbalancesheet = 'https://www.marketwatch.com/investing/stock/'+ticker+'/financials/balance-sheet'

    text_soup_financials = BeautifulSoup(requests.get(urlfinancials).text,"lxml")
    text_soup_balancesheet = BeautifulSoup(requests.get(urlbalancesheet).text,"lxml")

    # Income statement
    titlesfinancials = text_soup_financials.findAll('td', {'class': 'rowTitle'})
    epslist = []
    netincomelist = []
    longtermdebtlist = []
    interestexpenselist = []
    ebitdalist = []

    for title in titlesfinancials:
        if 'EPS (Basic)' in title.text:
            epslist.append ([td.text for td in title.findNextSiblings (attrs={'class': 'valueCell'}) if td.text])
        if 'Net Income' in title.text:
            netincomelist.append([td.text for td in title.findNextSiblings(attrs={'class': 'valueCell'}) if td.text])
        if 'Interest Expense' in title.text:
            interestexpenselist.append([td.text for td in title.findNextSiblings(attrs={'class': 'valueCell'}) if td.text])
        if 'EBITDA' in title.text:
            ebitdalist.append([td.text for td in title.findNextSiblings(attrs={'class': 'valueCell'}) if td.text])

    # Balance sheet
    titlesbalancesheet = text_soup_balancesheet.findAll('td', {'class': 'rowTitle'})
    equitylist = []

    for title in titlesbalancesheet:
        if 'Total Shareholders\' Equity' in title.text:
            equitylist.append ([td.text for td in title.findNextSiblings (attrs={'class': 'valueCell'}) if td.text])
        if 'Long-Term Debt' in title.text:
            longtermdebtlist.append([td.text for td in title.findNextSiblings(attrs={'class': 'valueCell'}) if td.text])

    eps = getelementinlist(epslist,0) #function(table,row)
    epsgrowth = getelementinlist(epslist,1)
    netincome = getelementinlist(netincomelist,0)
    shareholderequity = getelementinlist(equitylist,0)
    roa = getelementinlist(equitylist,1)

    longtermdebt = getelementinlist(longtermdebtlist,0)
    interestexpense = getelementinlist(interestexpenselist,0)
    ebitda = getelementinlist(ebitdalist,0)

    df = pd.DataFrame({'eps': eps, 'epsgrowth': epsgrowth, 'netincome': netincome, 'shareholderequity': shareholderequity, 'roa': roa, 'longtermdebt': longtermdebt, 'interestexpense': interestexpense, 'ebitda': ebitda}, index=[2015, 2016, 2017, 2018, 2019])

    return df


# Getting financial reporting df
def getfinancialreportingdfformatted(ticker):
    df = getfinancialreportingdf(ticker)
    # Format all the number in dataframe
    dfformatted = df.apply(format)

    # Adding roe, interest coverage ratio
    dfformatted['roe'] = dfformatted.netincome/dfformatted.shareholderequity
    dfformatted['interestcoverageratio'] = dfformatted.ebitda/dfformatted.interestexpense

#     Insert ticker and df
    return dfformatted

getfinancialreportingdf('swks')