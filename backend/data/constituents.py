# S&P 500 + NASDAQ 100 universe
# Excludes BF.B, BRK.B (dot-ticker edge cases that fail Alpaca API)
# NOTES: Survivorship bias exists — this is the CURRENT index composition.

SP500_TICKERS = [
    'A','AAPL','ABBV','ABNB','ABT','ACGL','ACN','ADBE','ADI','ADM','ADP','ADSK',
    'AEE','AEP','AES','AFL','AIG','AIZ','AJG','AKAM','ALB','ALGN','ALL','ALLE',
    'ALNY','AMAT','AMCR','AMD','AME','AMGN','AMP','AMT','AMZN','ANET','AON','AOS',
    'APA','APD','APH','APO','APP','APTV','ARE','ARES','ARM','ASML','ATO','AVB',
    'AVGO','AVY','AWK','AXON','AXP','AZO','BA','BAC','BALL','BAX','BBY','BDX',
    'BEN','BG','BIIB','BK','BKNG','BKR','BLDR','BLK','BMY','BR','BRO','BSX',
    'BX','BXP','C','CAG','CAH','CARR','CAT','CB','CBOE','CBRE','CCEP','CCI',
    'CCL','CDNS','CDW','CEG','CF','CFG','CHD','CHRW','CHTR','CI','CIEN','CINF',
    'CL','CLX','CMCSA','CME','CMG','CMI','CMS','CNC','CNP','COF','COIN','COO',
    'COP','COR','COST','CPAY','CPB','CPRT','CPT','CRH','CRL','CRM','CRWD','CSCO',
    'CSGP','CSX','CTAS','CTRA','CTSH','CTVA','CVNA','CVS','CVX','D','DAL','DASH',
    'DD','DDOG','DE','DECK','DELL','DG','DGX','DHI','DHR','DIS','DLR','DLTR',
    'DOC','DOV','DOW','DPZ','DRI','DTE','DUK','DVA','DVN','DXCM','EA','EBAY',
    'ECL','ED','EFX','EG','EIX','EL','ELV','EME','EMR','EOG','EPAM','EQIX',
    'EQR','EQT','ERIE','ES','ESS','ETN','ETR','EVRG','EW','EXC','EXE','EXPD',
    'EXPE','EXR','F','FANG','FAST','FCX','FDS','FDX','FE','FER','FFIV','FICO',
    'FIS','FISV','FITB','FIX','FOXA','FRT','FSLR','FTNT','FTV','GD','GDDY','GE',
    'GEHC','GEN','GEV','GILD','GIS','GL','GLW','GM','GNRC','GOOG','GOOGL','GPC',
    'GPN','GRMN','GS','GWW','HAL','HAS','HBAN','HCA','HD','HIG','HII','HLT',
    'HOLX','HON','HOOD','HPE','HPQ','HRL','HSIC','HST','HSY','HUBB','HUM','HWM',
    'IBKR','IBM','ICE','IDXX','IEX','IFF','INCY','INSM','INTC','INTU','INVH',
    'IP','IQV','IR','IRM','ISRG','IT','ITW','IVZ','J','JBHT','JBL','JCI',
    'JKHY','JNJ','JPM','KDP','KEY','KEYS','KHC','KIM','KKR','KLAC','KMB','KMI',
    'KO','KR','KVUE','L','LDOS','LEN','LH','LHX','LII','LIN','LLY','LMT',
    'LNT','LOW','LRCX','LULU','LUV','LVS','LW','LYB','LYV','MA','MAA','MAR',
    'MAS','MCD','MCHP','MCK','MCO','MDLZ','MDT','MELI','MET','META','MGM','MKC',
    'MLM','MMM','MNST','MO','MOH','MOS','MPC','MPWR','MRK','MRNA','MRVL','MS',
    'MSCI','MSFT','MSI','MSTR','MTB','MTCH','MTD','MU','NCLH','NDAQ','NDSN',
    'NEE','NEM','NFLX','NI','NKE','NOC','NOW','NRG','NSC','NTAP','NTRS','NUE',
    'NVDA','NVR','NWSA','NXPI','O','ODFL','OKE','OMC','ON','ORCL','ORLY','OTIS',
    'OXY','PANW','PAYC','PAYX','PCAR','PCG','PDD','PEG','PEP','PFE','PFG','PG',
    'PGR','PH','PHM','PKG','PLD','PLTR','PM','PNC','PNR','PNW','PODD','POOL',
    'PPG','PPL','PRU','PSA','PSX','PTC','PWR','PYPL','QCOM','RCL','REG','REGN',
    'RF','RJF','RL','RMD','ROK','ROL','ROP','ROST','RSG','RTX','RVTY','SBAC',
    'SBUX','SCHW','SHOP','SHW','SJM','SLB','SMCI','SNA','SNPS','SO','SOLV','SPG',
    'SPGI','SRE','STE','STLD','STT','STX','STZ','SW','SWK','SWKS','SYF','SYK',
    'SYY','T','TAP','TDG','TDY','TEAM','TECH','TEL','TER','TFC','TGT','TJX',
    'TKO','TMO','TMUS','TPL','TPR','TRGP','TRI','TRMB','TROW','TRV','TSCO',
    'TSLA','TSN','TT','TTD','TTWO','TXN','TXT','TYL','UAL','UBER','UDR','UHS',
    'ULTA','UNH','UNP','UPS','URI','USB','V','VICI','VLO','VLTO','VMC','VRSK',
    'VRSN','VRTX','VST','VTR','VTRS','VZ','WAB','WAT','WBD','WDAY','WDC','WEC',
    'WELL','WFC','WM','WMB','WMT','WRB','WSM','WST','WTW','WY','WYNN','XEL',
    'XOM','XYL','YUM','ZBH','ZBRA','ZS','ZTS',
]

NASDAQ100_TICKERS = [
    'AAPL','ABNB','ADBE','ADI','ADP','ADSK','AEP','AMAT','AMD','AMGN','AMZN',
    'ANSS','APP','ARM','ASML','AVGO','AXON','BIIB','BKNG','BKR','CCEP','CDNS',
    'CDW','CEG','CHTR','CMCSA','COST','CPRT','CRWD','CSCO','CSGP','CSX','CTAS',
    'CTSH','DASH','DDOG','DLTR','DXCM','EA','EXC','FANG','FAST','FTNT','GEHC',
    'GEN','GILD','GOOG','GOOGL','HON','HOOD','IDXX','ILMN','INTC','INTU','ISRG',
    'KDP','KHC','KLAC','LRCX','LULU','MAR','MCHP','MDLZ','MELI','META','MNST',
    'MRNA','MRVL','MSFT','MU','NFLX','NDAQ','NVDA','NXPI','ODFL','ON','ORLY',
    'PANW','PAYX','PCAR','PDD','PEP','PLTR','PYPL','QCOM','REGN','ROST','SBUX',
    'SHOP','SMCI','SNPS','TEAM','TT','TMUS','TSLA','TTD','TTWO','TXN','VRSK',
    'VRTX','WBD','WDAY','XEL','ZS',
]

# Single default universe: every stock in SPY (S&P 500) + QQQ (NASDAQ 100),
# de-duplicated, sorted. This is the only universe the app uses.
SPY_QQQ_TICKERS = sorted(set(SP500_TICKERS) | set(NASDAQ100_TICKERS))

UNIVERSE_PRESETS = {
    'spy_qqq': SPY_QQQ_TICKERS,
    # legacy aliases kept so old saved configs / requests still resolve
    'sp500': SPY_QQQ_TICKERS,
    'nasdaq100': SPY_QQQ_TICKERS,
    'top50': SPY_QQQ_TICKERS,
}
