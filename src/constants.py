import pandas_market_calendars as mcal

default_lookback: int = 20  # default lookback window for sentiment classification
# base_url = "https://api.orats.io/datav2/"
strftime = f"%Y-%m-%d"

exchange = "NYSE"
exch = mcal.get_calendar(exchange)
