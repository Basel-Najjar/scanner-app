import pandas_market_calendars as mcal

extreme_threshold = 9.0
breakout_threshold = 5.0
default_lookback = 14
string_date_format = f"%Y-%m-%d"
exchange = "NYSE"
exch = mcal.get_calendar(exchange)
