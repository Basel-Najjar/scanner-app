import os

import streamlit as st
from datetime import timedelta
import pandas as pd
from src.pull_data import pull_watchlist_data
from src.utils import import_watchlist, get_last_market_day, get_market_hours
from src.market_profile import (
    calculate_momentum_ranks,
    scan_momentum_ranks,
    plot_scan_results,
)


@st.cache_resource
def load_watchlist():
    return import_watchlist(os.path.abspath(r"files/watchlist.txt"))


@st.cache_resource
def load_last_market_day():
    return get_last_market_day()


@st.cache_resource
def load_data(watchlist, starting_day, last_market_day):
    return pull_watchlist_data(
        watchlist, start=starting_day, end=last_market_day + timedelta(days=1)
    )


def main():
    st.title("Momentum Scan")
    watchlist = load_watchlist()
    st.dataframe(pd.DataFrame(watchlist, columns=["Watchlist"]))
    last_market_day = load_last_market_day()
    starting_day = last_market_day - timedelta(days=30)
    if st.button("Run Momentum Scan"):
        with st.spinner("Running scan..."):
            df_raw = load_data(watchlist, starting_day, last_market_day)
            st.write(f"Pulled data from {starting_day} to {last_market_day}")
            momentum = calculate_momentum_ranks(df_raw)
            scan = scan_momentum_ranks(momentum)
            st.write("Scan complete!")

        st.success("Scan complete.")
        for symbol in scan.columns:
            plot = plot_scan_results(scan, symbol)
            st.plotly_chart(plot)


if __name__ == "__main__":
    main()
