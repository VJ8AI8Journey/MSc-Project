from __future__ import annotations

from finrl.test import test


def trade(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    trade_mode="backtesting",
    if_vix=True,
    **kwargs,
):
    if trade_mode == "backtesting":
        # use test function for backtesting mode
        test(
            start_date,
            end_date,
            ticker_list,
            data_source,
            time_interval,
            technical_indicator_list,
            drl_lib,
            env,
            model_name,
            if_vix=True,
            **kwargs,
        )

    else:
        raise ValueError(
            "Invalid mode input! Please input either 'backtesting' or 'paper_trading'."
        )
