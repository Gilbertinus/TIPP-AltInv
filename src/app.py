import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.express as px
import pandas_datareader as web
from dash import Dash, dcc, html, Input, Output, callback
import dash_table as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'

df = pd.read_excel("https://raw.githubusercontent.com/Gilbertinus/TIPP/master/AI_data_topic2.xlsx", index_col=0, parse_dates=True)
df.iloc[:,-2:] = df.iloc[:,-2:] / 12


list_col = ['Vol', 'Excess return', 'Beta', 'Jensen', 'Sharpe',
            'Treynor', 'Sortino', 'Calmar', 'Sterling',
            'M2', 'Skew', 'Kurt']
list_row = ["Risky", "TIPP"]

df_table = pd.DataFrame().reindex(index=list_row, columns=list_col).fillna(0)
df_table = df_table.reset_index()
df_table = df_table.rename(columns={"index":"Portfolio"})

form = [{"specifier":".2%"}, {"specifier":".2%"}, {"specifier":".2f"},
        {"specifier":".4f"}, {"specifier":".2f"}, {"specifier":".2f"},
        {"specifier":".2f"}, {"specifier":".2f"}, {"specifier":".2f"},
        {"specifier":".2f"}, {"specifier":".2f"}, {"specifier":".2f"}]

dict1 = [{"id":"Portfolio", "name":"Portfolio"}]
dict2 = [{"name":i, "id":i, "format":form[num], "type":"numeric"} for num, i in enumerate(list_col)]
dict1.extend(dict2)



def tipp(data:pd.Series, 
         rf:(int or float or pd.Series)=0, 
         floor:(float or int)=1, 
         max_risky:(float or int)=1, 
         min_risky:(float or int)=0, 
         lock:(float or int)=0, 
         cap_inj:(float or int)=0, 
         cap_amount:(float or int)=1, 
         init_cap:(float or int)=100000,
         nb_trades:int=12):
    """
    This function calculates the TIPP and all the values that are important for it.
    
    Arguments:
    ––––––––––
    - data (Series): contains the price series for the portfolio or asset in question.

    - rf (int, float, Series): contains risk-free rates.
        By default, rf=0.

    - floor (int, float): contains the % of floor.
        By default, floor=1 (100%).

    - max_risky (int, float): contains the maximum % of risky assets.
        This value is used to calculate the capital invested in risky assets.
        By default, max_risky=1 (100%).

    - min_risky (int, float): contains the minimum % of risky assets.
        This value is used to calculate the capital invested in risky assets.
        By default, min_risky=0 (0%).

    - lock (int, float): contains the lock-in. This value is used to calculate the TIPP.
        By default, lock=0 (0%).

    - cap_inj (float, int): contains the minimum value before a capital reinjection is required. 
        It is calculated as the ratio between TIPP in t and Ratchet in t-1.
        By default, cap_inj=0 (0%), so TIPP{t}/Rat{t-1} > 0%.

    - cap_amount (float, int): contains the value of the capital to be reinjected if any is required. 
        It is defined as the percentage of the difference between the ratchet and the TIPP.
        By default, cap_amount=1 (100%).

    - init_cap (float, int): contains the initial capital invested in the portfolio.
        By default, init_cap=100,000.

    - nb_trades (int): contains the number of trades per year.
        By default, nb_trades=12 (monthly data).

    
    Returns:
    ––––––––
    - df (DataFrame): a DataFrame that contains:
        - BH: the value if invested only in the portfolio.
        - TIPP: the value of the TIPP.
        - Ratchet: the value of the ratchet capital.
        - Floor: the value of the floor.
        - Cushion: the value of the cushion.
        - Risky: the capital invested in the risky asset.
        - Safe: the capital invested in the safe asset.
        - Capital Injection: the capital to inject if we need to.
    """

    # If rf is not a series, then we create a series with the same indexes as data.
    if not isinstance(rf, pd.Series):
        rf = pd.Series().reindex_like(data).fillna(0).add(rf)
    
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    # Create Series that will contains different values
    TIPP = pd.Series().reindex_like(data).fillna(0) # TIPP
    F = pd.Series().reindex_like(data).fillna(0) # FLOOR
    C = pd.Series().reindex_like(data).fillna(0) # Cushion
    R = pd.Series().reindex_like(data).fillna(0) # Risky asset
    S = pd.Series().reindex_like(data).fillna(0) # Safe asset
    Rat = pd.Series().reindex_like(data).fillna(0) # Ratchet
    CapInj = pd.Series().reindex_like(data).fillna(0) # Capital Injection

    #### t=0
    # Compute the first value of portfolio
    TIPP.iloc[0] = init_cap
    Rat.iloc[0] = init_cap

    # Compute the floor
    F.iloc[0] = floor * TIPP.iloc[0]

    # Compute the cushion
    C.iloc[0] = TIPP.iloc[0] - F.iloc[0]

    # Compute the multiplier
    m = TIPP.iloc[0] / C.iloc[0]
    
    # Compute the risky asset and safe asset
    R.iloc[0] = max(min(m * C.iloc[0], 
                        TIPP.iloc[0]*max_risky), 
                    TIPP.iloc[0]*min_risky)
    S.iloc[0] = TIPP.iloc[0] - R.iloc[0]

    ### t>0
    for i in range(1, len(data)):
        # Value of TIPP
        TIPP.iloc[i] = R.iloc[i-1] * data.iloc[i]/data.iloc[i-1] + S.iloc[i-1] * np.exp(rf.iloc[i] / nb_trades)

        # If TIPP{t} / Rat{t-1} < cap_inj, we need to reinject some capital, based on cap_amount.
        if TIPP.iloc[i] / Rat.iloc[i-1] < cap_inj:
            CapInj.iloc[i] = (Rat.iloc[i-1] - TIPP.iloc[i]) * cap_amount

        # Ratchet capital
        Rat.iloc[i] = Rat.iloc[i-1] - CapInj.iloc[i]
        # If TIPP{t} / Rat{t-1} >= 1+lock, the ratchet at t is given by the value of TIPP at t
        if TIPP.iloc[i] / Rat.iloc[i-1] >= 1 + lock:
            Rat.iloc[i] = TIPP.iloc[i] - CapInj.iloc[i]
        
        # Floor
        F.iloc[i] = floor * Rat.iloc[i]

        # Cushion
        C.iloc[i] = TIPP.iloc[i] - F.iloc[i]

        # Part invested in the risky asset
        R.iloc[i] = max(min(m * C.iloc[i], 
                            TIPP.iloc[i]*max_risky), 
                        TIPP.iloc[i]*min_risky)
        # Part invested in the safe asset
        S.iloc[i] = TIPP.iloc[i] - R.iloc[i]

    ### Value of the portfolio
    BH = pd.Series().reindex_like(data).fillna(0)
    BH.iloc[0] = init_cap
    BH.iloc[1:] = BH.iloc[0] * data.iloc[1:] / data.iloc[0]

    ### Creation of DataFrame    
    df = pd.DataFrame([BH, TIPP, Rat, F, C, R, S, CapInj],
                      index=["BH", "TIPP", "Ratchet", "Floor", "Cushion", "Risky", "Safe", "Capital Injection"]).T

    return df

# Drawdown
def drawdown(df:(pd.core.frame.DataFrame or pd.core.series.Series), 
             minimum:bool=False):
    """
    This function is used to calculate the Drawdown for a dataset.

    Arguments:
    ––––––––––
    - df (DataFrame or Series): contains the data for the drawdown.
    - minimum (bool): if True, return the Maximum Drawdown, otherwise, it returns the Drawdown. 
        By default, minimum=False.

    Returns:
    ––––––––
    - drawdown (DataFrame or Series): contains the drawdown.
    """
    
    # Compute the maximum value over the time
    maximum = df.cummax()
    # Compute the drawdown
    drawdown = df / maximum - 1

    # If minimum is True, return the Maximum Drawdown
    if minimum is True:
        return drawdown.cummin()
    # Otherwise, return the Drawdown
    return drawdown

def stats_table(data, risk_free, benchmark, drawdown):

    # Compute the returns
    index_ret_BH = data.pct_change().dropna()
    ret_benchmark = benchmark.pct_change().dropna()

    # Transform to Series if it is a DataFrame
    if isinstance(index_ret_BH, pd.DataFrame):
        index_ret_BH = index_ret_BH.squeeze()
    if isinstance(ret_benchmark, pd.DataFrame):
        ret_benchmark = ret_benchmark.squeeze()
    
    # Keep the same index as index_ret_BH if risk free is a Pandas object
    if isinstance(risk_free, (pd.Series, pd.DataFrame)):
        risk_free = risk_free.loc[index_ret_BH.index]
    elif isinstance(risk_free, (float, int)):
        risk_free = pd.Series().reindex_like(index_ret_BH).fillna(0).add(risk_free)
    
    # Annualized volatility
    Volatility_BH = index_ret_BH.std()*(12**0.5)

    stats_BH = pd.DataFrame(Volatility_BH, columns=["Vol"], index=["Stats"])
    stats_BH["Excess return"] = index_ret_BH.subtract(risk_free, axis=0).mean()*12

    # Compute the beta and the alpha
    beta, alpha, _, _, _ = linregress(ret_benchmark.subtract(risk_free, axis=0), 
                                      index_ret_BH.subtract(risk_free, axis=0))    
    stats_BH["Beta"] = beta
    stats_BH["Jensen"] = alpha

    # Compute different ratios
    stats_BH["Sharpe"] = stats_BH["Excess return"]/stats_BH["Vol"]
    stats_BH["Treynor"] = stats_BH["Excess return"] / beta
    stats_BH["Sortino"] = stats_BH["Excess return"]/(index_ret_BH[index_ret_BH<0].std()*(12**0.5))
    stats_BH["Calmar"]  = (index_ret_BH.mean()*12)/np.abs(drawdown.cummin().iloc[-1])
    stats_BH["Sterling"] = stats_BH["Excess return"]/np.abs(drawdown.cummin().iloc[-1])
    stats_BH["M2"] = stats_BH["Sharpe"] * (ret_benchmark.values.std()*(12**0.5)) + (risk_free.mean()*12)
    
    # Skewness and kurtosis
    stats_BH["Skew"] = index_ret_BH.skew()
    stats_BH["Kurt"] = index_ret_BH.kurt() + 3 # +3 because .kurt()=Excess kurtosis

    return stats_BH




font = "sans-serif"

app = Dash(__name__)
server = app.server

app.layout = html.Div(style={"backgroundColor":"white"}, 
                      children=[
        
        html.H1("Portfolio with TIPP", style={'textAlign':'center', "font-family":font}),

        html.Div([
            html.Div([
                dcc.Markdown("""
                         With this application, you can design your own TIPP with the parameters you want.

                        You can also include the assets you want. A portfolio is created based on Equally Weighted, so each asset will have a weight equal to 1/N.

                        All figures to be entered are in %, except for the initial capital. For example, if we want a lock-in of 5%, then we need to enter the value 5 in the corresponding box.
                         """, style={"font-family":font}),

                html.H4("Select the assets to include:", style={"font-family":font}),
                dcc.Checklist(id="assets",
                              options=[
                                  {"label": "S&P500", "value":"SPX"},
                                  {"label":"Bloomberg US Aggregate", "value":'Bloomberg US AGG'},
                                  {"label":"NASDAQ", "value":"NASDAQ"},
                                  {"label":"S&P US Treasury Bond", "value":'SP US Treasury Bond'},
                                  {"label":"VIX", "value":"VIX"}
                                  ],
                              value=['SPX', 'Bloomberg US AGG', 'NASDAQ', 'SP US Treasury Bond', 'VIX'],
                              inline=True),

                html.H4("Select the index:", style={"font-family":font}),
                dcc.Dropdown([{"label":"Russell 3000", "value":"Russell3000"}, 
                               {"label":"S&P500", "value":"SPX"}],
                             "Russell3000",
                             id="index_selection"),

                html.H4("If constant risk-free rate, insert a value in %:"),
                dcc.Input(value="", type="text", id="risk_free_selection"),


                
                #html.H4("Select the period of time:"),
                #dcc.DatePickerRange(min_date_allowed=df.index[0],
                #                   max_date_allowed=df.index[-1],
                #                   end_date=df.index[-1],
                #                   id="year_selection"),


                html.H4("Enter the initial capital:", style={"font-family":font}),
                dcc.Input(value=100000, type="text", id="CapInit_selection"),

                html.H4("Chose the % to protect:", style={"font-family":font}),
                dcc.Slider(min=0, max=100, step=1, value=90, id="floor_selection",
                           marks={i:f"{i}%" for i in range(0, 101, 10)},
                           tooltip={"placement": "bottom", "always_visible": True}),

                html.H4("Select the value for the capital injection:", style={"font-family":font}),
                dcc.Slider(0, 100, 0.5, value=80, id="capInj_selection",
                           marks={i:f"{i}%" for i in range(0, 101, 5)},
                           tooltip={"placement": "bottom", "always_visible": True}),

                html.H4("Enter the % to reinject in capital:", style={"font-family":font}),
                dcc.Input(value=100, type="text", id="capital_selection"),

                html.H4("Enter the % of lock-in", style={"font-family":font}),
                dcc.Input(value=2, type="text", id="lockin_selection"),

                html.H4("Enter the minimum and maximum % in risky asset:", style={"font-family":font}),
                dcc.RangeSlider(min=0, max=200, step=1,
                                value=[0, 100],
                                marks={i:f"{i}%" for i in range(0, 201, 10)},
                                id="risky_selection",
                                tooltip={"placement": "bottom", "always_visible": True}),

            ], style={"width":"45%", "display":"inline-block"}),
            
            html.Div([dcc.Graph(id='Graph1'),
                      dcc.Graph(id='Graph2'),
                      dcc.Graph(id='Graph3'),
                      dcc.Graph(id='Graph4'),
                      dt.DataTable(df_table.to_dict("records"), columns=dict1, id="table1")],
                     style={"width":"50%", "display":"inline-block", "float":"right"})
        ]),
    ]
)




@callback(
    Output("Graph1", "figure"),
    Input("assets", "value"),
    Input("index_selection", "value"),
    Input("risk_free_selection", "value"),
    #Input("year_selection", "value"),
    Input("CapInit_selection", "value"),
    Input("floor_selection", "value"),
    Input("capInj_selection", "value"),
    Input("capital_selection", "value"),
    Input("lockin_selection", "value"),
    Input("risky_selection", "value")
)
def update_graph(assets, index_sel, risk_free, 
             #year_sel, 
             capInit, floor, capInj, capSel, lockin, risky):
    
    df_port, Benchmark, rf = new_df(assets, index_sel, risk_free)
    ptf_tipp, dd, list_ret = new_tipp(df_port, rf, capInit, capInj, capSel, lockin, risky, floor)


    fig = make_subplots(y_title="Value ($)", column_titles=["Performance"])
    fig.add_trace(go.Scatter(x=ptf_tipp.index, y=ptf_tipp.BH, name="Risk portfolio", line=dict(color="#0072BD")))
    fig.add_trace(go.Scatter(x=ptf_tipp.index, y=ptf_tipp.TIPP, name="TIPP", line=dict(color="#77AC30")))
    fig.add_trace(go.Scatter(x=ptf_tipp.index, y=ptf_tipp.Floor, name="Floor", line=dict(color="#A2142F")))

    return fig




@callback(
    Output("Graph2", "figure"),
    Input("assets", "value"),
    Input("index_selection", "value"),
    Input("risk_free_selection", "value"),
    #Input("year_selection", "value"),
    Input("CapInit_selection", "value"),
    Input("floor_selection", "value"),
    Input("capInj_selection", "value"),
    Input("capital_selection", "value"),
    Input("lockin_selection", "value"),
    Input("risky_selection", "value")
)
def update_graph2(assets, index_sel, risk_free, 
             #year_sel, 
             capInit, floor, capInj, capSel, lockin, risky):
    
    df_port, Benchmark, rf = new_df(assets, index_sel, risk_free)
    ptf_tipp, dd, list_ret = new_tipp(df_port, rf, capInit, capInj, capSel, lockin, risky, floor)


    fig = make_subplots(y_title="Value (%)", column_titles=["Drawdown"])
    fig.add_trace(go.Scatter(x=ptf_tipp.index, y=dd.BH, name="Risk portfolio", line=dict(color="#0072BD")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ptf_tipp.index, y=dd.TIPP, name="TIPP", line=dict(color="#77AC30")), row=1, col=1)
    return fig



@callback(
    Output("Graph3", "figure"),
    Input("assets", "value"),
    Input("index_selection", "value"),
    Input("risk_free_selection", "value"),
    #Input("year_selection", "value"),
    Input("CapInit_selection", "value"),
    Input("floor_selection", "value"),
    Input("capInj_selection", "value"),
    Input("capital_selection", "value"),
    Input("lockin_selection", "value"),
    Input("risky_selection", "value")
)
def update_graph3(assets, index_sel, risk_free, 
             #year_sel, 
             capInit, floor, capInj, capSel, lockin, risky):
    
    df_port, Benchmark, rf = new_df(assets, index_sel, risk_free)
    ptf_tipp, dd, list_ret = new_tipp(df_port, rf, capInit, capInj, capSel, lockin, risky, floor)


    fig = make_subplots(y_title="Value ($)", column_titles=["Capital Injection"])
    fig.add_trace(go.Scatter(x=ptf_tipp.index, y=ptf_tipp["Capital Injection"].cumsum(), line=dict(color="#0072BD")))
    #fig.update_yaxes("Value ($)", row=3)
    return fig


@callback(
    Output("Graph4", "figure"),
    Input("assets", "value"),
    Input("index_selection", "value"),
    Input("risk_free_selection", "value"),
    #Input("year_selection", "value"),
    Input("CapInit_selection", "value"),
    Input("floor_selection", "value"),
    Input("capInj_selection", "value"),
    Input("capital_selection", "value"),
    Input("lockin_selection", "value"),
    Input("risky_selection", "value")
)
def update_graph4(assets, index_sel, risk_free, 
             #year_sel, 
             capInit, floor, capInj, capSel, lockin, risky):
    
    df_port, Benchmark, rf = new_df(assets, index_sel, risk_free)
    ptf_tipp, dd, list_ret = new_tipp(df_port, rf, capInit, capInj, capSel, lockin, risky, floor)

    fig = make_subplots(y_title="Number of observations", x_title="Returns", column_titles=["Distribution of returns"])
    fig.add_trace(go.Histogram(x=list_ret[0], marker_color="#0072BD", name="Risk Portfolio"))
    fig.add_trace(go.Histogram(x=list_ret[1], marker_color="#77AC30", name="TIPP"))
    return fig



@callback(
    Output("table1", "data"),
    Input("assets", "value"),
    Input("index_selection", "value"),
    Input("risk_free_selection", "value"),
    #Input("year_selection", "value"),
    Input("CapInit_selection", "value"),
    Input("floor_selection", "value"),
    Input("capInj_selection", "value"),
    Input("capital_selection", "value"),
    Input("lockin_selection", "value"),
    Input("risky_selection", "value")
)
def update_table(assets, index_sel, risk_free, 
             #year_sel, 
             capInit, floor, capInj, capSel, lockin, risky):
    
    df_port, Benchmark, rf = new_df(assets, index_sel, risk_free)
    ptf_tipp, dd, list_ret = new_tipp(df_port, rf, capInit, capInj, capSel, lockin, risky, floor)

    statsBH = stats_table(ptf_tipp.BH, rf, Benchmark, dd.BH)
    statsTIPP = stats_table(ptf_tipp.TIPP, rf, Benchmark, dd.TIPP)
    stats = pd.concat([statsBH, statsTIPP])
    stats = stats.set_axis(["Risky", "TIPP"], axis=0)
    stats = stats.reset_index()
    stats = stats.rename(columns={"index":"Portfolio"})
    
    return stats.to_dict("records")




def new_df(assets, index_sel, risk_free):
    df2 = df.copy()
    df2 = df2.loc[:, assets]

    weight = np.ones(df2.shape[1]) / df2.shape[1]
    df_port = (df2 * weight).sum(axis=1)

    Benchmark = df.loc[:, index_sel]

    # Compute the risk-free asset
    if risk_free == "":
        rf = df.iloc[:, -3]
    else:
        rf = float(risk_free) / 100

    return (df_port, Benchmark, rf)


def new_tipp(df_port, rf, capInit, capInj, capSel, lockin, risky, floor):
    
    # Initialize the values
    capInit = float(capInit)
    capInj = float(capInj) / 100
    capSel = float(capSel) / 100
    lockin = float(lockin) / 100
    risky = [float(x)/100 for x in risky]
    floor = float(floor) / 100
    
    # Compute the TIPP
    ptf_tipp = tipp(df_port, 
                    rf, 
                    floor, 
                    risky[1], risky[0], lockin, capInj, capSel, capInit)
   
    # Compute the Drawdown
    dd = drawdown(ptf_tipp)

    # Compute the returns
    list_ret = [ptf_tipp.BH.pct_change(), ptf_tipp.TIPP.pct_change()]

    return (ptf_tipp, dd, list_ret)


if __name__ == "__main__":
    app.run(debug=True)
