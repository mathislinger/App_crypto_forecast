# Running app available at http://localhost:8050/mli/projet_info/crypto/

import pickle

# Modules needed
import bs4
from urllib import request
import pandas as pd
import numpy as np
import time
import re
import sys
import datetime
import requests
import io

# Modules for the prediction
from stockstats import StockDataFrame
from sklearn.ensemble import RandomForestClassifier

# Twitter modules needed
import tweepy
import datetime as dt
from textblob import TextBlob
import base64
from io import BytesIO

# For Wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
from PIL import Image, ImageDraw, ImageFont

# Dash application
import dash
import dash_core_components as dcc
import dash_html_components as html

import colorlover as cl
import datetime as dt
import flask
from flask_cors import CORS
import os
from pandas_datareader.data import DataReader
import time

# Set os.dir
os.chdir('/Users/Linger/Desktop/ENSAE_MS/S2/crypto_forecast')

# BEGIN: Scrap symbols for dropdown options as well as hashtags for the Twitter map
request_text = request.urlopen('https://fr.finance.yahoo.com/crypto-monnaies?offset=0&count=100').read()
text = bs4.BeautifulSoup(request_text, "lxml")
list_symbols = [ i.text for i in text.find('table').findAll('td', {'class' : 'Va(m) Fz(s) Ta(start) Pstart(6px) Pend(10px) Whs(nw)'}) ]
hashtags = [ i.text.split(' ', 1)[0] for i in text.find('table').findAll('td', {'class' : 'Va(m) Fz(s) Ta(start) Pend(10px)'}) ]
# END: Scrap symbols for dropdown options as well as hashtags for the Twitter map

# BEGIN: set Twitter API
auth = tweepy.OAuthHandler('4po3iSj9wYwjtjhtgZo5vs4G6', 'NKQEZJ1RVxDsGUNipfqNBF1uvKGtaZouy6vrzCU1ddUG6V35zu')
auth.set_access_token('927947724267446273-CvCtLeWnselhi05pcRVkln1JzXdSHgK', 'DBXlJ2hYMkrB4uVMS2HzoiypO40xbN5J9gepyPopuzlrH')

api = tweepy.API(auth)
# END: set Twitter API

# BEGIN: Functions to download the Yahoo financial data
def get_cookie_value(r):
    return {'B': r.cookies['B']}

def get_page_data(symbol):
    url = "https://finance.yahoo.com/quote/%s/?p=%s" % (symbol, symbol)
    r = requests.get(url)
    cookie = get_cookie_value(r)
    lines = r.content.decode('unicode-escape').strip(). replace('}', '\n')
    return cookie, lines.split('\n')

def find_crumb_store(lines):
    for l in lines:
        if re.findall(r'CrumbStore', l):
            return l
    print("Did not find CrumbStore")

def split_crumb_store(v):
    return v.split(':')[2].strip('"')

def get_cookie_crumb(symbol):
    cookie, lines = get_page_data(symbol)
    crumb = split_crumb_store(find_crumb_store(lines))
    return cookie, crumb

def get_data(symbol, start_date, end_date, cookie, crumb):
    filename = '%s.csv' % (symbol)
    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (symbol, start_date, end_date, crumb)
    response = requests.get(url, cookies=cookie)
    urlData = response.content
    rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
    return rawData
    
def get_now_epoch():
    return int(time.time())

def download_quotes(symbol):
    start_date = 0
    end_date = get_now_epoch()
    cookie, crumb = get_cookie_crumb(symbol)
    return get_data(symbol, start_date, end_date, cookie, crumb)
# END: Functions to download the Yahoo financial data

# BEGIN: Functions to create features for prediction
def T(data, col, n_days, target_margin):
    matrix = np.hstack([np.array(data[col].pct_change(periods=i+1).shift(-(i+1))).reshape((-1,1)) for i
                        in range(n_days)])
    return np.apply_along_axis(func1d=lambda x: sum(x[np.logical_or((np.abs(x) > target_margin), (np.isnan(x)))]),
                    axis=1, arr=matrix)
# END: Functions to create features for prediction

# BEGIN: Function get tweets
def get_tweet(hashtag, max_tweets = 100):
    searched_tweets = [status for status in tweepy.Cursor(api.search, q=hashtag, lang = 'en',
                                                          result_type = 'mixed', tweet_mode="extended", count = 100,
                                                          since = (dt.datetime.today() - dt.timedelta(days=7)
                                                                  ).date()).items(max_tweets)]
    tweet_list = []
    for tweet in searched_tweets:
        tweet_list.append(tweet._json)
    text_tweets = []
    #loc_tweet = []
    for tweet in tweet_list:
        text_tweets.append(tweet['full_text'])
        #loc_tweet.append(tweet['user']['location'])
    return list(set(text_tweets))#, list(text_tweets)
# END: Function get tweets

# BEGIN: Function to display Boolinger bands on candlechart
def bbands(price, window_size=6, num_of_std=2):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band
# END: Function to display Boolinger bands on candlechart


# BEGIN: Get templates
app = dash.Dash(
    'stock-tickers',
    url_base_pathname='/mli/projet_info/crypto/')
server = app.server
CORS(server)

if 'DYNO' in os.environ:
    app.config.routes_pathname_prefix = '/dash/gallery/stock-tickers/'
    app.config.requests_pathname_prefix = 'https://dash-stock-tickers.herokuapp.com/dash/gallery/stock-tickers/'

app.config['suppress_callback_exceptions']=True
app.scripts.config.serve_locally = False
dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-finance-1.28.0.min.js'

colorscale = cl.scales['9']['qual']['Paired']
# END: Get templates


# BEGIN: Web app structure
app.layout = html.Div([
    html.Div([
        html.H2('Crypto Currency Project',
                style={'display': 'inline',
                       'float': 'left',
                       'font-size': '2.65em',
                       'margin-left': '7px',
                       'font-weight': 'bolder',
                       'font-family': 'Product Sans',
                       'color': "rgba(117, 117, 117, 0.95)",
                       'margin-top': '20px',
                       'margin-bottom': '0'
                       }),
        html.Img(src="http://www.ensai.fr/files/_media/images/l_ecole/Partenaires/Ecoles/ensae_logo_dev.png",
                style={
                    'height': '100px',
                    'float': 'right'
                },
        ),
    ]),
    dcc.Dropdown(
        id='stock-ticker-input',
        options=[{'label': s, 'value': s} for s in list_symbols],
        value='ETH-EUR',
    ),
    html.Div(id='graphs'),
    html.Div(id='sentiments', style={'textAlign': 'center', 'font-family': 'Product Sans', 'font-size': '1.2em', 'marginTop': 40}),
    html.Div([
    html.Div('Number of days', style={'textAlign': 'center', 'font-family': 'Product Sans', 'font-size': '1.2em', 'marginTop': 40}),
    dcc.Slider(id='number-of-days',
        min=1,
        max=10,
        step=1,
        value=3,
        marks={i+1: '{}'.format(i+1) for i in range(10)}),
    html.Div(id='updatemode-output-container', style={'textAlign': 'center', 'font-family': 'Product Sans', 'font-size': '1.2em', 'marginTop': 40}),
    dcc.Slider(id='target-margin',
        min=0,
        max=1,
        step=.05,
        value=.1,
        )]),
    html.Div(id='prediction')
], className="container")
# END: Web app structure


# BEGIN: Callbacks
@app.callback(
    dash.dependencies.Output('updatemode-output-container', 'children'),
    [dash.dependencies.Input('target-margin', 'value')])
def display_value(value):
    return 'Target margin: buy or sell when the variation in price is at least {}%'.format(value*100)

@app.callback(
    dash.dependencies.Output('graphs','children'),
    [dash.dependencies.Input('stock-ticker-input', 'value')])
def update_graph(tickers):
    try:
        df = download_quotes(tickers)
    except:
        graphs.append(html.H3(
            'Data is not available for {}'.format(tickers),
            style={'marginTop': 20, 'marginBottom': 20}
        ))

    # Create candlestick
    candlestick = [{
        'x': df.Date, 'yaxis': 'y2',
        'open': df['Open'],
        'high': df['High'],
        'low': df['Low'],
        'close': df['Close'],
        'type': 'candlestick',
        'name': tickers,
        'legendgroup': tickers,
        'increasing': {'line': {'color': colorscale[0]}},
        'decreasing': {'line': {'color': colorscale[1]}}
    }]
    # Create Bollinger bands
    bb_bands = bbands(df.Close)
    bollinger_traces = [{
        'x': df.Date, 'y': y, 'yaxis': 'y2',
        'type': 'scatter', 'mode': 'lines',
        'line': {'width': 1, 'color': colorscale[(i*2) % len(colorscale)]},
        'hoverinfo': 'none',
        'legendgroup': tickers,
        'showlegend': True if i == 0 else False,
        'name': 'Bollinger bands'
    } for i, y in enumerate(bb_bands)]
    # Create Volume chart
    # Set volume bar chart colors
    colors = []
    for i in range(len(df.Close)):
        if i != 0:
            if df.Close[i] > df.Close[i-1]:
                colors.append(colorscale[0])
            else:
                colors.append(colorscale[1])
        else:
            colors.append(colorscale[1])
    volume = [{
        'x': df.Date, 'y': df.Volume,
        'type': 'bar', #'yaxis': 'y',
        'marker': {'color': colors},
        'name': 'Volume'
    }]
    # Add rangeselecctor
    rangeselector=dict(
    visibe = True,
    x = 0, y = 0.9,
    bgcolor = 'rgba(150, 200, 250, 0.4)',
    font = dict( size = 13 ),
    buttons=list([
        dict(count=1,
            label='reset',
            step='all'),
        dict(count=3,
            label='3 mo',
            step='month',
            stepmode='backward'),
        dict(count=2,
            label='2 mo',
            step='month',
            stepmode='backward'),
        dict(count=1,
            label='1 mo',
            step='month',
            stepmode='backward'),
        dict(step='all')
    ]))
    # Append the graph
    graphs = dcc.Graph(
        id=tickers,
        figure={
            'data': candlestick + bollinger_traces + volume,
            'layout': {
                'margin': {'b': 0, 'r': 10, 'l': 60, 't': 70},
                'legend': {'orientation': 'h', 'yanchor': 'bottom', 'x': 0.3, 'y': 0.9},
                'xaxis': {'rangeselector': rangeselector},
                'yaxis': {'showticklabels': False},
                'title': '{} Stock market'.format(tickers),
            }
        }
    )

    return graphs


@app.callback(
    dash.dependencies.Output('sentiments','children'),
    [dash.dependencies.Input('stock-ticker-input', 'value')])
def sentiment(tickers):
    hashtag = '#' + hashtags[list_symbols.index(tickers)]
    #with open("test.txt", "rb") as fp:#
    #    tweet_list = pickle.load(fp)#
    tweet_list = get_tweet(hashtag)

    ## Sentiments
    sentimentlist = []
    subjectivitylist = []
    for tweet in tweet_list:
        analysis = TextBlob(tweet).sentiment
        sentimentlist.append(analysis.polarity)
        subjectivitylist.append(analysis.subjectivity)
    sentimentavg = float(sum(sentimentlist) / max(len(sentimentlist), 1))
    subjectivityavg = float(sum(subjectivitylist) / max(len(subjectivitylist), 1))
    return 'For {}, average polarity is {} and average subjectivity is {}'.format(hashtag, round(sentimentavg, 2), round(subjectivityavg, 2))


@app.callback(
    dash.dependencies.Output('prediction','children'),
    [dash.dependencies.Input('stock-ticker-input', 'value'), dash.dependencies.Input('number-of-days', 'value'), dash.dependencies.Input('target-margin', 'value')])
def prediction(tickers, nb_days, target_margin):
    try:
        data = download_quotes(tickers)
    except:
        graphs.append(html.H3(
            'Data is not available for {}'.format(tickers),
            style={'marginTop': 20, 'marginBottom': 20}
        ))
    # Set the parameters for the model
    n_days_ = nb_days
    target_margin_ = target_margin
    # Start to create the features
    data['dai_avg_price'] = (data['Close'] + data['High'] + data['Low']) / 3
    data['Index'] = T(data, 'dai_avg_price', n_days_, target_margin_)
    data.loc[data.Index < (-(target_margin_)), 'signal'] = 'sell'
    data.loc[np.abs(data.Index) <= target_margin_, 'signal'] = 'hold'
    data.loc[data.Index > target_margin_, 'signal'] = 'buy'
    data = data.drop(['Index'], axis = 1)
    stock = StockDataFrame.retype(data)
    indicators_list = ['macd', 'kdjk', 'boll', 'tr', 'atr', 'dma', 'pdi', 'mdi', 'dx', 'adx', 'adxr', 'trix', 'vr']
    for i in indicators_list:
        stock.get(i)
    # Keep today's day to predict
    vec_for_pred = data.drop(['signal'], axis = 1).iloc[len(data)-n_days_:,:]
    # Delete nan and infinity values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    # Do the train set
    X = data.drop(['signal'], axis = 1).as_matrix()
    y = data['signal'].ravel()
    # Model
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X,y)
    # Prediction
    pred = rf.predict(vec_for_pred.as_matrix())
    text = 'For the next {} days, you should {}'.format(n_days_, pred[len(pred)-1])
    if pred[len(pred)-1] == 'hold':
        return html.Div(text, style={'textAlign': 'center', 'color': 'orange', 'font-family': 'Product Sans', 'font-size': '2em', 'marginTop': 40})
    elif pred[len(pred)-1] == 'sell':
        return html.Div(text, style={'textAlign': 'center', 'color': 'red', 'font-family': 'Product Sans', 'font-size': '2em', 'marginTop': 40})
    else:
        return html.Div(text, style={'textAlign': 'center', 'color': 'green', 'font-family': 'Product Sans', 'font-size': '2em', 'marginTop': 40})






external_css = ["https://fonts.googleapis.com/css?family=Product+Sans:400,400i,700,700i",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/2cc54b8c03f4126569a3440aae611bbef1d7a5dd/stylesheet.css"]

for css in external_css:
    app.css.append_css({"external_url": css})


if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })


if __name__ == '__main__':
    app.run_server(debug=True)