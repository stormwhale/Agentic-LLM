#smc0.4 - added caching for some tools to avoid API limits
import os
from dotenv import load_dotenv
import requests
import datetime as dt
from datetime import datetime, timedelta
from ollama import chat
import pandas as pd
import plotly.express as px
import streamlit as st
import feedparser
import time
from cache_manager import CacheManager
from openai import OpenAI
import json

# Load the API key from the local environment file
load_dotenv("vantage_key.env")
vantage_key = os.getenv("vantage_key", "")

load_dotenv("TD_api_key.env")
TD_api_key = os.getenv("TD_api_key", "")


#Define SMC class:
class MarketData:
    def __init__(self, vantage_api_key, TD_api_key):
        self.historical_data_memory = {}
        self.vantage_api_key = vantage_api_key
        self.TD_api_key = TD_api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.base_url_2 = "https://api.twelvedata.com/"
        self.base_url_3 = "https://api.twelvedata.com/price"
        self.base_url_4 = "https://api.twelvedata.com/time_series"
        self.cache = CacheManager()

    #Define breaking news method:
    def get_breaking_news(self, keywords=None):
        '''
        Fetches BREAKING NEWS from premium journalism sources (Reuters, BBC, WSJ, FT).
        Use this tool ONLY for:
        - Active geopolitical events (wars, conflicts, sanctions, diplomatic crises)
        - Breaking world news (natural disasters, political upheavals, major policy announcements)
        - Ongoing crisis updates (e.g., "What is happening with Iran RIGHT NOW?")
        
        DO NOT use for:
        - Stock market analysis or financial metrics
        - Company-specific news (use get_news_sentiment instead)
        - Sector trends or analyst opinions (use get_market_news instead)

        Args:
            keywords (str, optional): Keywords to filter news 
                                      (e.g., "Iran oil shipping", "Ukraine war sanctions").
        '''
        feeds = [
            ('Reuters Business', 'https://feeds.reuters.com/reuters/businessNews'),
            ('BBC Business', 'https://feeds.bbci.co.uk/news/business/rss.xml'),
            ('BBC World', 'https://feeds.bbci.co.uk/news/world/rss.xml'),
            ('FT Markets', 'https://www.ft.com/rss/home'),
            ('WSJ Markets', 'https://feeds.wsj.com/xml/rss/3_7085.xml')
        ]

        rss_news_collection = []

        for source_name, feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    rss_news_collection.append({
                        'title':        entry.get('title', ''),
                        'summary':      entry.get('summary', ''),
                        'url':          entry.get('link'),
                        'published_at': entry.get('published'),
                        'source':       source_name
                    })
            except Exception:
                continue

        #Tier 1 (2+ meaningful tokens):
        if keywords:
            meaningful = [kw for kw in keywords.lower().split() if len(kw) >= 3]

            strict = [
                item for item in rss_news_collection
                if sum(1 for kw in meaningful if kw in (item['title'] + ' ' + item['summary']).lower()) >= min(2, len(meaningful))
            ]
            if strict:
                return strict

            loose = [
                item for item in rss_news_collection
                if any(kw in (item['title'] + ' ' + item['summary']).lower() for kw in meaningful)
            ]
            if loose:
                return loose
            
            return (
                f"TOOL_FAILURE: No articles matching '{keywords}' were found in the RSS feeds. "
                f"Do NOT use training knowledge to substitute. Tell the user no current breaking news was retrieved."
            )
                    
        return rss_news_collection

    #Define the general news method:
    def get_general_news(self, topics='financial_markets', keywords=None):
        '''
        Fetches FINANCIAL MARKET NEWS with sentiment scores from analyst and finance sources.
        Use this tool ONLY for:
        - Sector trends and market analysis (e.g., "How is the energy sector doing?")
        - Economic indicators (Fed rates, inflation, GDP)
        - Industry-wide financial developments (e.g., "What is happening in tech stocks?")
        
        DO NOT use for:
        - Breaking world events or active conflicts (use get_breaking_news instead)
        - Company-specific news (use get_news_sentiment instead)

        Args:
            topics (str): Market topic (blockchain, earnings, ipo, mergers_and_acquisitions,
                          financial_markets, economy_fiscal, economy_monetary, economy_macro,
                          energy_transportation, finance, life_sciences, manufacturing,
                          real_estate, retail_wholesale, technology).
            keywords (str, optional): Additional keywords to narrow the search.
        '''
        params = {
            'function': 'NEWS_SENTIMENT',
            'topics': topics,
            'limit': 10,
            'sort': 'LATEST',
            'apikey': self.vantage_api_key
        }
        if keywords:
            params['keywords'] = keywords

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'feed' not in data:
                return f'No news found for topic: {topics}'

            news_collection = []

            for article in data.get('feed'):
                news_collection.append({
                    'title': article.get('title'),
                    'summary': article.get('summary'),
                    'url': article.get('url'),
                    'published_at': article.get('time_published'),
                    'source': article.get('source'),
                    'overall_sentiment': article.get('overall_sentiment_label')
                })

            return news_collection

        except Exception as e:
            return f"Error fetching general news: {str(e)}"

    
    #Define get_end_of_day_price method:
    def get_end_of_day_price(self, ticker):
        """
        Fetch the end of day price of a stock using Alpha Vantage API.

        Args:
            ticker (str): The stock symbol of the company.
        
        Returns:
            dict: A dictionary containing the current price and change percent.
        """
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker,
            'apikey': self.vantage_api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            #Check for API limits or errors:
            if 'Global Quote' not in data:
                return self.get_specific_date_price(ticker)

            quote = data.get('Global Quote', {})
            price = quote.get('05. price')
            change_percent = quote.get('10. change percent')

            if price:
                return {
                    'Current_Date': dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'symbol': ticker,
                    'price': float(price),
                    'change_percent': change_percent
                }
            else:
                return f"Error: Price for {ticker} not found in response."

        except Exception as e:
            return f"Network Error:{str(e)}"
    
    #Define get_earnings_call_transcript method:
    def get_earnings_call_transcript(self, ticker, quarter):
        '''
        This API returns the earnings call transcript for a given company in a specific quarter, covering over 15 years of history and enriched with LLM-based sentiment signals.
        If user does not specify a quarter, use the most recent available quarter.

        Args:
            ticker (str): The stock symbol of the company.
            quarter (str): Fiscal quarter in YYYYQM format. For example: quarter=2024Q1. Any quarter since 2010Q1 is supported.
        
        Returns:
            dict: A dictionary containing the earnings call transcript and sentiment scores.
        '''
        key = f'Earnings_call_transcript:{ticker}:{quarter}'
        cached = self.cache.get(key)

        if cached:
            print(f'[CACHE HIT] {key}')
            cached_at = cached.get('cached_at', 'unknown date')
            cached['transcript'] = f'[Cached as of {cached_at}]\n' + cached.get('transcript', '')
            return cached
        
        params = {
            'function': 'EARNINGS_CALL_TRANSCRIPT',
            'symbol': ticker,
            'quarter': quarter,
            'apikey': self.vantage_api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            #Check for API limits or errors:
            if 'transcript' not in data:
                return f'Error: Transcript not found for {ticker} in {quarter}'
            
            symbol = data.get('symbol')
            quarter = data.get('quarter')
            transcript = data.get('transcript', [])

            formatted_dialogue = []
            for block in transcript:
                speaker = block.get('speaker')
                title = block.get('title')
                content = block.get('content')
                sentiment = block.get('sentiment')
                time_published = block.get('time_published')

                formatted_dialogue.append(f'**Speaker: {speaker}**\n **Title: {title}**\n {content}\n **Sentiment: {sentiment}**\n **Time Published: {time_published}**\n')
            
            #Join all transcripts into one:
            full_transcript = '\n'.join(formatted_dialogue)
            
            #Safeguard from memory limitation:
            max_char = 50000
            if len(full_transcript) > max_char:
                full_transcript = full_transcript[:max_char] + "\n...[Transcript truncated due to length]..."
            
            result = {
                'symbol': symbol,
                'quarter': quarter,
                'transcript': full_transcript
            }
            
            #Add timestamp to the cached data:
            result['cached_at'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.cache.set(key, result, ttl_hours = None)

            return result
            
        except Exception as e:
            return f'Error fetching {ticker}transcript: {str(e)}'

    #Define get_news_sentiment method:
    def get_news_sentiment(self, ticker):
        '''
        Fetches sentiment scores and recent news for a SPECIFIC company.
        Use this tool ONLY when the user asks about news or sentiment for a specific stock ticker (e.g., AAPL, TSLA).
        DO NOT use for general market news or broad topics.

        Args:
            ticker (str): The stock symbol of the company.
        
        Returns:
            dict: A dictionary containing the news and sentiment scores.
        '''
        ticker_list = [t.strip().upper() for t in ticker.split(',')]
        av_tickers = ','.join(ticker_list)

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': av_tickers,
            'limit': 50,
            'apikey': self.vantage_api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            #Check for API limits or errors
            if 'feed' not in data:
                return f'limits reached or no news found for {av_tickers}'
            
            structured_news = []
            for article in data['feed']:
                sentiments = article.get('ticker_sentiment', [])

                # Check if ANY of the requested tickers match the article's sentiment data
                sentiment_ticker = None
                for target_ticker in ticker_list:
                    match = next((s for s in sentiments if s['ticker'] == target_ticker), None)
                    if match:
                        sentiment_ticker = match
                        break
                
                if sentiment_ticker:
                    relevance = float(sentiment_ticker['relevance_score'])
                    all_relevance_score = [float(s['relevance_score']) for s in sentiments]
                    max_relevance = max(all_relevance_score if all_relevance_score else [0])
                    primary_focus = (relevance >= max_relevance) or (relevance > 0.8)

                    if relevance > 0.4 and primary_focus:
                        structured_news.append({
                            'title': article['title'],
                            'relevance_score': relevance,
                            'ticker_sentiment': sentiment_ticker.get('ticker_sentiment_label'),
                            'ticker_sentiment_score': float(sentiment_ticker.get('ticker_sentiment_score', 0)),
                            'overall_sentiment': article['overall_sentiment_label'],
                            'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                            'date': dt.datetime.strptime(article['time_published'], "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M"),
                            'source': article['source_domain'],
                            'url': article['url']
                        })
                        
            sorted_news = sorted(structured_news, key=lambda x: x['relevance_score'], reverse=True)
                        
            return sorted_news[:7]
            
        except Exception as e:
            return f'Error fetching news: {str(e)}'
    
    #Define get_company_info method:
    def get_company_info(self, ticker):
        '''
        Fetches general corporate information (e.g., sector, industry, CEO, market cap).

        Args:
            ticker (str): The stock symbol of the company.
        
        Returns:
            dict: A dictionary containing the company information.
        '''
        ticker_list = [t.strip().upper() for t in ticker.split(',')]
        all_results = {}
        api_calls_made = 0

        for t in ticker_list:
            key = f'company_info:{t}'
            cached = self.cache.get(key)

            if cached:
                print(f"CACHE HIT {key}")
                cached_at = cached.get('_cached_at', 'unknown date')
                cached['_cache_note'] = f'[Cached as of {cached_at}]\n'
                all_results[t] = cached
                continue
            
            print(f"CACHE MISS: Fetching from API for {t}")
            
            if api_calls_made > 0:
                time.sleep(2)

            params = {
                'function': 'OVERVIEW',
                'symbol': t,
                'apikey': self.vantage_api_key
            }

            try:
                response = requests.get(self.base_url, params=params)
                data = response.json()
                api_calls_made += 1

                if not data:
                    all_results[t] = f"No overview data found for {t}"
                    continue
                
                if "Note" in data or "Information" in data:
                    all_results[t] = "API Limit Reached (Alpha Vantage)."
                    continue
                
                #Add timestamp to the cached data:
                data['_cached_at'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.cache.set(key, data, ttl_hours = None)
                all_results[t] = data
            except Exception as e:
                all_results[t] = f"Error fetching company info: {str(e)}"
        
        return all_results if len(ticker_list) > 1 else all_results.get(ticker_list[0])
        
    
    def get_etf_profile(self, ticker):
        '''
        Fetches ETF profile including the index tracked, expense ratio, holdings, and asset allocation.
        Use this tool whenever a user asks about an ETF, its composition, expense ratio,
        what index it tracks, or wants to compare ETFs.
        ALWAYS use this tool before making any claims about an ETF.

        Args:
            ticker (str): The ETF ticker symbol (e.g., VOO, QQQ, QQQM, SPY).

        Returns:
            dict: ETF profile including expense ratio, net assets, holdings, and sector allocation.
        '''
        ticker_list = [t.strip().upper() for t in ticker.split(',')]
        all_results = {}

        for i, t in enumerate(ticker_list):
            # Add delay between calls to avoid rate limiting
            if i > 0:
                time.sleep(10)  # 10 second delay = max 6 calls per minute
            
            params = {
                'function': 'ETF_PROFILE',
                'symbol': t,
                'apikey': self.vantage_api_key
            }

            try:
                response = requests.get(self.base_url, params=params)
                data = response.json()

                if 'Note' in data or 'Information' in data:
                    all_results[t] = f'Error: API limit reached for {t}'
                    continue

                if not data or 'net_assets' not in data:
                    fallback = self._get_etf_fallback(t)
                    all_results[t] = fallback
                    continue

                all_results[t] = {
                    'ticker': t,
                    'net_assets': data.get('net_assets'),
                    'net_expense_ratio': data.get('net_expense_ratio'),
                    'portfolio_turnover': data.get('portfolio_turnover'),
                    'dividend_yield': data.get('dividend_yield'),
                    'inception_date': data.get('inception_date'),
                    'leveraged': data.get('leveraged'),
                    'top_holdings': data.get('holdings', [])[:10],
                    'sectors': data.get('sectors', [])
                }

            except Exception as e:
                all_results[t] = f'Error fetching ETF profile for {t}: {str(e)}'

        # Return single result or dict of results
        if len(ticker_list) == 1:
            return all_results[ticker_list[0]]
        return all_results

    def _get_etf_fallback(self, ticker):
        '''Fallback to OVERVIEW endpoint when ETF_PROFILE is unavailable.'''
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': self.vantage_api_key
        }
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if not data or 'Symbol' not in data:
                return f'Error: No data available for ETF {ticker}'

            return {
                'ticker': ticker,
                'name': data.get('Name'),
                'description': data.get('Description'),
                'asset_type': data.get('AssetType'),
                'exchange': data.get('Exchange'),
                'dividend_yield': data.get('DividendYield'),
                'market_cap': data.get('MarketCapitalization'),
                'note': 'Limited ETF data available — full profile not supported for this ticker'
            }
        except Exception as e:
            return f'Error fetching ETF fallback data: {str(e)}'

    #Define get_end_of_day_price2 method:
    def get_specific_date_price(self, ticker, date=None) -> dict:
        '''
        Use this tool when the user asks for:
        - 'closing price yesterday'
        - 'what did X close at on [date]'
        - 'end of day price for [date]'

        Do NOT use get_current_price for historical closing prices.

        Args:
            ticker (str): The stock symbol of the company.
            date (str): The date in YYYY-MM-DD format.
    
        Returns:
            dict: A dictionary containing the ticker closing price with date or None if API limit is reached.
        '''
        try:
            if date is None:
                url = self.base_url_2 + 'eod'
                params = {
                    'symbol': ticker,
                    'apikey': self.TD_api_key
                }
                response = requests.get(url, params=params)
                data = response.json()

                if not data or data.get('status') == 'error':
                    return f"TD data error: {data.get('message', 'Unknown error')}"
                
                return {
                    'ticker': ticker,
                    'current_price': data.get('last'),
                    'as_of_date': data.get('datetime')
                }
            else:
                #Specific date:
                url = self.base_url_4
                params = {
                    'symbol': ticker,
                    'interval': '1day',
                    'outputsize': 1,
                    'end_date': date + ' 23:59:59',
                    'apikey': self.TD_api_key
                }

                response = requests.get(url, params=params)
                data = response.json()

                if not data or data.get('status') == 'error':
                    return f"TD data error: {data.get('message', 'Unknown error')}"
            
                values = data.get('values', [])
                if not values:
                    return f"No data found for {ticker} on {date}"

                entry = values[0]
                return_date = entry.get('datetime') #To catch weekends and holidays:
                #Define a quick function to format the date to be more human readable:
                def _format_date(date_str):
                    return dt.datetime.strptime(date_str, '%Y-%m-%d').strftime('%B %d, %Y')

                if return_date != date:
                    note = f'Note: No trading date for {_format_date(date)} was found. It could have been a weekend or public holiday. The returned date {_format_date(return_date)} is the most recent available trading day.'
                else:
                    note = f'Note: The requested date {_format_date(date)} is the most recent available trading day.'
            
                return {
                    'ticker': data.get('meta', {}).get('symbol', ticker),
                    'closing_price': entry.get('close'),
                    'as_of_date': entry.get('datetime'),
                    'note': note
                }
        except Exception as e:
            return(f"Error fetching end of day price: {str(e)}")
    
    #Define get_current_price method:
    def get_current_price(self, ticker):
        '''
        Get the real-time stock price.

        Args:
            ticker (str): The stock symbol of the company.
        
        Returns:
            dict: A dictionary containing the ticker current price with date or None if API limit is reached.
        '''
        ticker_list = [t.strip().upper() for t in ticker.split(',')]
        results = []

        for t in ticker_list:
            params = {
                'symbol': t,
                'apikey': self.TD_api_key
            }

            try:
                response = requests.get(self.base_url_3, params=params)
                data = response.json()

                if not data or data.get('status') == 'error':
                    error_msg = data.get('message', 'Unknown error')
                    results.append({'ticker': t, 'error': error_msg})
                    continue
                        
                #Extract the date and closing price:
                results.append({
                    'ticker': t,
                    'current_price': round(float(data.get('price')), 2),
                    'as_of_date': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                return(f"Error fetching current price: {str(e)}")
        
        return results[0] if len(results) == 1 else results
    
    #Define get_historical_data method:
    def get_historical_data(self, ticker, interval, start_date=None, end_date=None):
        '''
        Retrieves historical data and time-series pricing for data analysis.
        Supports multiple tickers passed as a comma-separated string (e.g., "AAPL, MSFT").
        Use this tool whenever a user asks for historical data, time-series, past stock prices, 
        or wants to calculate historical profit, Return on Investment (ROI), or past performance of an investment.

        Args:
            ticker (str): The stock symbol of the company.
            interval (str, optional): The time interval (e.g., 1min, 1day, 1week, 1month). Defaults to 1day.
            start_date (str, optional): The start date (e.g., "2025-01-01 00:00:00").
            end_date (str, optional): The end date (e.g., "2026-03-01 00:00:00").
        '''
        ticker_list = [t.strip().upper() for t in ticker.split(',')]
        all_dfs = []
        errors = []

        #Pre-fetch company name for context:
        company_names = {t: t for t in ticker_list}
        
        #Main loop to fetch data:
        for t in ticker_list:
            params = {
                'symbol': t,
                'interval': interval,
                'apikey': self.TD_api_key
            }

            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date

            try:
                response = requests.get(self.base_url_4, params=params)
                data = response.json()

                if not data or data.get('status') == 'error':
                    error_msg = data.get('message', 'Unknown error')
                    errors.append(f'{t}: {error_msg}')
                    continue
            
                #Extract data into pandas dataframe:
                df = pd.DataFrame(data['values'])
                df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                #Create a new column for the ticker:
                df['ticker'] = t

                #Convert datetime to datetime objects:
                df['datetime'] = pd.to_datetime(df['datetime'])

                #Convert numeric columns to floats:
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].astype(float)

                all_dfs.append(df)

            except Exception as e:
                errors.append(f'Error fetching {t}: {str(e)}')
        
        #Combine the dataframes:
        if not all_dfs:
            return "\n".join(errors) if errors else "No data retrieved."
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        #Store the df in class memory for the create_plot tool:
        for t in ticker_list:
            self.historical_data_memory[t] = combined_df[combined_df['ticker'] == t]
        
        llm_response_text = f"Successfully retrieved data for: {', '.join(ticker_list)}.\n\n"
        
        if errors:
            llm_response_text += f"Errors encountered: \n{', '.join(errors)}"

        #Create condensed markdown summary:
        for t in ticker_list:
            ticker_df = combined_df[combined_df['ticker'] == t]
            if not ticker_df.empty:
                ticker_df = ticker_df.sort_values(by='datetime')

                #Use Python to calculate some statistics of the data to save tokens from the Agent:
                max_price = ticker_df['high'].max()
                min_price = ticker_df['low'].min()
                avg_price = ticker_df['close'].mean()
                avg_volume = ticker_df['volume'].mean()

                name = company_names.get(t, t)
                llm_response_text += f"### Data Summary for {t} ({name}):\n"
                llm_response_text += f"- **Date Range:** {ticker_df['datetime'].iloc[0].strftime('%Y-%m-%d')} to {ticker_df['datetime'].iloc[-1]}\n"
                llm_response_text += f"- **Period High:** ${max_price:.2f}\n"
                llm_response_text += f"- **Period Low:** ${min_price:.2f}\n"
                llm_response_text += f"- **Period average:** ${avg_price:.2f}\n"
                llm_response_text += f"- **Period volume:** {avg_volume:.2f}\n"
                #Only sends first and last 5 rows to save tokens:
                llm_response_text += "First 5 rows:\n"
                llm_response_text += ticker_df.head(5).to_string(index=False) + "\n"
                llm_response_text += "Last 5 rows:\n"
                llm_response_text += ticker_df.tail(5).to_string(index=False) + "\n\n"
                llm_response_text += f"(Full {len(ticker_df)} rows stored in memory for charting)\n\n"
    
        return llm_response_text

    #Define create_plot method:
    def create_plot(self, ticker: str):
        '''
        Creates an interactive time series plot of the historical data.
        Use this tool whenever a user explicitly asks to see a plot, chart, or graph of historical data.
        
        CRITICAL INSTRUCTION: ONLY pass the ticker symbol as a string (e.g. "AAPL" or "AAPL, MSFT"). 
        DO NOT pass data arrays or dataframes. The tool already has the data in memory.

        Args:
            ticker (str): The stock symbol(s) to plot, comma-separated for multiple (e.g. "AAPL, MSFT").
        '''
        ticker_list = [t.strip().upper() for t in ticker.split(',')]

        if not hasattr(self, 'historical_data_memory') or not self.historical_data_memory:
            return f"Error: No data is in memory. Please fetch historical data first."

        dfs_to_plot = []
        missing_tickers = []

        #Gather all the tickers that user wants to plot:
        for t in ticker_list:
            if t in self.historical_data_memory and not self.historical_data_memory[t].empty:
                dfs_to_plot.append(self.historical_data_memory[t])
            else:
                missing_tickers.append(t)
        
        #Abort if no data is found:
        if not dfs_to_plot:
            return f"Error: No data is in memory for {ticker}. Please fetch historical data first."

        combined_df = pd.concat(dfs_to_plot, ignore_index=True)

        #To create the plot:
        plotted_tickers = [t for t in ticker_list if t not in missing_tickers]
        fig = px.line(
            combined_df,
            x='datetime',
            y='close',
            color='ticker',
            title=f"Stock Price(s) for {' & '.join(plotted_tickers)}"
        )

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black'
        )
        self._last_fig = fig

        success_msg = f"Successfully generated and displayed the plot for {', '.join(plotted_tickers)}"
        if missing_tickers:
            success_msg += f". Note: {', '.join(missing_tickers)} could not be plotted (no data in memory)."
        
        return success_msg
        
def _is_tool_error(result_str: str) -> bool:
    """Returns True if the tool result string signals a failure."""
    if not result_str:
        return True
    lowered = result_str.strip().lower()
    error_prefixes = ("error", "tool_failure", "no data", "limit reached", "api limit", "unknown error")
    return any(lowered.startswith(p) for p in error_prefixes)


def _wrap_tool_failure(function_name: str, ticker: str, detail: str) -> str:
    return (
        f"TOOL_FAILURE: {function_name} could not retrieve data for {ticker}. "
        f"Details: {detail}. "
        f"You MUST tell the user the data fetch failed and why. "
        f"Do NOT use your training knowledge to estimate, substitute, or fill in any "
        f"prices, statistics, dates, or figures. Acknowledge the failure honestly."
    )


#Create streamlit app:
def run_streamlit_app():
    st.title("📈 Stock Market Chatbot (SMC)")

    with st.sidebar:
        st.header('Step 1:🔑 API keys required')
        st.write('You need to provide the following API keys to use this chatbot:')
        
        user_Alpha_vantage_key = st.text_input(
            "Alpha Vantage API Key",
            value=vantage_key,
            type="password")
        st.markdown('[Get a free Alpha Vantage key](https://www.alphavantage.co/support/#api-key)')

        user_TD_api_key = st.text_input(
            "TD API Key",
            value=TD_api_key,
            type="password")
        st.markdown('[Get a free Twelve Data API key](https://twelvedata.com/pricing)')

        st.divider()
        st.header('Step 2:🤖 Download qwen3.5:9b')
        st.write('You need to download model qwen3.5:9b from ollama to have the chatbot work locally.')
        st.markdown('[Download qwen3.5:9b](https://ollama.com/library/qwen3.5:9b)')
        
        #Stops the LLM if the keys are missing:
        if not user_Alpha_vantage_key or not user_TD_api_key:
            st.info("🚫 Missing API keys. Please provide all keys to use the SMC.")
            st.stop()

    #Initialize the MarketData instance and API keys:
    if 'market_data' not in st.session_state:
        st.session_state.market_data = MarketData(
            user_Alpha_vantage_key,
            user_TD_api_key
        )

    #Chat history:
    if 'messages' not in st.session_state:
        with st.chat_message('assistant'):
            st.write("Hello! 😃 I'm your Stock Market Chatbot. How can I help you with your stock market queries today? 👌")
        st.session_state.messages = [
            {'role': 'system',
            'content': (
                f'Today is {dt.datetime.now().strftime("%Y-%m-%d %H:%M")}.\n\n'
                'You are a financial analyst assistant with tools for live prices, news, '
                'earnings transcripts, and charts. Never use memory for financial figures — '
                'always use a tool.\n\n'

                '## RULES\n'
                '- All prices, % moves, and financial figures must come from a tool call.\n'
                '- Always cite the source and date with every figure.\n'
                '- Follow SOURCE_NOTE and TOOL_FAILURE instructions exactly when they appear in tool results.\n'
                '- Never recommend buying or selling.\n\n'

                '## TOOL USAGE\n'
                '- Convert company names to tickers (Apple → AAPL).\n'
                '- Multiple tickers in one call: "AAPL,MSFT" not separate calls.\n'
                '- Charts: call get_historical_data first, then create_plot.\n'
                '- Dates: YYYY-MM-DD, calculated from today.\n\n'

                '## RESPONSE FORMAT (analysis only)\n'
                '**Summary:** [1 sentence]\n'
                '**What the data shows:** [key facts with sources]\n'
                '**What it means:** [2–3 sentences]\n'
                '**Watch for:** [1–2 risks or catalysts]\n\n'
                'Simple factual questions: answer directly. Be concise.')}
        ]

    for msg in st.session_state.messages:
        if msg['role'] in ['user', 'assistant'] and msg.get('content'):
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if msg.get('chart') is not None:
                    st.plotly_chart(msg['chart'], use_container_width=True, theme=None)

    #User input:
    if prompt := st.chat_input('Ask me anything about the stock market =)'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        #To display user messages in the chat:
        with st.chat_message('user'):
            st.markdown(prompt)
        
        #Get A.I. response:
        with st.chat_message('assistant'):
            message_placeholder = st.empty()

            with st.spinner('Analyzing data...'):
                while True:
                    response = chat(
                        model='qwen3.5:9b',
                        messages=st.session_state.messages,
                        tools=[
                            st.session_state.market_data.get_end_of_day_price,
                            st.session_state.market_data.get_news_sentiment,
                            st.session_state.market_data.get_company_info,
                            st.session_state.market_data.get_current_price,
                            st.session_state.market_data.get_specific_date_price,
                            st.session_state.market_data.get_historical_data,
                            st.session_state.market_data.create_plot,
                            st.session_state.market_data.get_earnings_call_transcript,
                            st.session_state.market_data.get_general_news,
                            st.session_state.market_data.get_breaking_news,
                            st.session_state.market_data.get_etf_profile
                        ],
                        think=True,
                        options={
                            'temperature': 0.2,
                            'top_p': 0.85,
                            'top_k': 20,
                            'min_p': 0.05,
                            'presence_penalty': 0.3, 
                            'repetition_penalty': 1.0,
                            'num_ctx': 8192,
                            'num_gpu': 99
                            }
                    )

                    #Extract message dictionary and append it to session state
                    message_dict = response.get('message', {}) if isinstance(response, dict) else response.message.model_dump()
                    st.session_state.messages.append(message_dict)

                    #Check if tools are called:
                    if message_dict.get('tool_calls'):
                        for tc in message_dict['tool_calls']:
                            function_name = tc['function']['name']
                            arguments = tc['function']['arguments']

                            #Skip get_general_news and get_breaking_news from safe_arg:
                            if function_name not in ('get_general_news', 'get_breaking_news'):
                                #forcing the argument into the function safely:
                                raw_val = arguments.get('ticker') or arguments.get('symbol') or arguments.get('tickers') or ''
                            
                                #Handle case where LLM passes a list:
                                if isinstance(raw_val, list):
                                    raw_val = raw_val[0] if len(raw_val) > 0 else ''

                                safe_arg = str(raw_val).strip().upper()

                                if not safe_arg:
                                    st.session_state.messages.append({
                                        'role': 'tool',
                                        'content': 'Error: No valid ticker symbol provided to the tool.',
                                        'tool_call_id': tc.get('id')
                                    })
                                    continue
                            
                            if function_name == 'get_general_news':
                                display_target = safe_category if 'safe_category' in locals() else 'market news'
                            elif function_name == 'get_breaking_news':
                                display_target = 'breaking news'
                            else:
                                display_target = safe_arg
                            st.toast(f"⚙️ Fetching data from {function_name} for {display_target}...")

                            try:
                                if function_name == 'get_end_of_day_price':
                                    result = st.session_state.market_data.get_end_of_day_price(safe_arg)
                                    result_str = str(result)
                                    if _is_tool_error(result_str):
                                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                                    else:
                                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
                                #Format the news to save tokens:
                                elif function_name == 'get_news_sentiment':
                                    result = st.session_state.market_data.get_news_sentiment(safe_arg)
                                    if isinstance(result, list):
                                        formatted_news = []
                                        for i, r in enumerate(result, 1):
                                            title = r.get("title", "No title")
                                            sentiment = r.get("overall_sentiment", "Unknown")
                                            date = r.get("date", "Unknown")
                                            source = r.get("source", "Unknown")
                                            url = r.get("url", "No link available")

                                            article_str = (
                                                f"[{i}] {title}\n"
                                                f" (Source: {source} | Sentiment: {sentiment} | Date: {date})\n"
                                                f" URL: {url}\n"
                                            )
                                            formatted_news.append(article_str)
                                        result_str = "\n\n".join(formatted_news) if formatted_news else "No news found."
                                    else:
                                        result_str = str(result)
                                        if _is_tool_error(result_str):
                                            result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                                elif function_name == 'get_company_info':
                                    result = st.session_state.market_data.get_company_info(safe_arg)
                                    result_str = str(result)
                                    if _is_tool_error(result_str):
                                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                                    else:
                                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
                                elif function_name == 'get_current_price':
                                    result = st.session_state.market_data.get_current_price(safe_arg)
                                    result_str = str(result)
                                    if _is_tool_error(result_str):
                                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                                    else:
                                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
                                elif function_name == 'get_specific_date_price':
                                    date = arguments.get('date') or arguments.get('datetime') or None
                                    result = st.session_state.market_data.get_specific_date_price(safe_arg, date)
                                    result_str = str(result)
                                    if _is_tool_error(result_str):
                                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                                    else:
                                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
                                elif function_name == 'get_historical_data':
                                    #safeguard the interval output:
                                    raw_interval = arguments.get('interval') or '1day'
                                    safe_interval = str(raw_interval).lower().replace(' ', '')
                                    
                                    start_date = arguments.get('start_date')
                                    end_date = arguments.get('end_date')
                                    result_str = st.session_state.market_data.get_historical_data(
                                        safe_arg, safe_interval, start_date, end_date
                                    )
                                    
                                    if not result_str or result_str.strip().startswith("Error") or result_str == "No data retrieved.":
                                        result_str = (
                                            f"TOOL_FAILURE: get_historical_data could not retrieve data for {safe_arg}. "
                                            f"Details: {result_str}. "
                                            f"You MUST inform the user that the data could not be fetched. "
                                            f"Do NOT use your training data to estimate or fill in any prices, dates, or statistics."
                                        )
                                    else:
                                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
                                elif function_name == 'create_plot':
                                    result = st.session_state.market_data.create_plot(safe_arg)
                                    result_str = str(result)
                                elif function_name == 'get_earnings_call_transcript':
                                    #Ensures the quarter parameter is passed:
                                    raw_quarter = arguments.get('quarter', '')
                                    safe_quarter = str(raw_quarter).strip().upper()
                                    if not safe_quarter:
                                        result_str = 'Error: No quarter provided. Please specify a quarter (e.g., 2024Q1)'
                                    else:
                                        result = st.session_state.market_data.get_earnings_call_transcript(safe_arg, safe_quarter)
                                        result_str = str(result)
                                elif function_name == 'get_general_news':
                                    raw_category = arguments.get('topics') or arguments.get('topic') or 'financial_markets'
                                    safe_category = str(raw_category).lower().strip()
                                    
                                    allowed_categories = [
                                        'blockchain', 'earnings', 'ipo', 'mergers_and_acquisitions',
                                        'financial_markets', 'economy_fiscal', 'economy_monetary',
                                        'economy_macro', 'energy_transportation', 'finance',
                                        'life_sciences', 'manufacturing', 'real_estate',
                                        'retail_wholesale', 'technology'
                                    ]
                                    
                                    if safe_category not in allowed_categories:
                                        safe_category = 'financial_markets'
                                    
                                    keywords = arguments.get('keywords') or arguments.get('search') or None

                                    result = st.session_state.market_data.get_general_news(safe_category, keywords)
                                    
                                    if isinstance(result, list):
                                        formatted_articles = []
                                        for i, article in enumerate(result, 1):
                                            article_str = (
                                                f"[{i}] {article.get('title', 'No title')}\n"
                                                f"Source: {article.get('source', 'Unknown')}\n"
                                                f"Published: {article.get('published_at', 'Unknown')}\n"
                                                f"Sentiment: {article.get('overall_sentiment', 'N/A')}\n"
                                                f"Summary: {article.get('summary', 'No summary')}\n"
                                                f"URL: {article.get('url', 'No URL')}\n"
                                                f"----\n"
                                            )
                                            formatted_articles.append(article_str)
                                        result_str = "\n\n".join(formatted_articles) if formatted_articles else "No articles found."
                                    else:
                                        result_str = str(result)
                                        if _is_tool_error(result_str):
                                            result_str = _wrap_tool_failure(function_name, safe_category, result_str)

                                elif function_name == 'get_breaking_news':
                                    keywords = arguments.get('keywords') or arguments.get('search') or None
                                    result = st.session_state.market_data.get_breaking_news(keywords)
                                    
                                    if isinstance(result, list):
                                        formatted_articles = []
                                        for i, article in enumerate(result, 1):
                                            article_str = (
                                                f"[{i}. {article.get('title', 'No title')}]\n"
                                                f"Source: {article.get('source', 'Unknown')}\n"
                                                f"Published: {article.get('published_at', '')}\n"
                                                f"Summary: {article.get('summary', 'No summary')}\n"
                                                f"URL: {article.get('url', 'No URL')}\n"
                                                f"----\n"
                                            )
                                            formatted_articles.append(article_str)
                                        result_str = "\n\n".join(formatted_articles) if formatted_articles else "No articles found."
                                    else:
                                        result_str = str(result)
                                        if _is_tool_error(result_str):
                                            result_str = _wrap_tool_failure(function_name, keywords, result_str)
                                elif function_name == 'get_etf_profile':
                                    ticker_list = [t.strip().upper() for t in safe_arg.split(',')]
    
                                    if len(ticker_list) > 1:
                                        st.toast(f"⚙️ Fetching {len(ticker_list)} ETF profiles — this may take ~{12*(len(ticker_list)-1)}s due to API limits...")

                                    result = st.session_state.market_data.get_etf_profile(safe_arg)

                                    if isinstance(result, dict):
                                        # Multiple ETFs returned
                                        if all(isinstance(v, dict) and 'net_assets' in v for v in result.values()):
                                            profiles = []
                                            for t, data in result.items():
                                                holdings_str = "\n".join([
                                                    f"  {i+1}. {h['symbol']} ({h['description']}): {float(h['weight'])*100:.2f}%"
                                                    for i, h in enumerate(data.get('top_holdings', []))
                                                ])
                                                sectors_str = "\n".join([
                                                    f"  {s['sector']}: {float(s['weight'])*100:.1f}%"
                                                    for s in data.get('sectors', [])
                                                ])
                                                profiles.append(
                                                    f"ETF Profile for {t}:\n"
                                                    f"Net Assets: ${int(data['net_assets']):,}\n"
                                                    f"Expense Ratio: {float(data['net_expense_ratio'])*100:.2f}%\n"
                                                    f"Dividend Yield: {float(data['dividend_yield'])*100:.2f}%\n"
                                                    f"Portfolio Turnover: {float(data['portfolio_turnover'])*100:.1f}%\n"
                                                    f"Inception Date: {data['inception_date']}\n"
                                                    f"Leveraged: {data['leveraged']}\n\n"
                                                    f"Top 10 Holdings:\n{holdings_str}\n\n"
                                                    f"Sector Allocation:\n{sectors_str}\n"
                                                )
                                            result_str = "\n\n---\n\n".join(profiles)
                                            
        
                                        # Single ETF returned
                                        elif 'net_assets' in result:
                                            holdings_str = "\n".join([
                                                f"  {i+1}. {h['symbol']} ({h['description']}): {float(h['weight'])*100:.2f}%"
                                                for i, h in enumerate(result.get('top_holdings', []))
                                            ])
                                            sectors_str = "\n".join([
                                                f"  {s['sector']}: {float(s['weight'])*100:.1f}%"
                                                for s in result.get('sectors', [])
                                            ])
                                            result_str = (
                                                f"ETF Profile for {result['ticker']}:\n"
                                                f"Net Assets: ${int(result['net_assets']):,}\n"
                                                f"Expense Ratio: {float(result['net_expense_ratio'])*100:.2f}%\n"
                                                f"Dividend Yield: {float(result['dividend_yield'])*100:.2f}%\n"
                                                f"Portfolio Turnover: {float(result['portfolio_turnover'])*100:.1f}%\n"
                                                f"Inception Date: {result['inception_date']}\n"
                                                f"Leveraged: {result['leveraged']}\n\n"
                                                f"Top 10 Holdings:\n{holdings_str}\n\n"
                                                f"Sector Allocation:\n{sectors_str}\n"
                                                f"\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
                                            )
                                        else:
                                            result_str = str(result)
                                            if _is_tool_error(result_str):
                                                result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                                    else:  
                                        result_str = str(result)
                                        if _is_tool_error(result_str):
                                            result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                                else:
                                    result_str = f"Error: Unknown function called: {function_name}"

                            except Exception as e:
                                result_str = f'Error: {str(e)}'

                            st.session_state.messages.append({
                                'role': 'tool',
                                'content': result_str,
                                'tool_call_id': tc.get('id')
                            })

                    else:
                        final_text = message_dict.get('content', '')
                        fig_to_show = getattr(st.session_state.market_data, '_last_fig', None)
                        if final_text:
                            message_placeholder.markdown(final_text)
                        if fig_to_show is not None:
                            st.plotly_chart(fig_to_show, use_container_width=True, theme=None)
                            st.session_state.messages[-1]['chart'] = fig_to_show
                            st.session_state.market_data._last_fig = None
                        break


if __name__ == '__main__':
    run_streamlit_app()
#==================================================================
#The following code is for evaluation purposes only.
#It is not intended for production use.


def get_smc_response(question, market_data_instance):
    """
    Standalone eval harness for RAGAS.
    Runs the full agentic tool-call loop against market_data_instance
    and returns the final text answer plus all retrieved tool contexts.

    Returns:
        dict: {
            'response': str,               # final LLM answer
            'retrieved_contexts': list[str] # one entry per tool call made
        }
    """

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #
    def _is_tool_error(s: str) -> bool:
        return any(
            s.strip().startswith(p)
            for p in ("Error", "error", "None", "No data", "[]", "{}")
        )

    def _wrap_tool_failure(fn: str, target: str, detail: str) -> str:
        return (
            f"TOOL_FAILURE: {fn} could not retrieve data for {target}. "
            f"Details: {detail}. "
            f"You MUST inform the user that the data could not be fetched. "
            f"Do NOT use your training data to estimate or fill in any figures."
        )

    # ------------------------------------------------------------------ #
    # system prompt                                                      #
    # ------------------------------------------------------------------ #
    messages = [
        {
            "role": "system",
            "content": (
                f"Today is {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}.\n\n"
                "You are a financial analyst assistant with tools for live prices, news, "
                "earnings transcripts, and charts. Never use memory for financial figures — "
                "always use a tool.\n\n"
                "## RULES\n"
                "- All prices, % moves, and financial figures must come from a tool call.\n"
                "- Always cite the source and date with every figure.\n"
                "- Follow SOURCE_NOTE and TOOL_FAILURE instructions exactly when they appear.\n"
                "- Never recommend buying or selling.\n\n"
                "## TOOL USAGE\n"
                "- Convert company names to tickers (Apple → AAPL).\n"
                "- Multiple tickers in one call: 'AAPL,MSFT' not separate calls.\n"
                "- Charts: call get_historical_data first, then create_plot.\n"
                "- Dates: YYYY-MM-DD, calculated from today.\n\n"
                "## RESPONSE FORMAT (analysis only)\n"
                "**Summary:** [1 sentence]\n"
                "**What the data shows:** [key facts with sources]\n"
                "**What it means:** [2–3 sentences]\n"
                "**Watch for:** [1–2 risks or catalysts]\n\n"
                "Simple factual questions: answer directly. Be concise."
            ),
        }
    ]
    messages.append({"role": "user", "content": question})

    context_accumulator: list[str] = []
    final_answer = ""
    max_iterations = 10

    context_accumulator: list[str] = []
    actual_tool_calls: list[dict] = []
    final_answer = ""
    max_iterations = 10

    # ------------------------------------------------------------------ #
    # agentic loop                                                       #
    # ------------------------------------------------------------------ #
    for iteration in range(max_iterations):
        response = chat(
            model="qwen3.5:9b",
            messages=messages,
            tools=[
                market_data_instance.get_end_of_day_price,
                market_data_instance.get_news_sentiment,
                market_data_instance.get_company_info,
                market_data_instance.get_current_price,
                market_data_instance.get_specific_date_price,
                market_data_instance.get_historical_data,
                market_data_instance.create_plot,
                market_data_instance.get_earnings_call_transcript,
                market_data_instance.get_general_news,
                market_data_instance.get_breaking_news,
                market_data_instance.get_etf_profile,
            ],
            options={
                "temperature": 0.2,
                "top_p": 0.85,
                "top_k": 20,
                "min_p": 0.05,
                "presence_penalty": 0.3,
                "repetition_penalty": 1.0,
                "num_ctx": 8192,
                "num_gpu": 99
            },
        )

        message_dict = (
            response.get("message", {})
            if isinstance(response, dict)
            else response.message.model_dump()
        )
        messages.append(message_dict)

        # ── No tool calls → this is the final answer ──────────────────── #
        if not message_dict.get("tool_calls"):
            final_answer = message_dict.get("content", "")
            break

        # ── Process every tool call in this turn ──────────────────────── #
        for tc in message_dict["tool_calls"]:
            function_name = tc["function"]["name"]
            arguments = tc["function"]["arguments"]
            result_str = ""

            #Capture the tool call for evaluation
            actual_tool_calls.append({
                "name": function_name,
                "arguments": arguments
            })

            # Resolve safe ticker argument (not needed for news-only calls) #
            safe_arg = ""
            if function_name not in ("get_general_news", "get_breaking_news"):
                raw_val = (
                    arguments.get("ticker")
                    or arguments.get("symbol")
                    or arguments.get("tickers")
                    or ""
                )
                if isinstance(raw_val, list):
                    raw_val = raw_val[0] if raw_val else ""
                safe_arg = str(raw_val).strip().upper()

                if not safe_arg:
                    result_str = "Error: No valid ticker symbol provided to the tool."
                    messages.append(
                        {
                            "role": "tool",
                            "content": result_str,
                            "tool_call_id": tc.get("id"),
                        }
                    )
                    context_accumulator.append(
                        f"[{function_name}()] {result_str}"
                    )
                    continue

            # ── Dispatch ─────────────────────────────────────────────── #
            try:
                if function_name == "get_end_of_day_price":
                    result = market_data_instance.get_end_of_day_price(safe_arg)
                    result_str = str(result)
                    if _is_tool_error(result_str):
                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."

                elif function_name == "get_news_sentiment":
                    result = market_data_instance.get_news_sentiment(safe_arg)
                    if isinstance(result, list):
                        parts = []
                        for i, r in enumerate(result, 1):
                            parts.append(
                                f"[{i}] {r.get('title', 'No title')}\n"
                                f" (Source: {r.get('source','Unknown')} | "
                                f"Sentiment: {r.get('overall_sentiment','Unknown')} | "
                                f"Date: {r.get('date','Unknown')})\n"
                                f" URL: {r.get('url','No link available')}\n"
                            )
                        result_str = "\n\n".join(parts) if parts else "No news found."
                    else:
                        result_str = str(result)
                        if _is_tool_error(result_str):
                            result_str = _wrap_tool_failure(function_name, safe_arg, result_str)

                elif function_name == "get_company_info":
                    result = market_data_instance.get_company_info(safe_arg)
                    result_str = str(result)
                    if _is_tool_error(result_str):
                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."

                elif function_name == "get_current_price":
                    result = market_data_instance.get_current_price(safe_arg)
                    result_str = str(result)
                    if _is_tool_error(result_str):
                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."

                elif function_name == "get_specific_date_price":
                    date = arguments.get("date") or arguments.get("datetime") or None
                    result = market_data_instance.get_specific_date_price(safe_arg, date)
                    result_str = str(result)
                    if _is_tool_error(result_str):
                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."

                elif function_name == "get_historical_data":
                    raw_interval = arguments.get("interval") or "1day"
                    safe_interval = str(raw_interval).lower().replace(" ", "")
                    start_date = arguments.get("start_date")
                    end_date = arguments.get("end_date")
                    result_str = market_data_instance.get_historical_data(
                        safe_arg, safe_interval, start_date, end_date
                    )
                    if (
                        not result_str
                        or result_str.strip().startswith("Error")
                        or result_str == "No data retrieved."
                    ):
                        result_str = _wrap_tool_failure(
                            function_name, safe_arg, result_str
                        )
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."

                elif function_name == "create_plot":
                    # In eval context we skip rendering; just log the call.
                    result = market_data_instance.create_plot(safe_arg)
                    result_str = f"[create_plot called for {safe_arg}; chart skipped in eval mode]"

                elif function_name == "get_earnings_call_transcript":
                    raw_quarter = arguments.get("quarter", "")
                    safe_quarter = str(raw_quarter).strip().upper()
                    if not safe_quarter:
                        result_str = "Error: No quarter provided. Specify a quarter (e.g. 2024Q1)."
                    else:
                        result = market_data_instance.get_earnings_call_transcript(
                            safe_arg, safe_quarter
                        )
                        result_str = str(result)

                elif function_name == "get_general_news":
                    raw_category = (
                        arguments.get("topics")
                        or arguments.get("topic")
                        or "financial_markets"
                    )
                    allowed = {
                        "blockchain", "earnings", "ipo", "mergers_and_acquisitions",
                        "financial_markets", "economy_fiscal", "economy_monetary",
                        "economy_macro", "energy_transportation", "finance",
                        "life_sciences", "manufacturing", "real_estate",
                        "retail_wholesale", "technology",
                    }
                    safe_category = str(raw_category).lower().strip()
                    if safe_category not in allowed:
                        safe_category = "financial_markets"
                    keywords = arguments.get("keywords") or arguments.get("search") or None
                    result = market_data_instance.get_general_news(safe_category, keywords)
                    safe_arg = safe_category  # use for context log below
                    if isinstance(result, list):
                        parts = []
                        for i, article in enumerate(result, 1):
                            parts.append(
                                f"[{i}] {article.get('title','No title')}\n"
                                f"Source: {article.get('source','Unknown')}\n"
                                f"Published: {article.get('published_at','Unknown')}\n"
                                f"Sentiment: {article.get('overall_sentiment','N/A')}\n"
                                f"Summary: {article.get('summary','No summary')}\n"
                                f"URL: {article.get('url','No URL')}\n----"
                            )
                        result_str = "\n\n".join(parts) if parts else "No articles found."
                    else:
                        result_str = str(result)
                        if _is_tool_error(result_str):
                            result_str = _wrap_tool_failure(function_name, safe_category, result_str)

                elif function_name == "get_breaking_news":
                    keywords = arguments.get("keywords") or arguments.get("search") or None
                    result = market_data_instance.get_breaking_news(keywords)
                    safe_arg = "breaking_news"  # use for context log below
                    if isinstance(result, list):
                        parts = []
                        for i, article in enumerate(result, 1):
                            parts.append(
                                f"[{i}. {article.get('title','No title')}]\n"
                                f"Source: {article.get('source','Unknown')}\n"
                                f"Published: {article.get('published_at','')}\n"
                                f"Summary: {article.get('summary','No summary')}\n"
                                f"URL: {article.get('url','No URL')}\n----"
                            )
                        result_str = "\n\n".join(parts) if parts else "No articles found."
                    else:
                        result_str = str(result)
                        if _is_tool_error(result_str):
                            result_str = _wrap_tool_failure(function_name, str(keywords), result_str)

                elif function_name == "get_etf_profile":
                    result = market_data_instance.get_etf_profile(safe_arg)

                    def _format_etf(ticker, data):
                        holdings = "\n".join(
                            f"  {i+1}. {h['symbol']} ({h['description']}): "
                            f"{float(h['weight'])*100:.2f}%"
                            for i, h in enumerate(data.get("top_holdings", []))
                        )
                        sectors = "\n".join(
                            f"  {s['sector']}: {float(s['weight'])*100:.1f}%"
                            for s in data.get("sectors", [])
                        )
                        return (
                            f"ETF Profile for {ticker}:\n"
                            f"Net Assets: ${int(data['net_assets']):,}\n"
                            f"Expense Ratio: {float(data['net_expense_ratio'])*100:.2f}%\n"
                            f"Dividend Yield: {float(data['dividend_yield'])*100:.2f}%\n"
                            f"Portfolio Turnover: {float(data['portfolio_turnover'])*100:.1f}%\n"
                            f"Inception Date: {data['inception_date']}\n"
                            f"Leveraged: {data['leveraged']}\n\n"
                            f"Top 10 Holdings:\n{holdings}\n\n"
                            f"Sector Allocation:\n{sectors}\n"
                        )

                    if isinstance(result, dict):
                        # Multiple ETFs: values are dicts with 'net_assets'
                        if all(
                            isinstance(v, dict) and "net_assets" in v
                            for v in result.values()
                        ):
                            profiles = [
                                _format_etf(t, d) for t, d in result.items()
                            ]
                            result_str = "\n\n---\n\n".join(profiles)
                        # Single ETF: dict has 'net_assets' at top level
                        elif "net_assets" in result:
                            result_str = _format_etf(result.get("ticker", safe_arg), result)
                            result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
                        else:
                            result_str = str(result)
                            if _is_tool_error(result_str):
                                result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str = str(result)
                        if _is_tool_error(result_str):
                            result_str = _wrap_tool_failure(function_name, safe_arg, result_str)

                else:
                    result_str = f"Error: Unknown function '{function_name}'."

            except Exception as exc:
                result_str = f"Error: {exc}"

            # ── Accumulate context for RAGAS ──────────────────────────── #
            context_accumulator.append(
                f"[{function_name}({safe_arg})] {result_str[:500]}"
            )

            # ── Feed result back into the conversation ─────────────────── #
            messages.append(
                {
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tc.get("id"),
                }
            )

        # After processing all tool calls in this turn, loop back to chat()

    return {
        "response": final_answer,
        "retrieved_contexts": context_accumulator if context_accumulator else ["No tools called"],
        "actual_tool_calls": actual_tool_calls
    }
#======================================================================
#This section is for generating responses using DeepSeek as the engine
#======================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_end_of_day_price",
            "description": "Fetch the end of day price of a stock using Alpha Vantage API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock symbol, e.g. AAPL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_price",
            "description": "Get the real-time stock price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock symbol(s), comma-separated for multiple e.g. 'AAPL,MSFT'"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_specific_date_price",
            "description": (
                "Fetch a stock's closing price on a specific date or the most recent end-of-day price. "
                "Use for: 'closing price yesterday', 'what did X close at on [date]', 'end of day price for [date]'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock symbol, e.g. AAPL"},
                    "date":   {"type": "string", "description": "Date in YYYY-MM-DD format. Omit for most recent EOD."}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_data",
            "description": (
                "Retrieves historical time-series pricing. Use when the user asks for historical data, "
                "past stock prices, or wants to calculate historical ROI or past performance. "
                "Supports multiple tickers as a comma-separated string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker":     {"type": "string", "description": "Stock symbol(s), comma-separated e.g. 'AAPL,MSFT'"},
                    "interval":   {"type": "string", "description": "Time interval: 1min, 1day, 1week, 1month. Default: 1day"},
                    "start_date": {"type": "string", "description": "Start date in 'YYYY-MM-DD 00:00:00' format"},
                    "end_date":   {"type": "string", "description": "End date in 'YYYY-MM-DD 00:00:00' format"}
                },
                "required": ["ticker", "interval"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_plot",
            "description": (
                "Creates an interactive time series plot of historical data already in memory. "
                "ONLY call this AFTER get_historical_data has been called. "
                "Pass only the ticker symbol string — never pass data arrays."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock symbol(s) to plot, comma-separated e.g. 'AAPL,MSFT'"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": (
                "Fetches sentiment scores and recent news for a SPECIFIC company. "
                "Use ONLY when the user asks about news or sentiment for a specific stock ticker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock symbol, e.g. AAPL"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Fetches general corporate information: sector, industry, CEO, market cap, description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock symbol(s), comma-separated"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_earnings_call_transcript",
            "description": (
                "Returns the earnings call transcript for a given company in a specific quarter. "
                "If user does not specify a quarter, use the most recent available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker":  {"type": "string", "description": "The stock symbol, e.g. AAPL"},
                    "quarter": {"type": "string", "description": "Fiscal quarter in YYYYQM format, e.g. 2024Q1"}
                },
                "required": ["ticker", "quarter"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_general_news",
            "description": (
                "Fetches FINANCIAL MARKET NEWS with sentiment from analyst and finance sources. "
                "Use for sector trends, economic indicators, and industry-wide financial developments. "
                "DO NOT use for breaking world events or company-specific news."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "string",
                        "description": (
                            "Market topic: blockchain, earnings, ipo, mergers_and_acquisitions, "
                            "financial_markets, economy_fiscal, economy_monetary, economy_macro, "
                            "energy_transportation, finance, life_sciences, manufacturing, "
                            "real_estate, retail_wholesale, technology"
                        )
                    },
                    "keywords": {"type": "string", "description": "Optional keywords to narrow the search"}
                },
                "required": ["topics"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_breaking_news",
            "description": (
                "Fetches BREAKING NEWS from premium journalism sources (Reuters, BBC, WSJ, FT). "
                "Use ONLY for active geopolitical events, ongoing crises, or major policy announcements. "
                "DO NOT use for stock analysis or company-specific news."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Keywords to filter news e.g. 'Iran oil shipping'"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_etf_profile",
            "description": (
                "Fetches ETF profile: index tracked, expense ratio, holdings, and asset allocation. "
                "ALWAYS use this before making any claims about an ETF."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "ETF ticker symbol e.g. VOO, QQQ, SPY"}
                },
                "required": ["ticker"]
            }
        }
    }
]
 
 
# ── Helper: check for tool errors ─────────────────────────────────────────────
def _is_tool_error(s: str) -> bool:
    return any(
        s.strip().lower().startswith(p)
        for p in ("error", "none", "no data", "[]", "{}", "tool_failure")
    )
 
 
def _wrap_tool_failure(fn: str, target: str, detail: str) -> str:
    return (
        f"TOOL_FAILURE: {fn} could not retrieve data for {target}. "
        f"Details: {detail}. "
        f"You MUST inform the user that the data could not be fetched. "
        f"Do NOT use your training data to estimate or fill in any figures."
    )
 
 
# ── Main eval function ─────────────────────────────────────────────────────────
def get_deepseek_smc_response(question: str, market_data_instance, deepseek_api_key: str) -> dict:
    """
    Eval harness for deepseek_smc_0_4.py. Runs the full agentic tool-call loop via DeepSeek
    and returns the final text answer plus all retrieved tool contexts.
 
    Args:
        question:              The user's question string.
        market_data_instance:  An initialised MarketData instance.
        deepseek_api_key:      Your DeepSeek API key.
 
    Returns:
        {
            'response':          str,        # final LLM answer
            'retrieved_contexts': list[str], # one entry per tool call made
            'actual_tool_calls':  list[dict] # raw tool call records
        }
    """
 
    client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
 
    messages = [
        {
            "role": "system",
            "content": (
                f"Today is {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}.\n\n"
                "You are a financial analyst assistant with tools for live prices, news, "
                "earnings transcripts, and charts. Never use memory for financial figures — "
                "always use a tool.\n\n"
                "## RULES\n"
                "- All prices, % moves, and financial figures must come from a tool call.\n"
                "- Always cite the source and date with every figure.\n"
                "- Follow SOURCE_NOTE and TOOL_FAILURE instructions exactly when they appear.\n"
                "- Never recommend buying or selling.\n\n"
                "## TOOL USAGE\n"
                "- Convert company names to tickers (Apple → AAPL).\n"
                "- Multiple tickers in one call: 'AAPL,MSFT' not separate calls.\n"
                "- Charts: call get_historical_data first, then create_plot.\n"
                "- Dates: YYYY-MM-DD, calculated from today.\n\n"
                "## RESPONSE FORMAT (analysis only)\n"
                "**Summary:** [1 sentence]\n"
                "**What the data shows:** [key facts with sources]\n"
                "**What it means:** [2–3 sentences]\n"
                "**Watch for:** [1–2 risks or catalysts]\n\n"
                "Simple factual questions: answer directly. Be concise."
            ),
        },
        {"role": "user", "content": question}
    ]
 
    context_accumulator: list[str] = []
    actual_tool_calls:   list[dict] = []
    final_answer = ""
    max_iterations = 10
 
    # ── Agentic loop ───────────────────────────────────────────────────────── #
    for iteration in range(max_iterations):
 
        response = client.chat.completions.create(
            model="deepseek-chat",   
            messages=messages,       
            tools=TOOL_DEFINITIONS,
            tool_choice="auto"
        )
 
        assistant_message = response.choices[0].message
        message_dict = assistant_message.model_dump()
        messages.append(message_dict)
 
        # ── No tool calls → final answer ──────────────────────────────────── #
        if not assistant_message.tool_calls:
            final_answer = assistant_message.content or ""
            break
 
        # ── Process each tool call ────────────────────────────────────────── #
        for tc in assistant_message.tool_calls:
            function_name = tc.function.name
            result_str    = ""
 
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
 
            actual_tool_calls.append({
                "name":      function_name,
                "arguments": arguments
            })
 
            safe_arg = ""
            if function_name not in ("get_general_news", "get_breaking_news"):
                raw_val = (
                    arguments.get("ticker")
                    or arguments.get("symbol")
                    or arguments.get("tickers")
                    or ""
                )
                if isinstance(raw_val, list):
                    raw_val = raw_val[0] if raw_val else ""
                safe_arg = str(raw_val).strip().upper()
 
                if not safe_arg:
                    result_str = "Error: No valid ticker symbol provided to the tool."
                    messages.append({
                        "role":         "tool",
                        "content":      result_str,
                        "tool_call_id": tc.id  # tc.id not tc.get("id") on SDK objects
                    })
                    context_accumulator.append(f"[{function_name}()] {result_str}")
                    continue
 
            # ── Dispatch ───────────────────────────────────────────────────── #
            try:
                if function_name == "get_end_of_day_price":
                    result = market_data_instance.get_end_of_day_price(safe_arg)
                    result_str = str(result)
                    if _is_tool_error(result_str):
                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
 
                elif function_name == "get_current_price":
                    result = market_data_instance.get_current_price(safe_arg)
                    result_str = str(result)
                    if _is_tool_error(result_str):
                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
 
                elif function_name == "get_specific_date_price":
                    date = arguments.get("date") or arguments.get("datetime") or None
                    result = market_data_instance.get_specific_date_price(safe_arg, date)
                    result_str = str(result)
                    if _is_tool_error(result_str):
                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
 
                elif function_name == "get_historical_data":
                    raw_interval = arguments.get("interval") or "1day"
                    safe_interval = str(raw_interval).lower().replace(" ", "")
                    start_date = arguments.get("start_date")
                    end_date   = arguments.get("end_date")
                    result_str = market_data_instance.get_historical_data(
                        safe_arg, safe_interval, start_date, end_date
                    )
                    if (
                        not result_str
                        or result_str.strip().startswith("Error")
                        or result_str == "No data retrieved."
                    ):
                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
 
                elif function_name == "create_plot":
                    market_data_instance.create_plot(safe_arg)
                    result_str = f"[create_plot called for {safe_arg}; chart skipped in eval mode]"
 
                elif function_name == "get_news_sentiment":
                    result = market_data_instance.get_news_sentiment(safe_arg)
                    if isinstance(result, list):
                        parts = [
                            f"[{i}] {r.get('title', 'No title')}\n"
                            f" (Source: {r.get('source','Unknown')} | "
                            f"Sentiment: {r.get('overall_sentiment','Unknown')} | "
                            f"Date: {r.get('date','Unknown')})\n"
                            f" URL: {r.get('url','No link available')}"
                            for i, r in enumerate(result, 1)
                        ]
                        result_str = "\n\n".join(parts) if parts else "No news found."
                    else:
                        result_str = str(result)
                        if _is_tool_error(result_str):
                            result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
 
                elif function_name == "get_company_info":
                    result = market_data_instance.get_company_info(safe_arg)
                    result_str = str(result)
                    if _is_tool_error(result_str):
                        result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
 
                elif function_name == "get_earnings_call_transcript":
                    raw_quarter = arguments.get("quarter", "")
                    safe_quarter = str(raw_quarter).strip().upper()
                    if not safe_quarter:
                        result_str = "Error: No quarter provided. Specify a quarter (e.g. 2024Q1)."
                    else:
                        result = market_data_instance.get_earnings_call_transcript(safe_arg, safe_quarter)
                        result_str = str(result)
 
                elif function_name == "get_general_news":
                    allowed = {
                        "blockchain", "earnings", "ipo", "mergers_and_acquisitions",
                        "financial_markets", "economy_fiscal", "economy_monetary",
                        "economy_macro", "energy_transportation", "finance",
                        "life_sciences", "manufacturing", "real_estate",
                        "retail_wholesale", "technology",
                    }
                    raw_category = arguments.get("topics") or arguments.get("topic") or "financial_markets"
                    safe_category = str(raw_category).lower().strip()
                    if safe_category not in allowed:
                        safe_category = "financial_markets"
                    keywords = arguments.get("keywords") or arguments.get("search") or None
                    result = market_data_instance.get_general_news(safe_category, keywords)
                    safe_arg = safe_category
                    if isinstance(result, list):
                        parts = [
                            f"[{i}] {a.get('title','No title')}\n"
                            f"Source: {a.get('source','Unknown')}\n"
                            f"Published: {a.get('published_at','Unknown')}\n"
                            f"Sentiment: {a.get('overall_sentiment','N/A')}\n"
                            f"Summary: {a.get('summary','No summary')}\n"
                            f"URL: {a.get('url','No URL')}\n----"
                            for i, a in enumerate(result, 1)
                        ]
                        result_str = "\n\n".join(parts) if parts else "No articles found."
                    else:
                        result_str = str(result)
                        if _is_tool_error(result_str):
                            result_str = _wrap_tool_failure(function_name, safe_category, result_str)
 
                elif function_name == "get_breaking_news":
                    keywords = arguments.get("keywords") or arguments.get("search") or None
                    result = market_data_instance.get_breaking_news(keywords)
                    safe_arg = "breaking_news"
                    if isinstance(result, list):
                        parts = [
                            f"[{i}. {a.get('title','No title')}]\n"
                            f"Source: {a.get('source','Unknown')}\n"
                            f"Published: {a.get('published_at','')}\n"
                            f"Summary: {a.get('summary','No summary')}\n"
                            f"URL: {a.get('url','No URL')}\n----"
                            for i, a in enumerate(result, 1)
                        ]
                        result_str = "\n\n".join(parts) if parts else "No articles found."
                    else:
                        result_str = str(result)
                        if _is_tool_error(result_str):
                            result_str = _wrap_tool_failure(function_name, str(keywords), result_str)
 
                elif function_name == "get_etf_profile":
                    result = market_data_instance.get_etf_profile(safe_arg)
 
                    def _format_etf(tkr, data):
                        holdings = "\n".join(
                            f"  {i+1}. {h['symbol']} ({h['description']}): {float(h['weight'])*100:.2f}%"
                            for i, h in enumerate(data.get("top_holdings", []))
                        )
                        sectors = "\n".join(
                            f"  {s['sector']}: {float(s['weight'])*100:.1f}%"
                            for s in data.get("sectors", [])
                        )
                        return (
                            f"ETF Profile for {tkr}:\n"
                            f"Net Assets: ${int(data['net_assets']):,}\n"
                            f"Expense Ratio: {float(data['net_expense_ratio'])*100:.2f}%\n"
                            f"Dividend Yield: {float(data['dividend_yield'])*100:.2f}%\n"
                            f"Portfolio Turnover: {float(data['portfolio_turnover'])*100:.1f}%\n"
                            f"Inception Date: {data['inception_date']}\n"
                            f"Leveraged: {data['leveraged']}\n\n"
                            f"Top 10 Holdings:\n{holdings}\n\n"
                            f"Sector Allocation:\n{sectors}\n"
                        )
 
                    if isinstance(result, dict):
                        if all(isinstance(v, dict) and "net_assets" in v for v in result.values()):
                            result_str = "\n\n---\n\n".join(
                                _format_etf(t, d) for t, d in result.items()
                            )
                        elif "net_assets" in result:
                            result_str = _format_etf(result.get("ticker", safe_arg), result)
                            result_str += "\nSOURCE_NOTE: No URL to cite. Do NOT fabricate a source link."
                        else:
                            result_str = str(result)
                            if _is_tool_error(result_str):
                                result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
                    else:
                        result_str = str(result)
                        if _is_tool_error(result_str):
                            result_str = _wrap_tool_failure(function_name, safe_arg, result_str)
 
                else:
                    result_str = f"Error: Unknown function '{function_name}'."
 
            except Exception as exc:
                result_str = f"Error: {exc}"
 
            context_accumulator.append(f"[{function_name}({safe_arg})] {result_str[:500]}")
 
            messages.append({
                "role":         "tool",
                "content":      result_str,
                "tool_call_id": tc.id   
            })
 
    return {
        "response":           final_answer,
        "retrieved_contexts": context_accumulator if context_accumulator else ["No tools called"],
        "actual_tool_calls":  actual_tool_calls
    }

