import os
import yfinance as yf
from dotenv import load_dotenv
import requests
import datetime as dt
from ollama import chat
import pandas as pd
import plotly.express as px
import streamlit as st



# Load the API key from the local environment file
load_dotenv("vantage_key.env")
vantage_key = os.getenv("vantage_key", "")

load_dotenv("TD_api_key.env")
TD_api_key = os.getenv("TD_api_key", "")

#Define SMC class:
class MarketData:
    def __init__(self, AlphaVantage_api_key, TD_api_key):
        self.AlphaVantage_api_key = AlphaVantage_api_key
        self.TD_api_key = TD_api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.base_url_2 = "https://api.twelvedata.com/"
        self.base_url_3 = "https://api.twelvedata.com/price"
        self.base_url_4 = "https://api.twelvedata.com/time_series"
    
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
            'apikey': self.AlphaVantage_api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            #Check for API limits or errors:
            if 'Global Quote' not in data:
                print("API Limit Reached. Please wait.")
                print("Using Twelve Data API as a backup...")
                return self.get_end_of_day_price2(ticker)

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

        Args:
            ticker (str): The stock symbol of the company.
            quarter (str): Fiscal quarter in YYYYQM format. For example: quarter=2024Q1. Any quarter since 2010Q1 is supported.
        
        Returns:
            dict: A dictionary containing the earnings call transcript and sentiment scores.
        '''
        params = {
            'function': 'EARNINGS_CALL_TRANSCRIPT',
            'symbol': ticker,
            'quarter': quarter,
            'apikey': self.AlphaVantage_api_key
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

                formatted_dialogue.append(f'**Speaker: {speaker}**\n **Title: {title}**\n {content}\n **Sentiment: {sentiment}**\n')
            
            #Join all transcripts into one:
            full_transcript = '\n'.join(formatted_dialogue)

            #Safeguard from memory limitation:
            max_char = 50000
            if len(full_transcript) > max_char:
                full_transcript = full_transcript[:max_char] + "\n...[Transcript truncated due to length]..."
            
            return {
                    'symbol': symbol,
                    'quarter': quarter,
                    'transcript': full_transcript
                    }
        except Exception as e:
            return f'Error fetching {ticker}transcript: {str(e)}'

    #Define get_news_sentiment method:
    def get_news_sentiment(self, ticker):
        '''
        Fetch news and sentiment scores from Alpha Vantage API.

        Args:
            ticker (str): The stock symbol of the company.
        
        Returns:
            dict: A dictionary containing the news and sentiment scores.
        '''

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'limit': 50,
            'apikey': self.AlphaVantage_api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            #Check for API limits or errors
            if 'feed' not in data:
                return f'limits reached or no news found for {ticker}'
            
            structured_news = []
            count = 0
            for article in data['feed']:
                sentiments = article.get('ticker_sentiment', [])
                sentiment_ticker = next((s for s in sentiments if s['ticker'] == ticker), None)

                if sentiment_ticker:
                    relevance = float(sentiment_ticker['relevance_score'])
                    all_relevance_score = [float(s['relevance_score']) for s in sentiments]
                    max_relevance = max(all_relevance_score if all_relevance_score else 0)
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
                        
            sorted_news = sorted(structured_news, key = lambda x: x['relevance_score'], reverse=True)
                        
            return sorted_news[:5]
            
        except Exception as e:
            return f'Error fetching news: {str(e)}'
    
    #Define get_company_info method:
    def get_company_info(self, ticker):
        '''
        This is for fetching general company information.

        Args:
            ticker (str): The stock symbol of the company.
        
        Returns:
            dict: A dictionary containing the company information.
        '''
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': self.AlphaVantage_api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if not data:
                return f"No overview data found for {ticker}"
            
            # Check for API limit
            if "Note" in data:
                return "API Limit Reached (Alpha Vantage)."

            return data
        except Exception as e:
            return f"Error fetching company info: {str(e)}" 

    #Define get_end_of_day_price2 method:
    def get_end_of_day_price2(self, ticker):
        '''
        This is the back up API call for end of day price when the Alpha Vantage API limit is reached

        Args:
            ticker (str): The stock symbol of the company.
        
        Returns:
            dict: A dictionary containing the ticker closing price with date or None if API limit is reached.
        '''
        
        params = {
            ''
            'symbol': ticker,
            'apikey': self.TD_api_key
        }

        try:
            response = requests.get(self.base_url_2, params=params)
            data = response.json()

            if not data or data.get('status') == 'error':
                error_msg = data.get('message', 'Unknown error')
                return(f'Twelve Data Error for {ticker}: {error_msg}')
                
                        
            #Extract the date and closing price:
            return {
                'ticker': data.get('symbol'),
                'closing_price': data.get('close'),
                'as_of_date': data.get('datetime')
            }

        except Exception as e:
            return(f'Error fetching end of day price: {str(e)}')
    
    #Define get_current_price method:
    def get_current_price(self, ticker):
        '''
        Get the real-time stock price.

        Args:
            ticker (str): The stock symbol of the company.
        
        Returns:
            dict: A dictionary containing the ticker current price with date or None if API limit is reached.
        '''
        
        params = {
            'symbol': ticker,
            'apikey': self.TD_api_key
        }

        try:
            response = requests.get(self.base_url_3, params=params)
            data = response.json()

            if not data or data.get('status') == 'error':
                error_msg = data.get('message', 'Unknown error')
                return(f'Twelve Data Error for {ticker}: {error_msg}')
            elif data.get('price') is None:
                return(f'Market may be closed or no current price is available for {ticker}')
                        
            #Extract the date and closing price:
            return {
                'ticker': ticker.upper(),
                'current_price': round(float(data.get('price')), 2),
                'as_of_date': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            return(f'Error fetching current price: {str(e)}')
    
    #Define get_historical_data method:
    def get_historical_data(self, ticker, interval, start_date=None, end_date=None):
        '''
        This is for getting historical data and times-series for data analysis.
        Supports multiple tickers passed as a comma-separated string (e.g., "AAPL, MSFT").
        If Start date and end date are not mentioned, these parameteres are ignored. The tool will use the default values.

        Args:
            ticker (str): The stock symbol of the company.
            interval (str): The time interval for the historical data (e.g., 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month).
            start_date (str, optional): The start date for the historical data (e.g., "2025-01-01 00:00:00").
            end_date (str, optional): The end date for the historical data (e.g., "2026-03-01 00:00:00").
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the ticker historical data with date or None if API limit is reached.
        '''
        ticker_list = [t.strip().upper() for t in ticker.split(',')]
        all_dfs = []
        errors = []

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
        return combined_df

    #Define create_plot method:
    def create_plot(self, ticker: str):
        '''
        This tool creates an interactive time series plot of the historical data.
        
        CRITICAL INSTRUCTION: ONLY pass the ticker symbol as a string (e.g. "AAPL"). 
        DO NOT pass data arrays or dataframes. The tool already has the data in memory.

        Args:
            ticker (str): The stock symbol to plot.
        '''
        if not hasattr(self, '_last_df') or self._last_df is None:
            return "Error: No data available to plot. Please fetch historical data first."

        df_comb = self._last_df

        #To create the plot:
        fig = px.line(
            df_comb,
            x='datetime',
            y='close',
            color='ticker',
            title=f'Stock Price(s) for {ticker.upper()}'
        )

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black'
        )
        self._last_fig = fig

        return f"Successfully generated and displayed the plot for {ticker}."
        
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
        st.header('Step 2:🤖 Download qwen3:8b')
        st.write('You need to download model qwen3:8b from ollama to have the chatbot work locally.')
        st.markdown('[Download qwen3:8b](https://ollama.com/library/qwen3:8b)')
        
        #Stops the LLM if the keys are missing:
        if not user_Alpha_vantage_key or not user_TD_api_key:
            st.info("🚫 Missing API keys. Please provide both keys to use the SMC.")
            st.stop()

    #Initialize the MarketData instance and API keys:
    if 'market_data' not in st.session_state:
        st.session_state.market_data = MarketData(user_Alpha_vantage_key, user_TD_api_key)

    #Chat history:
    if 'messages' not in st.session_state:
        with st.chat_message('assistant'):
            st.write("Hello! 😃 I'm your Stock Market Chatbot. How can I help you with your stock market queries today? 👌")
        st.session_state.messages = [
            {'role': 'system',
            'content': (
                f'Current date and time: {dt.datetime.now().strftime("%Y-%m-%d %H:%M")}\n'
                'You are a professional financial analyst. You must ALWAYS use tools to fetch stock data before answering. '
                'CRITICAL RULE: NEVER guess, estimate, or hallucinate stock prices or news. If you do not have the exact data from a tool, you MUST call a tool to get it.\n'
                'CRITICAL RULE: NEVER write out tool calls, JSON, or parameters in your conversational text response. You must use the native tool calling functionality. Just call the tool directly.\n'
                'Always convert company names to their official stock ticker symbols (e.g., Apple -> AAPL, Amazon -> AMZN) before calling tools. NEVER pass the full company name.\n'
                'When summarizing news, you MUST cite your sources by providing the exact URL link given to you in the tool data. Do not make up URLs.\n'
                'If the user asks to plot stock prices, use the get_historical_data tool first, and then use the create_plot tool. The plot is displayed in streamlit window.\n'
                'CRITICAL RULE: When using create_plot, ONLY pass the ticker symbol. DO NOT pass the raw data array back to the tool.\n'
                'When fetching historical data, use the current date to calculate start_date and end_date if the user asks for a specific timeframe (e.g., "last 6 months", "year to date").\n'
                'Format dates as "YYYY-MM-DD".\n'
                'If the user does NOT specify a timeframe, leave start_date and end_date blank to let the tool use its default, BUT ensure the interval makes sense (e.g., "1day" for daily charts).\n'
                "If the user asks to plot multiple stocks, pass all tickers as a single comma-separated string into the get_historical_data tool (e.g., 'AAPL, MSFT').\n"
                'If an API tool returns an error, inform the user about the issue.\n'
                'If user asks for earnings call transcript, summarize the transcript. Calculate or extract the quarter parameter from prompt. Format the quarter as "YYYYQ1" or "YYYYQ2" or "YYYYQ3" or "YYYYQ4".'
            )}
        ]

    for msg in st.session_state.messages:
        if msg['role'] in ['user', 'assistant'] and msg.get('content'):
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if msg.get('chart') is not None:
                    st.plotly_chart(msg['chart'], use_container_width=True)
    #User input:
    if prompt := st.chat_input('Ask me anything about the stock market =)'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        #To display user messages in the chat:
        with st.chat_message('user'):
            st.markdown(prompt)
        
        #Add to message history:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        #Get A.I. response:
        with st.chat_message('assistant'):
            message_placeholder = st.empty()

            with st.spinner('Analyzing date...'):
                while True:
                    response = chat(
                        model = 'qwen3:8b',
                        messages=st.session_state.messages,
                        tools=[
                            st.session_state.market_data.get_end_of_day_price,
                            st.session_state.market_data.get_news_sentiment,
                            st.session_state.market_data.get_company_info,
                            st.session_state.market_data.get_current_price,
                            st.session_state.market_data.get_historical_data,
                            st.session_state.market_data.create_plot,
                            st.session_state.market_data.get_earnings_call_transcript
                        ],
                        options={'temperature': 0.0,
                                 'num_ctx':16384}
                    )

                    #Extract message dictionary and append it to session state
                    message_dict = response.get('message', {}) if isinstance(response, dict) else response.message.model_dump()
                    st.session_state.messages.append(message_dict)

                    #Check if tools are called:
                    if message_dict.get('tool_calls'):
                        for tc in message_dict['tool_calls']:
                            function_name = tc['function']['name']
                            arguments = tc['function']['arguments']

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

                            st.toast(f"⚙️ Fetching data from {function_name} for {safe_arg}...")

                            try:
                                if function_name =='get_end_of_day_price':
                                    result = st.session_state.market_data.get_end_of_day_price(safe_arg)
                                    result_str = str(result)
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
                                                f" Link: {url}\n"
                                            )
                                            formatted_news.append(article_str)
                                        result_str = "\n\n".join(formatted_news) if formatted_news else "No news found."
                                    else:
                                        result_str = str(result)
                                elif function_name == 'get_company_info':
                                    result = st.session_state.market_data.get_company_info(safe_arg)
                                    result_str = str(result)
                                elif function_name == 'get_current_price':
                                    result = st.session_state.market_data.get_current_price(safe_arg)
                                    result_str = str(result)
                                elif function_name == 'get_historical_data':
                                    #safe guard the interval output:
                                    raw_interval = arguments.get('interval') or '1day'
                                    safe_interval = str(raw_interval).lower().replace(' ', '')
                                    
                                    start_date = arguments.get('start_date')
                                    end_date = arguments.get('end_date')
                                    result = st.session_state.market_data.get_historical_data(safe_arg, safe_interval, start_date, end_date)
                                    if isinstance(result, pd.DataFrame):
                                        min_date = result['datetime'].min().strftime('%Y-%m-%d')
                                        max_date = result['datetime'].max().strftime('%Y-%m-%d')
                                        result_str = f"Fetched historical data for {safe_arg} from {min_date} to {max_date}. Data snippet:\n{result.tail().to_string()}"
                                        # We could store this for create_plot, but for now we'll just return the info
                                        st.session_state.market_data._last_df = result 
                                    else:
                                        result_str = f"No historical data found for {safe_arg}."
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
                                else:
                                    result_str = f"Error: Unknown function called: {function_name}"

                            except Exception as e:
                                result_str = f'Error: {str(e)}'

                            #Append the tool result to the message history:
                            st.session_state.messages.append({
                                'role': 'tool',
                                'content': result_str,
                                'tool_call_id': tc.get('id')
                            })
                    else:
                        #No tool calls were made:
                        final_text = message_dict.get('content', '')

                        #Check for any renders:
                        fig_to_show = getattr(st.session_state.market_data, '_last_fig', None)
                        if final_text:
                            message_placeholder.markdown(final_text)

                        if fig_to_show is not None:
                            st.plotly_chart(fig_to_show, use_container_width=True)
                            st.session_state.messages[-1]['chart'] = fig_to_show
                            st.session_state.market_data._last_fig = None
                        break
    


if __name__ == '__main__':
    run_streamlit_app()
