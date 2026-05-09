import os
import json
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from deepeval.dataset import EvaluationDataset
from datetime import datetime
from deepeval.synthesizer.config import StylingConfig
#deepseek key:
load_dotenv('deepseek_key.env')
deepseek_api_key = os.getenv("deepseek_key")

#Define the Deepseek llm:
class Deepseek(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOpenAI(
            model='deepseek-chat',
            openai_api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
            model_kwargs={"response_format": {"type": "json_object"}},
            temperature=0.0
        )

    def load_model(self):
        return self.model
    
    def _clean_json_output(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
        
    def generate(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nPlease ensure your response is formatted entirely as JSON."
        response = self.model.invoke(prompt)
        return self._clean_json_output(response.content)

    async def a_generate(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nPlease ensure your response is formatted entirely as JSON."
        response = await self.model.ainvoke(prompt)
        return self._clean_json_output(response.content)

    def get_model_name(self) -> str:
        return "deepseek-chat"
    
styling = StylingConfig(
    input_format=(
        "Short, realistic user message to a stock market chatbot. "
        "Use casual language and real tickers (AAPL, TSLA, NVDA, SOFI, SPY, VOO, QQQ). "
        "Never mention tool names or API parameters directly."
    ),
    expected_output_format=(
        "A direct factual answer citing the specific price, date, sentiment, "
        "or data the tool would return. No buy/sell recommendations."
    ),
    task=(
        "Answering real-time and historical stock market queries "
        "using financial data tools."
    ),
    scenario=(
        "Retail investors and day traders asking about live prices, "
        "end-of-day closes, news sentiment, ETF profiles, earnings calls, "
        "and historical price charts."
    ),
)

synthesizer = Synthesizer(model=Deepseek(), styling_config=styling)

custom_docs = [
    Document(
        page_content="""
        Tool: get_current_price
        Purpose: Get the real-time live price of a stock or ETF. Use only for live market checks.

        Example conversations:
        User: What is Apple's stock price right now?
        Assistant: [{"name": "get_current_price", "arguments": {"ticker": "AAPL"}}]

        User: How much does one share of the company that makes iPhones cost at this exact moment?
        Assistant: [{"name": "get_current_price", "arguments": {"ticker": "AAPL"}}]

        User: Give me the live quote for NVDA and AMD.
        Assistant: [{"name": "get_current_price", "arguments": {"ticker": "NVDA"}}, {"name": "get_current_price", "arguments": {"ticker": "AMD"}}]

        User: Is SPY up or down right now?
        Assistant: [{"name": "get_current_price", "arguments": {"ticker": "SPY"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_current_price"}
    ),
    Document(
        page_content="""
        Tool: get_end_of_day_price
        Purpose: Get the closing price and daily change at the end of the most recent trading day.

        Example conversations:
        User: Where did Tesla close today?
        Assistant: [{"name": "get_end_of_day_price", "arguments": {"ticker": "TSLA"}}]

        User: How much did PLTR drop by the closing bell?
        Assistant: [{"name": "get_end_of_day_price", "arguments": {"ticker": "PLTR"}}]

        User: What was the daily change percentage for AMD at yesterday's close?
        Assistant: [{"name": "get_end_of_day_price", "arguments": {"ticker": "AMD"}}]

        User: Give me the final wrap-up numbers for MSFT today.
        Assistant: [{"name": "get_end_of_day_price", "arguments": {"ticker": "MSFT"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_end_of_day_price"}
    ),
    Document(
        page_content="""
        Tool: get_historical_data
        Purpose: Get historical price time-series data for analysis over a period of time. Returns raw data without a chart. Use for ranges longer than one day.

        Example conversations:
        User: Give me Meta's price history for the last 30 days.
        Assistant: [{"name": "get_historical_data", "arguments": {"ticker": "META", "interval": "1day", "start_date": "30 days ago", "end_date": "today"}}]

        User: How has VOO performed YTD?
        Assistant: [{"name": "get_historical_data", "arguments": {"ticker": "VOO", "interval": "1day", "start_date": "beginning of the year", "end_date": "today"}}]

        User: Compare the volatility of SOFI and QS since the start of Q3 2023.
        Assistant: [{"name": "get_historical_data", "arguments": {"ticker": "SOFI, QS", "interval": "1day", "start_date": "2023-07-01", "end_date": "today"}}]

        User: What were NVDA's intraday movements like last Monday morning?
        Assistant: [{"name": "get_historical_data", "arguments": {"ticker": "NVDA", "interval": "15min", "start_date": "last Monday 09:30", "end_date": "last Monday 12:00"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_historical_data"}
    ),
    Document(
        page_content="""
        Tool: get_news_sentiment
        Purpose: Fetch recent news and sentiment scores for a specific company or ticker.

        Example conversations:
        User: I heard bad things about Starbucks recently, what's the actual news?
        Assistant: [{"name": "get_news_sentiment", "arguments": {"ticker": "SBUX"}}]

        User: Why is Tesla crashing today? Any news?
        Assistant: [{"name": "get_news_sentiment", "arguments": {"ticker": "TSLA"}}]

        User: Are analysts feeling bullish or bearish on Palantir this week?
        Assistant: [{"name": "get_news_sentiment", "arguments": {"ticker": "PLTR"}}]

        User: What's the PR outlook for the company led by Tim Cook?
        Assistant: [{"name": "get_news_sentiment", "arguments": {"ticker": "AAPL"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_news_sentiment"}
    ),
    Document(
        page_content="""
        Tool: get_earnings_call_transcript
        Purpose: Get the earnings call transcript for a company in a specific quarter.

        Example conversations:
        User: What did Apple's leadership say about AI in their Q1 2024 call?
        Assistant: [{"name": "get_earnings_call_transcript", "arguments": {"ticker": "AAPL", "quarter": "2024Q1"}}]

        User: Did MSFT beat their forward guidance on the most recent call?
        Assistant: [{"name": "get_earnings_call_transcript", "arguments": {"ticker": "MSFT", "quarter": "most recent"}}]

        User: Pull the transcript from Amazon's Q3 2023 earnings report.
        Assistant: [{"name": "get_earnings_call_transcript", "arguments": {"ticker": "AMZN", "quarter": "2023Q3"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_earnings_call_transcript"}
    ),
    Document(
        page_content="""
        Tool: get_specific_date_price
        Purpose: Get the closing price for one specific historical date. Use when the user asks for a single point in the past (e.g., "yesterday", "on Christmas 2023"). Do NOT use for live prices or date ranges.

        Example conversations:
        User: What did Tesla close at on March 14, 2023?
        Assistant: [{"name": "get_specific_date_price", "arguments": {"ticker": "TSLA", "date": "2023-03-14"}}]

        User: Show me Apple's closing price the day before Halloween last year.
        Assistant: [{"name": "get_specific_date_price", "arguments": {"ticker": "AAPL", "date": "October 30 of last year"}}]

        User: Where was QQQ priced exactly one year ago today?
        Assistant: [{"name": "get_specific_date_price", "arguments": {"ticker": "QQQ", "date": "1 year ago"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_specific_date_price"}
    ),
    Document(
        page_content="""
        Tool: get_etf_profile
        Purpose: Get ETF-specific data including expense ratio, top holdings, and sector allocation. Never use for individual stocks.

        Example conversations:
        User: What is QQQ's expense ratio and what's inside it?
        Assistant: [{"name": "get_etf_profile", "arguments": {"ticker": "QQQ"}}]

        User: Compare the management fees and tech allocation of VOO versus SPY.
        Assistant: [{"name": "get_etf_profile", "arguments": {"ticker": "VOO"}}, {"name": "get_etf_profile", "arguments": {"ticker": "SPY"}}]

        User: What sectors does the Russell 2000 ETF invest in?
        Assistant: [{"name": "get_etf_profile", "arguments": {"ticker": "IWM"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_etf_profile"}
    ),
    Document(
        page_content="""
        Tool: get_general_news
        Purpose: Fetch market-wide financial news by topic or sector. Use for broad market topics (e.g., tech, biotech, economy), NOT for specific companies.

        Example conversations:
        User: What's happening in the broader tech sector today?
        Assistant: [{"name": "get_general_news", "arguments": {"topics": "technology"}}]

        User: Any macro news about Fed interest rates and inflation?
        Assistant: [{"name": "get_general_news", "arguments": {"topics": "economy_monetary, economy_macro"}}]

        User: How is the housing market looking right now?
        Assistant: [{"name": "get_general_news", "arguments": {"topics": "real_estate"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_general_news"}
    ),
    Document(
        page_content="""
        Tool: get_breaking_news
        Purpose: Fetch breaking world news from Reuters, BBC, WSJ, FT. Use ONLY for active geopolitical events, crises, macro policies, or disasters. Do NOT use for standard finance/stock news.

        Example conversations:
        User: What's happening with the trade war right now?
        Assistant: [{"name": "get_breaking_news", "arguments": {"keywords": "trade war tariffs"}}]

        User: Is there any breaking coverage on the earthquake in Japan?
        Assistant: [{"name": "get_breaking_news", "arguments": {"keywords": "Japan earthquake"}}]

        User: Did the White House just announce new semiconductor export bans?
        Assistant: [{"name": "get_breaking_news", "arguments": {"keywords": "White House semiconductor export bans"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_breaking_news"}
    ),
    Document(
        page_content="""
        Tool: get_company_info
        Purpose: Get general corporate information like sector, industry, CEO, market cap, headquarters location, and business description.

        Example conversations:
        User: What's the total valuation and market cap of Netflix?
        Assistant: [{"name": "get_company_info", "arguments": {"ticker": "NFLX"}}]

        User: Tell me about Palantir as a business—who runs it and what do they actually do?
        Assistant: [{"name": "get_company_info", "arguments": {"ticker": "PLTR"}}]

        User: Where is Apple headquartered and how many people work there?
        Assistant: [{"name": "get_company_info", "arguments": {"ticker": "AAPL"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "get_company_info"}
    ),
    Document(
        page_content="""
        Tool: create_plot
        Purpose: Create an interactive time series chart of historical stock data. Must be paired with get_historical_data. ONLY trigger when the user asks to "plot", "graph", "visualize", or "map out".

        Example conversations:
        User: Graph Apple's stock price over the last 6 months.
        Assistant: [{"name": "get_historical_data", "arguments": {"ticker": "AAPL", "interval": "1day", "start_date": "6 months ago", "end_date": "today"}}, {"name": "create_plot", "arguments": {"ticker": "AAPL"}}]

        User: Visualize TSLA and RIVN side by side from March 2025 to now.
        Assistant: [{"name": "get_historical_data", "arguments": {"ticker": "TSLA, RIVN", "interval": "1day", "start_date": "2025-03-01", "end_date": "today"}}, {"name": "create_plot", "arguments": {"ticker": "TSLA, RIVN"}}]

        User: Map out the 2-year trajectory of PLTR.
        Assistant: [{"name": "get_historical_data", "arguments": {"ticker": "PLTR", "interval": "1day", "start_date": "2 years ago", "end_date": "today"}}, {"name": "create_plot", "arguments": {"ticker": "PLTR"}}]
        """,
        metadata={"source": "tool_documentation", "tool": "create_plot"}
    )
]

tool_context = [[doc.page_content] for doc in custom_docs]

generate_goldens = synthesizer.generate_goldens_from_contexts(
    contexts=tool_context,
    max_goldens_per_context= 1, #This is to control how many questions are generated per tool. 11 tools in total.
    include_expected_output=True    
)

dataset = EvaluationDataset(goldens=generate_goldens)
timestamp = datetime.now().strftime('%Y/%m/%d_%H:%M:%S')
os.makedirs("./generated_questions", exist_ok=True)

synthesizer.save_as(
    file_type="json",
    directory="./generated_questions",
    file_name=f"deepeval_questions_raw_{timestamp}"
)
print(f"Generated {len(generate_goldens)} goldens")
for g in generate_goldens[:3]:
    print(f"  Q: {g.input}")
    print(f"  A: {g.expected_output}")

#==========================================================================================
#This part is to format the expected output to be in the correct format for the evaluation
#==========================================================================================

# Replace with actual file name
input_filename = f'./generated_questions/deepeval_questions_raw_{timestamp}.json' 
output_filename = f'./generated_questions/deepeval_questions_formatted_{timestamp}.json'

with open(input_filename, 'r') as f:
    dataset = json.load(f)

cleaned_dataset = []

for item in dataset:
    raw_expected = item.get("expected_output", "")
    reference_tools = []
    
    try:
        # Step 1: Parse the outer JSON wrapper {"response": "..."}
        outer_parsed = json.loads(raw_expected)
        
        # Step 2: Extract the inner string containing the list of tools
        inner_string = outer_parsed.get("response", "[]")
        
        # Step 3: Parse the inner string into an actual Python list of dictionaries
        reference_tools = json.loads(inner_string)
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse item '{item.get('input')}': {e}")
        reference_tools = []

    # Step 4: Map to the clean format required by your DeepEval testing script
    cleaned_item = {
        "user_input": item.get("input"),
        "response": None,                         
        "retrieved_contexts": item.get("context"),
        "reference_tool_calls": reference_tools,  
        "actual_tool_calls": []                   
    }
    
    cleaned_dataset.append(cleaned_item)

# Save the beautifully clean dataset
with open(output_filename, 'w') as f:
    json.dump(cleaned_dataset, f, indent=4)

print(f"Successfully cleaned {len(cleaned_dataset)} items and saved to {output_filename}")