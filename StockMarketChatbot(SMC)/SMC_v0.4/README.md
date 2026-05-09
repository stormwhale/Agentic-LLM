# Stock Market Chatbot (SMC)
 **Status: Completed Prototype as of 5/9/2026**

 * This project was developed as a Capstone Project for the M.S. in Data Science program at the CUNY School of Professional Studies.*

## Objective: 
A locally hosted, RAG-based stock market chatbot designed to autonomously utilize financial APIs. This project prioritizes:
1. **Privacy & Cost-Efficiency:** Fully local execution ensures user data remains private, with zero inference costs.
2. **Domain-Specific Accuracy:** Striking a fine balance between high-performance execution and insightful, financially literate responses.
3. **Autonomous Tool Use:** The agent dynamically routes queries to the appropriate external API tools.

## Architecture & Tech Stack:
* **LLM:** [Qwen 3.5:9B](https://ollama.com/) – Selected for its optimal balance of performance, API-calling intelligence, and hardware efficiency for local deployment.
* **Primary Data Source:** [AlphaVantage API](https://www.alphavantage.co/) – Provides real-time financial news and sentiment scoring (Free Tier).
* **Fallback Data Source:** [TwelveData API](https://twelvedata.com/) – Acts as an autonomous fallback for stock prices, company information, and historical data to handle AlphaVantage rate limits seamlessly.

## Getting Started Steps:

To run this project locally, you will need the following:

1. **Local LLM Environment:** * Install [Ollama](https://ollama.com/).
   * Pull the required model by running: `ollama run qwen3.5:9B`.
   * Have Ollama running in the background before launching the chatbot.
2. **API Keys:** * Users must generate their own free API keys for AlphaVantage and TwelveData. 
   * These keys will be inputted directly into the chatbot interface upon launch.
   * You may also set the API keys in the .env file to avoid repeatedly entering them.
3. **Run the "run.bat" script**

## SMC 0.4 Release was evaluated using DeepEval against DeepseekV-3.2 and achieved similar scores in the following metrics:
  * Faithfullness
  * Tool Correctness
  * Answer Relevancy
  * Arugment Correctness

## 🗺️ Roadmap / Current Status
  * [x] Define architecture and select API tools
  * [x] Implement basic RAG pipeline
  * [x] Integrate AlphaVantage / TwelveData fallback logic
  * [x] Build local user interface
  * [x] Review the need for additional functionality
  * [x] Conduct DeepEval evaluation