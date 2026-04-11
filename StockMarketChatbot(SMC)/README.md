# Stock Market Chatbot (SMC)
 **Status: Work in Progress (Active Development)**

 * This project is being developed as a Capstone Project for the M.S. in Data Science program at the CUNY School of Professional Studies.*

## Objective: 
A locally hosted, RAG-based stock market chatbot designed to autonomously utilize financial APIs. This project prioritizes:
1. **Privacy & Cost-Efficiency:** Fully local execution ensures user data remains private, with zero inference costs.
2. **Domain-Specific Accuracy:** Striking a fine balance between high-performance execution and insightful, financially literate responses.
3. **Autonomous Tool Use:** The agent dynamically routes queries to the appropriate external API tools.

## Architecture & Tech Stack:
* **LLM:** [Qwen 3.5:9B](https://ollama.com/) – Selected for its optimal balance of performance, API-calling intelligence, and hardware efficiency for local deployment.
* **Primary Data Source:** [AlphaVantage API](https://www.alphavantage.co/) – Provides real-time financial news and sentiment scoring (Free Tier).
* **Fallback Data Source:** [TwelveData API](https://twelvedata.com/) – Acts as an autonomous fallback for stock prices, company information, and historical data to handle AlphaVantage rate limits seamlessly.

## Getting Started (Prerequisites):
  To run this project locally, you will need the following:

1. **Local LLM Environment:** * Install [Ollama](https://ollama.com/).
   * Pull the required model by running: `ollama run qwen3.5:9B` (or your specific model tag).
2. **API Keys:** * Users must generate their own free API keys for AlphaVantage and TwelveData. 
   * These keys will be inputted directly into the chatbot interface upon launch.

## Evaluation:
The final pipeline and LLM outputs will be rigorously evaluated using the **RAGAS (Retrieval Augmented Generation Assessment)** framework to ensure the retrieved context and generated answers meet strict quality thresholds.

## 🗺️ Roadmap / Current Status
  * [x] Define architecture and select API tools
  * [x] Implement basic RAG pipeline
  * [x] Integrate AlphaVantage / TwelveData fallback logic
  * [x] Build local user interface
  * [x] Review the need for additional functionality
  * [x] Conduct DeepEval evaluation
