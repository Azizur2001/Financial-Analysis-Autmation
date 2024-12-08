import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import openai
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
from groq import Groq
from google.colab import userdata
from dotenv import load_dotenv
import os
from newsapi import NewsApiClient
import requests



load_dotenv()

# Set API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
groq_api_key = os.getenv("GROQ_API_KEY")
newsapi_key = os.getenv("NEWSAPI_KEY")
newsapi = NewsApiClient(api_key=newsapi_key)


client = openai
client2 = Groq(api_key=groq_api_key)


# Initialize Pinecone
try:
    pinecone_client = Pinecone(api_key=pinecone_api_key)
except Exception as e:
    st.error(f"Error initializing Pinecone client: {e}")
    raise SystemExit(1)

# Define index name and namespace
index_name = "stock-project"
namespace = "stock-descriptions"

try:
    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=pinecone_environment
            )
        )
    index = pinecone_client.Index(index_name)
except Exception as e:
    st.error(f"Error setting up Pinecone index: {e}")
    raise SystemExit(1)

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
hf_model = SentenceTransformer(embedding_model_name)



@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    """
    Fetch stock data using yfinance.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "description": info.get("longBusinessSummary", "Description not available."),
        "url": info.get("website", "#"),
        "earnings_growth": info.get("earningsGrowth", "Data not available"),
        "revenue_growth": info.get("revenueGrowth", "Data not available"),
        "gross_margins": info.get("grossMargins", "Data not available"),
        "ebitda_margins": info.get("ebitdaMargins", "Data not available"),
        "52_week_change": info.get("52WeekChange", "Data not available"),
    }


# GROQ
def generate_factor_explanations(stock_data):
    """
    Generate dynamic factor explanations using the Groq LLM for each stock.
    """
    explanations = {}
    for stock in stock_data:
        try:
            system_prompt = """
            You are an expert financial analyst. Analyze the following stock data and provide specific explanations for the following factors:
            - Growth Potential
            - Market Competition
            - Financial Health
            - Innovation
            - Industry Trends
            - Regulatory Environment
            Be specific to the stock's performance and characteristics.
            """
            stock_description = (
                f"Name: {stock['name']}\n"
                f"Earnings Growth: {stock['earnings_growth']}\n"
                f"Revenue Growth: {stock['revenue_growth']}\n"
                f"Gross Margins: {stock['gross_margins']}\n"
                f"EBITDA Margins: {stock['ebitda_margins']}\n"
                f"52-Week Change: {stock['52_week_change']}\n"
                f"Description: {stock['description']}"
            )
            chat_completion = client2.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": stock_description}
                ]
            )
            explanations[stock['name']] = chat_completion.choices[0].message.content
        except Exception as e:
            explanations[stock['name']] = f"Error generating explanations: {e}"
    return explanations


def create_factor_explanations(stock_data):
    """
    Display factor explanations for each stock using Groq-generated content.
    """
    explanations = generate_factor_explanations(stock_data)
    for stock in stock_data:
        with st.expander(f"Explanations for {stock['name']}"):
            explanation = explanations[stock['name']]
            if "Error" in explanation:
                st.error(explanation)
            else:
                st.markdown(explanation)




@st.cache_data(ttl=3600)
def fetch_price_history(ticker, start_date, end_date):
    """
    Fetch historical price data for a given stock ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)

        # if history.empty:
        #     st.warning(f"No historical data found for {ticker}.")

        return history
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure


def query_pinecone_index(query_embedding):
    """
    Query the Pinecone index using the provided embedding.
    """
    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=10,  # Increase to fetch more results for deduplication
            include_metadata=True,
            namespace=namespace
        )
        return results
    except Exception as e:
        st.error(f"Error querying Pinecone index: {e}")
        return None

def fetch_unique_tickers(results, max_results=6):
    """
    Extract unique stock tickers from Pinecone query results.
    """
    unique_tickers = []
    seen_tickers = set()

    for match in results['matches']:
        ticker = match.metadata.get("Ticker")
        if ticker and ticker not in seen_tickers:
            unique_tickers.append(ticker)
            seen_tickers.add(ticker)
            if len(unique_tickers) >= max_results:
                break

    return unique_tickers

def create_stock_card(stock_data):
    def format_value(value):
        if isinstance(value, (int, float)):
            return f"{value * 100:.2f}%" if value != "Data not available" else value
        return value

    st.markdown(f"""
        <div style='background-color: #2C2C2E; padding: 15px; border-radius: 10px; color: white; margin-bottom: 10px;'>
            <h3>{stock_data['name']}</h3>
            <p>{stock_data['description'][:200]}...</p>
            <a href='{stock_data['url']}' target='_blank' style='color: #1E90FF;'>Website</a>
            <hr>
            <p><b>Earnings Growth:</b> {format_value(stock_data['earnings_growth'])}</p>
            <p><b>Revenue Growth:</b> {format_value(stock_data['revenue_growth'])}</p>
            <p><b>Gross Margins:</b> {format_value(stock_data['gross_margins'])}</p>
            <p><b>EBITDA Margins:</b> {format_value(stock_data['ebitda_margins'])}</p>
            <p><b>52-Week Change:</b> {format_value(stock_data['52_week_change'])}</p>
        </div>
    """, unsafe_allow_html=True)



def plot_stock_prices(tickers, start_date, end_date):
    plt.figure(figsize=(10, 5))
    data_found = False  # Flag to check if any valid data exists

    for ticker in tickers:
        try:
            history = fetch_price_history(ticker, start_date, end_date)

            # Check if history is empty
            if history.empty:
                continue  # Skip this ticker if no data is found

            # Ensure 'Close' column exists
            if 'Close' not in history.columns:
                continue  # Skip this ticker if 'Close' column is missing

            # Normalize prices and plot
            normalized_prices = (history['Close'] / history['Close'].iloc[0]) * 100
            plt.plot(history.index, normalized_prices, label=ticker)
            data_found = True  # Set flag if valid data is found

        except Exception as e:
            # Log errors silently to avoid crashing the app
            continue

    # Finalize plot or show warning
    if data_found:
        plt.legend()
        plt.title("Normalized Stock Price History (% Change)")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price (% Change)")
        st.pyplot(plt)
    else:
        st.warning("No valid historical price data found for the selected stocks.")



def create_radar_chart(stock_data):
    """
    Create a radar chart for the selected stocks based on key financial metrics.
    """
    # Define the categories for the radar chart
    categories = ['Earnings Growth', 'Revenue Growth', 'Gross Margins', 'EBITDA Margins']
    fig = go.Figure()

    # Check if stock_data has valid entries
    valid_data_found = False

    for stock in stock_data:
        # Validate and prepare the data for the radar chart
        values = [
            stock.get('earnings_growth', 0) * 100 if isinstance(stock.get('earnings_growth'), (int, float)) else 0,
            stock.get('revenue_growth', 0) * 100 if isinstance(stock.get('revenue_growth'), (int, float)) else 0,
            stock.get('gross_margins', 0) * 100 if isinstance(stock.get('gross_margins'), (int, float)) else 0,
            stock.get('ebitda_margins', 0) * 100 if isinstance(stock.get('ebitda_margins'), (int, float)) else 0,
        ]

        # Check if all values are 0 (no meaningful data)
        if any(value > 0 for value in values):
            valid_data_found = True
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=stock.get('name', 'Unknown')
            ))

    # Only display the radar chart if valid data is found
    if valid_data_found:
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True
        )
        st.plotly_chart(fig)
    else:
        st.warning("No valid data available to display the radar chart.")


def process_with_llm(query, stock_data=None):
    """
    Generate a detailed stock comparison summary using the Groq LLM.
    """
    try:
        system_prompt = """
       You are an expert stock analyst. Use the provided stock data to confidently answer the user's query.
#         Format the response properly, ensuring readability and clarity. Use complete sentences, and reference the stock data accurately.
        """

        if stock_data:
            # Prepare the stock data for the LLM input
            formatted_data = "\n".join([
                f"Name: {item['name']}\n"
                f"Earnings Growth: {item['earnings_growth']}\n"
                f"Revenue Growth: {item['revenue_growth']}\n"
                f"Gross Margins: {item['gross_margins']}\n"
                f"EBITDA Margins: {item['ebitda_margins']}\n"
                f"52-Week Change: {item['52_week_change']}\n"
                f"Description: {item['description'][:200]}"
                for item in stock_data
            ])
        else:
            formatted_data = "No stock data available."

        # Use the Groq LLM for generating the response
        chat_completion = client2.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nStock Data:\n{formatted_data}"}
            ]
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        # Return an error message if something goes wrong
        return f"Error generating Stock Comparison Summary: {e}"


@st.cache_data(ttl=3600)
def fetch_news(stock_name):
    """
    Fetch recent news articles for a given stock using NewsAPI.
    """
    try:
        articles = newsapi.get_everything(
            q=stock_name,
            language="en",
            sort_by="publishedAt",
            page_size=5  # Limit to 5 articles for display
        )
        return articles.get("articles", [])
    except Exception as e:
        st.error(f"Error fetching news for {stock_name}: {e}")
        return []


def display_news(stock_name):
    """
    Display news articles for a specific stock.
    """
    st.subheader(f"Recent News for {stock_name}")
    articles = fetch_news(stock_name)
    if articles:
        for article in articles:
            with st.container():
                st.markdown(f"**{article['title']}**")
                st.write(f"Source: {article['source']['name']} | Published at: {article['publishedAt']}")
                st.markdown("---")
                # Add the "Read More" link in a smaller button-like text
                st.markdown(
                    f"<a href='{article['url']}' target='_blank' style='display: inline-block; background-color: #007BFF; color: white; padding: 5px 10px; border-radius: 5px; text-decoration: none;'>Read More</a>",
                    unsafe_allow_html=True
                )
                st.markdown("---")
    else:
        st.write("No recent news articles found.")

# Main
def main():
    st.title("Automated Stock Analysis")
    st.write("Enter a description of the kinds of stocks you are looking for and explore real-time insights.")

    # Query Input
    user_query = st.text_input("Enter your query (e.g., 'data center builders'):")

    if st.button("Find Stocks"):
        if user_query.strip():
            with st.spinner("Processing your query..."):
                query_embedding = hf_model.encode(user_query)
                results = query_pinecone_index(query_embedding)

                if not results or not results.get("matches"):
                    st.warning("No relevant stocks found!")
                    return

                tickers = fetch_unique_tickers(results, max_results=6)

                if not tickers:
                    st.warning("No unique stocks found. Try refining your query!")
                    return

                stock_data = [fetch_stock_data(ticker) for ticker in tickers]

                st.subheader("Stock Overview")
                for stock in stock_data:
                    create_stock_card(stock)

                st.subheader("Stock Price Trends")
                start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
                end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
                plot_stock_prices(tickers, start_date, end_date)

                st.subheader("Market Trend Radar")
                selected_stocks = st.multiselect(
                    "Select stocks to compare:",
                    options=[stock["name"] for stock in stock_data],
                    default=[stock["name"] for stock in stock_data],
                )
                filtered_stock_data = [stock for stock in stock_data if stock["name"] in selected_stocks]
                create_radar_chart(filtered_stock_data)

                st.subheader("Stock Comparison Summary")
                summary = process_with_llm(user_query, stock_data)
                st.write(summary)

                st.subheader("Factor Explanations")
                create_factor_explanations(stock_data)

                # News Section
                st.subheader("Recent News")
                for stock in stock_data:
                    st.markdown(f"### {stock['name']}")
                    display_news(stock["name"])
                    st.markdown("---")
        else:
            st.warning("Please enter a query!")






if __name__ == "__main__":
    main()

