

import pandas as pd
import yfinance as yf
import datetime
from IPython.display import display  # Import display for better table presentation

# Get the current date and the date 365 days ago
today = datetime.date.today()
one_year_ago = today - datetime.timedelta(days=365)

# Define the list of top 100 Nasdaq stocks
nasdaq_top_100 = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'META', 'BRK-B', 'UNH',
    'JNJ', 'V', 'JPM', 'PG', 'XOM', 'HD', 'LLY', 'MA', 'CVX', 'MRK', 'AVGO', 'KO',
    'PEP', 'COST', 'ABBV', 'MCD', 'ADBE', 'CSCO', 'TMO', 'NEE', 'CRM', 'WMT', 'DHR',
    'DIS', 'ABT', 'ACN', 'LIN', 'TXN', 'CMCSA', 'NFLX', 'QCOM', 'INTC', 'PFE', 'MS',
    'RTX', 'HON', 'ORCL', 'AMGN', 'SCHW', 'PM', 'BMY', 'LOW', 'IBM', 'SPGI', 'BLK',
    'ADP', 'MDLZ', 'SBUX', 'CB', 'NOW', 'AMAT', 'INTU', 'DE', 'GS', 'LMT', 'UNP',
    'GILD', 'CAT', 'T', 'CI', 'BKNG', 'ISRG', 'ADI', 'VZ', 'UPS', 'FIS', 'PLD',
    'BA', 'MMM', 'C', 'EQIX', 'AMT', 'EL', 'SYK', 'SO', 'DUK', 'D', 'MMC', 'FDX',
    'MU', 'GM', 'BKNG'
]

# Initialize an empty DataFrame for storing the stock data
df_stocks = pd.DataFrame()

# Loop through each stock and fetch historical data
for stock in nasdaq_top_100:
    try:
        # Create ticker object and retrieve historical data
        ticker = yf.Ticker(stock)
        df = ticker.history(start=one_year_ago, end=today)

        # Check if data was retrieved successfully
        if not df.empty:
            df['Ticker'] = stock  # Add a column for the ticker symbol
            df['Price_Date'] = df.index.date  # Add a column for the price date
            df_stocks = pd.concat([df_stocks, df], ignore_index=True)
        else:
            print(f"No data available for {stock}")
    except Exception as e:
        print(f"Error retrieving data for {stock}: {e}")

# Reset index to make 'Price_Date' a column instead of an index
df_stocks.reset_index(drop=True, inplace=True)

# Display the DataFrame containing historical stock data in a better format
display(df_stocks)


df_stocks.head()

import pandas as pd
import pmdarima as pm
from IPython.display import display

# Initialize an empty list to collect forecast results
all_forecasts = []

# Get the unique tickers from the DataFrame
unique_tickers = df_stocks['Ticker'].unique()

# Iterate through each unique ticker
for selected_stock in unique_tickers:
    # Filter the data for the selected stock
    stock_data = df_stocks[df_stocks['Ticker'] == selected_stock]

    # Ensure 'Price_Date' is in datetime format
    stock_data['Price_Date'] = pd.to_datetime(stock_data['Price_Date'])

    # Drop duplicates by date, you can choose to take the last price or mean or any other method
    stock_data = stock_data.sort_values('Price_Date').drop_duplicates(subset='Price_Date', keep='last')

    # Set 'Price_Date' as the index
    stock_data.set_index('Price_Date', inplace=True)

    # Set the frequency of the date index
    stock_data = stock_data.asfreq('B')  # Business day frequency

    # Select the closing prices for modeling
    closing_prices = stock_data['Close']

    # Fit Auto ARIMA model
    model = pm.ARIMA(order=(5, 1, 0))
    model.fit(closing_prices)

    # Forecast the next 10 days
    n_periods = 10
    try:
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

        # Create a DataFrame for the forecast results
        forecast_index = pd.date_range(start=closing_prices.index[-1] + pd.Timedelta(days=1), periods=n_periods, freq='B')
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

        # Reset the index to make dates a column
        forecast_df.reset_index(inplace=True)
        forecast_df.rename(columns={'index': 'Date'}, inplace=True)

        # Add the current price and ticker to the DataFrame
        current_price = closing_prices.iloc[-1]
        forecast_df['Current Price'] = current_price
        forecast_df['Ticker'] = selected_stock

        # Rename the 'Forecast' column to reflect it being the forecasted prices
        forecast_df.rename(columns={'Forecast': 'Predicted Price'}, inplace=True)
        forecast_df['Predicted Price'] = forecast_df['Predicted Price'].round(1)
        # Reorganize columns to have 'Date' first and 'Ticker' second
        forecast_df = forecast_df[['Ticker', 'Current Price', 'Date', 'Predicted Price']]

        # Append the forecast DataFrame to the list
        all_forecasts.append(forecast_df)

    except Exception as e:
        print(f"An error occurred while forecasting for {selected_stock}: {e}")

# Concatenate all forecast DataFrames into a single DataFrame
final_forecast_df = pd.concat(all_forecasts, ignore_index=True)

# Pivot the table
pivot_df = final_forecast_df.pivot(index=['Ticker', 'Current Price'], columns='Date', values='Predicted Price')

# Reset the index to make TICKER and CURRENT PRICE as columns
pivot_df.reset_index(inplace=True)

# Format date columns to 'MMM D YYYY' (e.g., 'Nov 4 2024')
pivot_df.columns.name = None  # Remove the name from the columns
pivot_df.columns = [col.strftime('%b %d %Y') if isinstance(col, pd.Timestamp) else col for col in pivot_df.columns]

# Sort columns in ascending order based on the date
date_columns = [col for col in pivot_df.columns if isinstance(col, str) and len(col.split()) == 3]
sorted_date_columns = sorted(date_columns, key=lambda x: pd.to_datetime(x))

# Construct the final columns order
final_columns = ['Ticker', 'Current Price'] + sorted_date_columns
pivot_df = pivot_df[final_columns]

# Display the final transposed forecast DataFrame
display(pivot_df)



from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2 template engine
templates = Jinja2Templates(directory="templates")




# Route to render the dataframe in HTML
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Convert the DataFrame to HTML table
    df_html = pivot_df.to_html(classes="table table-striped", index=False)
    # Pass the HTML content to Jinja2 template
    return templates.TemplateResponse("index.html", {"request": request, "df_html": df_html})





