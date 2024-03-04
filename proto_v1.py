import streamlit as st
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import pandas as pd
from PIL import Image
import yfinance as yf
import requests
import json
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}

def scrape_stock_data(ticker):
    """Scrapes stock data from Yahoo Finance.

    Args:
        ticker: The ticker symbol of the company to scrape data for.

    Returns:
        A Pandas DataFrame object containing the stock data.
    """

    # Get the company's stock data from Yahoo Finance.
    stock_data = yf.Ticker(ticker).history(period='1d')

    # Return the stock data as a Pandas DataFrame object.
    return stock_data

def calculate_dcf(ticker, growth_rate):

    stock_cf = yf.Ticker(ticker).cashflow
    cashflow_df = pd.DataFrame(stock_cf)

    # Most recent free cash flow
    free_cash_flow = (cashflow_df.loc['Free Cash Flow'][0] + cashflow_df.loc['Free Cash Flow'][1] + cashflow_df.loc['Free Cash Flow'][2])/3
    st.write("Average cashflow for projection is:", free_cash_flow)

    # Calculate the present value of the FCFs.
    pv_of_fcfs = 0
    pv_of_lcf = 0
    for year in range(1, 10):
        pv_of_fcfs += ((free_cash_flow) * ((1 + (growth_rate/100)) ** (year - 1))) / (1 + 0.08) ** year
        pv_of_lcf = (free_cash_flow) * ((1 + (growth_rate/100)) ** (year - 1))
        #st.write(((free_cash_flow) * ((1 + growth_rate) ** (year - 1)))/1000000)

    # Calculate the terminal value of the FCFs.
    # terminal_value = free_cash_flow * (1 + growth_rate) ** 9 / (0.1 - growth_rate)
    terminal_value = (pv_of_lcf)/(((growth_rate/100) - 0.02)*((1+0.1) ** 9))

    # Calculate the total DCF. (Enterprise value)
    dcf = pv_of_fcfs + terminal_value

    cash_casheq = pd.DataFrame(yf.Ticker(ticker).balance_sheet).loc['Cash And Cash Equivalents'][0]
    total_debt = pd.DataFrame(yf.Ticker(ticker).balance_sheet).loc['Total Debt'][0]
    equity_value = dcf + cash_casheq - total_debt

    total_shares = pd.DataFrame(yf.Ticker(ticker).balance_sheet).loc['Share Issued'][0]

    instrinsic_value = equity_value/(total_shares)

    st.write("Calculated Instrinsic Value through the DCF method is:", instrinsic_value)

    return instrinsic_value

def eps_valuation(ticker, ttm_eps, growth_rate, growth_decline_rate):

    final_eps_val = 0.00
    final_eps_val = ttm_eps * (1 + (growth_rate/100))
    for year in range(1,5):
        final_eps_val = final_eps_val * (1 + ((growth_rate/100) * ((1 - (growth_decline_rate/100))**(year-1))))
        #st.write("EPS in Y", year+1, ":", final_eps_val)
    
    # st.write(yf.Ticker(ticker).info)
    #st.write("Get Forward P/E")
    #st.write(get_forward_pe_from_website(ticker))

    #Projected Value
    final_eps_val = final_eps_val * (get_forward_pe_from_website(ticker))

    #Discounting to present value
    final_eps_val = final_eps_val/((1+0.1)**(year+1))

    return final_eps_val

def get_forward_pe_from_website(ticker):
    try:
        # Replace 'URL' with the actual URL of the website providing P/E information
        url = f'https://finance.yahoo.com/quote/{ticker}'

        # Send a GET request to the URL
        response = requests.get(url, headers=headers, timeout=5)

        #st.write("Response status code: ", response)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Locate the element containing the P/E ratio
            forward_pe_element = soup.find('div', {'class': 'D(ib) W(1/2) Bxz(bb) Pstart(12px) Va(t) ie-7_D(i) ie-7_Pos(a) smartphone_D(b) smartphone_W(100%) smartphone_Pstart(0px) smartphone_BdB smartphone_Bdc($seperatorColor)'}).find_all('td', {'class':'Ta(end) Fw(600) Lh(14px)'})[2].text
            #st.write(forward_pe_element)

            # Extract the P/E ratio
            if forward_pe_element == 'N/A':
                forward_pe_ratio = 0
            else:
                forward_pe_ratio = float(forward_pe_element)

            #st.write(forward_pe_ratio)
            return forward_pe_ratio

        else:
            print(f"Failed to fetch data. Status Code: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def get_epsttm_from_website(ticker):
    try:
        # Replace 'URL' with the actual URL of the website providing eps ttm information
        url = f'https://finance.yahoo.com/quote/{ticker}'

        # Send a GET request to the URL
        response = requests.get(url, headers=headers, timeout=5)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Locate the element containing the EPS(ttm) value
            epsttm_element = soup.find('div', {'class': 'D(ib) W(1/2) Bxz(bb) Pstart(12px) Va(t) ie-7_D(i) ie-7_Pos(a) smartphone_D(b) smartphone_W(100%) smartphone_Pstart(0px) smartphone_BdB smartphone_Bdc($seperatorColor)'}).find_all('td', {'class':'Ta(end) Fw(600) Lh(14px)'})[3].text
            #st.write(epsttm_element)

            # Extract the EPS(ttm) value
            epsttm = float(epsttm_element)

            #st.write(epsttm)
            return epsttm

        else:
            print(f"Failed to fetch data. Status Code: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def get_marketcap_from_website(ticker):
    try:
        # Replace 'URL' with the actual URL of the website providing eps ttm information
        url = f'https://finance.yahoo.com/quote/' + ticker + '/key-statistics'

        # Send a GET request to the URL
        response = requests.get(url, headers=headers, timeout=5)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Locate the price of the ticker
            current_market_cap_text = soup.find('div', {'class': 'Pos(r) Mt(10px)'}).find_all('td', {'class':'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})[0].text
            #current_market_cap = float(current_market_price_text)
            st.write(current_market_cap_text)

            current_market_price_text = soup.find('div', {'class': 'D(ib) Mend(20px)'}).find('fin-streamer', {'class':'Fw(b) Fz(36px) Mb(-4px) D(ib)'}).text
            current_market_price = float(current_market_price_text)
            st.write(current_market_price)

            #st.write(epsttm)
            return current_market_cap_text

        else:
            print(f"Failed to fetch data. Status Code: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def get_epspedata(ticker):

    # Get the HTML content of the Yahoo Finance page for the specified ticker symbol.
    url = f"https://finance.yahoo.com/quote/{ticker}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract P/E ratio and TTM EPS from the HTML content.
    pe_ratio_element = get_forward_pe_from_website(ticker)
    ttm_eps_element = get_epsttm_from_website(ticker)

    if pe_ratio_element is None or ttm_eps_element is None:
        raise Exception(f"Error retrieving data for ticker '{ticker}'")

    #pe_ratio = float(pe_ratio_element.text.strip())
    #ttm_eps = float(ttm_eps_element.text.strip())

    return (pe_ratio_element, ttm_eps_element)

def plot_pe_vs_ttm_eps(ticker):
    # Get the stock data.
    pe_ratio = get_forward_pe_from_website(ticker)
    ttm_eps = get_epsttm_from_website(ticker)

    # Create the plot.
    plt.figure(figsize=(10, 6))
    plt.scatter(ttm_eps, pe_ratio)
    plt.xlabel("TTM EPS")
    plt.ylabel("P/E Ratio")
    plt.title(f"P/E vs. TTM EPS for {ticker}")
    plt.grid(True)
    plt.show()

def get_stock_eps_fcf_data(ticker):

    # Construct the API request URL.
    url = "https://www.alphavantage.co/query?function=EARNINGS&symbol=" + ticker + "&apikey=H2DTDYF60C9PNH79"

    # Make the API request.
    response = requests.get(url)

    st.write(url)

    # Check the response status code.
    if response.status_code != 200:
        raise Exception("Error retrieving EPS data from Alpha Vantage: " + response.content)
    
    #Parse the JSON response
    earnings_data = json.loads(response.content)

    #Extract EPS and FCF values from earnings data.
    eps_values = []
    fcf_values = []
    for quarter in earnings_data["quarterlyEarnings"]:
        eps_values.append(quarter["reportedEPS"])
        fcf_values.append(quarter["freeCashflow"])

    return (eps_values, fcf_values)

def plot_eps_vs_fcf(ticker):

    #Get the stock data
    eps_values, fcf_values = get_stock_eps_fcf_data(ticker)

    #Create the plot
    plt.figure(figsize = (10,6))
    plt.plot(eps_values, label="Earnings Per Share")
    plt.plot(fcf_values, label="Free Cash Flow")
    plt.xlabel("Quarter")
    plt.ylabel("Value")
    plt.title(f"EPS vs FCF for {ticker}")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)


def display_dcf(dcf):

  # Display the DCF of the stock to the user.
  st.write(f"The DCF of the stock is: {dcf:.2f}")

def is_valid_ticker(ticker):
    if ticker is not None and ticker:
        return True
    else:
        return False

add_sidebar = st.sidebar.selectbox('Menu', ('Valuation Tool', 'Visualizer', 'Sample models'))


def valuation():

    # Get the ticker symbol from the user.
    ticker = st.text_input("Enter a ticker symbol: ")
    growth_rate = st.number_input("Enter the growth rate in % (for 10% enter 10.00): ")
    growth_decline_rate = st.number_input("Enter the growth decline rate in % (for 5% enter 5.00): ")

    if is_valid_ticker(ticker) and growth_rate != 0 and growth_decline_rate != 0:
        # Scrape the stock data for the specified ticker.
        stock_df = scrape_stock_data(ticker)

        # Display the stock data to the user.
        st.write(stock_df)

        # Calculate DCF for the specified ticker.
        stock_dcf = calculate_dcf(ticker, growth_rate)

        ttm_eps = get_epsttm_from_website(ticker)
        eps_val = eps_valuation(ticker, ttm_eps, growth_rate, growth_decline_rate)

        #st.write("EPS value obtained with inputs is:", eps_val)
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("DCF (Intrinsic)")
            st.title("%.2f" %stock_dcf)

        with col2:
            st.subheader("P/E (Relative)")
            st.title("%.2f" %eps_val)

        st.divider()

        #plot_eps_vs_fcf(ticker)

        #plot_pe_vs_ttm_eps(ticker)



    else:
        st.warning("Please enter a ticker symbol, growth rate and decline rate to proceed")

def visualizer():
    ticker = st.text_input("Enter a ticker symbol: ")

    if ticker:

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        st.title("Stock Price Visualization")

        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            st.line_chart(stock_data["Close"].reset_index().set_index("Date"), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error. Please retry")
    

    st.write('Work in Progress')
    

def models():
    st.write('Hello')

    # Function to fetch historical stock prices
    def get_stock_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        return data

    # Function to calculate annualized returns and covariances
    def calculate_statistics(prices):
        returns = prices.pct_change().mean() * 252
        cov_matrix = prices.pct_change().cov() * 252
        return returns, cov_matrix

    # Function to calculate portfolio metrics
    def maximize_sharpe_ratio(weights, returns, cov_matrix):

        # Calculate portfolio return and volatility
        # weights = weights.reshape(-1,1)
        portfolio_return = np.sum(returns * weights)  # 252 trading days in a year
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = -portfolio_return / portfolio_volatility
        return sharpe_ratio

    # Function to calculate the objective function
    # def objective_function(weights, returns, cov_matrix, risk_aversion):
    #    portfolio_return, portfolio_volatility = calculate_portfolio_metrics(weights, returns, cov_matrix)
    #    return -(portfolio_return - risk_aversion * 0.5 * portfolio_volatility ** 2)

    # Portfolio optimization
    def optimize_portfolio(returns, cov_matrix):
        num_assets = len(returns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array(num_assets * [1. / num_assets]) #Convert to a numpy

        result = minimize(
            maximize_sharpe_ratio, initial_weights,
            args=(returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x

    # Streamlit App
    st.title("Portfolio Optimization Strategy")

    # Sidebar - Input parameters
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    tickers = st.sidebar.text_input("Enter Tickers (comma-separated)", "AAPL, GOOGL, MSFT").split(',')

    # Fetch historical stock prices and calculate returns
    stock_data = pd.DataFrame({ticker: get_stock_data(ticker, start_date, end_date) for ticker in tickers})

    #Calculate statistics
    returns, cov_matrix = calculate_statistics(stock_data)

    # Display stock data
    st.subheader("Historical Stock Prices")
    st.write(stock_data)

    # Calculate expected returns and covariance matrix
    # expected_returns = returns.mean()
    # cov_matrix = returns.cov()

    # Input risk aversion
    # risk_aversion = st.sidebar.number_input("Risk Aversion Coefficient", value=2.0)

    # Calculate optimized weights
    weights = optimize_portfolio(returns, cov_matrix)

    # Display results
    st.subheader("Optimal Asset Weights")
    for i, weight in enumerate(weights):
        st.write(f"{tickers[i]}: {weight:.4f}")

    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    st.subheader("Portfolio Metrics")
    st.write(f"Expected Return: {portfolio_return:.4f}")
    st.write(f"Portfolio Volatility: {portfolio_volatility:.4f}")

    # Visualization (you can customize this based on your preferences)
    import matplotlib.pyplot as plt

    # Scatter plot of portfolios
    plt.scatter(portfolio_volatility, portfolio_return, c='red', marker='o')
    plt.title('Mean-Variance Optimization')
    plt.xlabel('Volatility')
    plt.ylabel('Return')

    # Highlight the optimized portfolio
    plt.scatter(portfolio_volatility, portfolio_return, c='blue', marker='x', label='Optimized Portfolio')
    plt.legend()
    st.pyplot(plt)


if add_sidebar == 'Valuation Tool':
    valuation()
if add_sidebar == 'Visualizer':
    visualizer()
if add_sidebar == 'Sample models':
    models()

# col1.metric("DCF (Intrinsic)", "%.2f" %stock_dcf)
# col2.metric("P/E (Relative)", "%.2f" %eps_val)