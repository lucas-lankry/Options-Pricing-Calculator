

API_KEY = 'c824d2d0bf46f7f2a17f733e9d883f2842cfcf3d0dc9de929118e950f0bc4319'
"""
Walmart Financial Statements Scraper - Income Statement, Balance Sheet, Cash Flow
10-K filings only for 2021-2024 with Excel export
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime

def get_walmart_cik():
    """Get Walmart's CIK from SEC database"""
    headers = {'User-Agent': "luckylankry@gmail.fr"}
    
    companyTickers = requests.get( 
        "https://www.sec.gov/files/company_tickers.json",
        headers=headers
    )
    
    companyData = pd.DataFrame.from_dict(companyTickers.json(), orient='index')
    companyData['cik_str'] = companyData['cik_str'].astype(str).str.zfill(10)
    
    walmart_data = companyData[companyData['ticker'] == 'WMT']
    
    if walmart_data.empty:
        print("Walmart not found!")
        return None, None
        
    walmart_cik = walmart_data['cik_str'].iloc[0]
    print(f"Found Walmart with CIK: {walmart_cik}")
    
    return walmart_cik, headers

def extract_financial_data(cik, headers, concept):
    """Extract specific financial concept from 10-K filings for 2021-2024"""
    try:
        response = requests.get(
            f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json',
            headers=headers
        )
        
        if response.status_code == 200:
            data = pd.DataFrame.from_dict(response.json()['units']['USD'])
            
            # Filter for 10-K only and years 2021-2024
            annual_data = data[data['form'] == '10-K'].copy()
            annual_data = annual_data[annual_data['fy'].isin([2021, 2022, 2023, 2024])]
            
            return annual_data[['end', 'val', 'fy']].sort_values('fy')
        else:
            print(f"Failed to get {concept}: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error getting {concept}: {e}")
        return pd.DataFrame()

def get_walmart_financial_statements():
    """Get comprehensive Walmart financial statements"""
    
    result = get_walmart_cik()
    if not result[0]:
        return None
    
    walmart_cik, headers = result
    
    # Define financial statement components
    financial_items = {
        # INCOME STATEMENT
        'Revenues': 'Revenues',
        'CostOfRevenue': 'CostOfRevenue', 
        'GrossProfit': 'GrossProfit',
        'OperatingExpenses': 'OperatingExpenses',
        'OperatingIncomeLoss': 'OperatingIncomeLoss',
        'InterestExpense': 'InterestExpense',
        'IncomeTaxExpense': 'IncomeTaxExpense',
        'NetIncomeLoss': 'NetIncomeLoss',
        
        # BALANCE SHEET - Assets
        'Assets': 'Assets',
        'AssetsCurrent': 'AssetsCurrent',
        'CashAndCashEquivalents': 'CashAndCashEquivalents',
        'AccountsReceivableNet': 'AccountsReceivableNet',
        'InventoryNet': 'InventoryNet',
        'PropertyPlantAndEquipmentNet': 'PropertyPlantAndEquipmentNet',
        'Goodwill': 'Goodwill',
        'IntangibleAssetsNetExcludingGoodwill': 'IntangibleAssetsNetExcludingGoodwill',
        
        # BALANCE SHEET - Liabilities & Equity
        'Liabilities': 'Liabilities',
        'LiabilitiesCurrent': 'LiabilitiesCurrent',
        'AccountsPayableCurrent': 'AccountsPayableCurrent',
        'DebtCurrent': 'DebtCurrent',
        'LongTermDebt': 'LongTermDebt',
        'StockholdersEquity': 'StockholdersEquity',
        
        # CASH FLOW STATEMENT
        'NetCashProvidedByUsedInOperatingActivities': 'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInInvestingActivities': 'NetCashProvidedByUsedInInvestingActivities',
        'NetCashProvidedByUsedInFinancingActivities': 'NetCashProvidedByUsedInFinancingActivities',
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect': 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect',
        'CapitalExpenditures': 'PaymentsToAcquirePropertyPlantAndEquipment',
        'Depreciation': 'DepreciationDepletionAndAmortization'
    }
    
    walmart_financials = {}
    
    print("\nExtracting Walmart Financial Statements (2021-2024, 10-K only)...")
    
    for name, concept in financial_items.items():
        print(f"Getting {name}...")
        data = extract_financial_data(walmart_cik, headers, concept)
        if not data.empty:
            walmart_financials[name] = data
    
    return walmart_financials

def create_financial_statements(financial_data):
    """Create organized financial statements"""
    
    # Get years for columns
    years = [2021, 2022, 2023, 2024]
    
    def create_statement_df(items, statement_name):
        """Helper function to create statement dataframes"""
        statement_data = {}
        
        for item in items:
            if item in financial_data and not financial_data[item].empty:
                values = {}
                for _, row in financial_data[item].iterrows():
                    values[row['fy']] = row['val']
                
                # Create row with all years
                row_data = []
                for year in years:
                    row_data.append(values.get(year, np.nan))
                
                statement_data[item] = row_data
        
        df = pd.DataFrame(statement_data, index=[f'FY{year}' for year in years]).T
        return df
    
    # INCOME STATEMENT
    income_statement_items = [
        'Revenues', 'CostOfRevenue', 'GrossProfit', 'OperatingExpenses',
        'OperatingIncomeLoss', 'InterestExpense', 'IncomeTaxExpense', 'NetIncomeLoss'
    ]
    
    # BALANCE SHEET
    balance_sheet_items = [
        'Assets', 'AssetsCurrent', 'CashAndCashEquivalents', 'AccountsReceivableNet',
        'InventoryNet', 'PropertyPlantAndEquipmentNet', 'Goodwill', 
        'IntangibleAssetsNetExcludingGoodwill', 'Liabilities', 'LiabilitiesCurrent',
        'AccountsPayableCurrent', 'DebtCurrent', 'LongTermDebt', 'StockholdersEquity'
    ]
    
    # CASH FLOW STATEMENT  
    cash_flow_items = [
        'NetCashProvidedByUsedInOperatingActivities', 'NetCashProvidedByUsedInInvestingActivities',
        'NetCashProvidedByUsedInFinancingActivities', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect',
        'CapitalExpenditures', 'Depreciation'
    ]
    
    # Create statements
    income_statement = create_statement_df(income_statement_items, "Income Statement")
    balance_sheet = create_statement_df(balance_sheet_items, "Balance Sheet")
    cash_flow_statement = create_statement_df(cash_flow_items, "Cash Flow Statement")
    
    return {
        'Income_Statement': income_statement,
        'Balance_Sheet': balance_sheet,
        'Cash_Flow_Statement': cash_flow_statement
    }

def format_financial_display(statements):
    """Format financial statements for better display"""
    
    for statement_name, df in statements.items():
        print(f"\n{'='*60}")
        print(f"WALMART INC. - {statement_name.replace('_', ' ').upper()}")
        print(f"{'='*60}")
        print("(All amounts in USD)")
        print()
        
        # Format numbers in millions
        df_display = df.copy()
        for col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"${x/1000000:,.0f}M" if pd.notna(x) else "N/A"
            )
        
        print(df_display.to_string())
        print()

def save_to_excel(statements, financial_data):
    """Save financial statements to Excel file with multiple sheets"""
    
    try:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Walmart_Financial_Statements_2021-2024_{timestamp}.xlsx"
        
        print(f"\nSaving data to Excel file: {filename}")
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            currency_format = workbook.add_format({
                'num_format': '$#,##0',
                'border': 1
            })
            
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'fg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1
            })
            
            # Save each financial statement to separate sheet
            for statement_name, df in statements.items():
                sheet_name = statement_name.replace('_', ' ')
                
                # Write to Excel
                df.to_excel(writer, sheet_name=sheet_name, startrow=3, startcol=0)
                
                # Get worksheet to format
                worksheet = writer.sheets[sheet_name]
                
                # Add title
                worksheet.merge_range('A1:E1', f'WALMART INC. - {sheet_name.upper()}', title_format)
                worksheet.write('A2', 'All amounts in USD', header_format)
                
                # Format headers
                for col_num, value in enumerate(df.columns):
                    worksheet.write(3, col_num + 1, value, header_format)
                
                # Format row headers (financial items)
                for row_num, value in enumerate(df.index):
                    worksheet.write(row_num + 4, 0, value, header_format)
                
                # Format data cells with currency
                for row in range(len(df)):
                    for col in range(len(df.columns)):
                        if pd.notna(df.iloc[row, col]):
                            worksheet.write(row + 4, col + 1, df.iloc[row, col], currency_format)
                
                # Auto-adjust column widths
                worksheet.set_column('A:A', 35)  # Item names column
                worksheet.set_column('B:E', 15)  # Year columns
            
            # Create a raw data sheet with all extracted items
            raw_data_list = []
            for item, data in financial_data.items():
                for _, row in data.iterrows():
                    raw_data_list.append({
                        'Financial_Item': item,
                        'Fiscal_Year': row['fy'],
                        'End_Date': row['end'],
                        'Value': row['val']
                    })
            
            raw_df = pd.DataFrame(raw_data_list)
            
            if not raw_df.empty:
                raw_df.to_excel(writer, sheet_name='Raw Data', index=False, startrow=1)
                
                worksheet = writer.sheets['Raw Data']
                worksheet.merge_range('A1:D1', 'WALMART INC. - RAW EXTRACTED DATA', title_format)
                
                # Format headers
                for col_num, value in enumerate(raw_df.columns):
                    worksheet.write(1, col_num, value, header_format)
                
                # Format value column with currency
                for row in range(len(raw_df)):
                    worksheet.write(row + 2, 3, raw_df.iloc[row, 3], currency_format)
                
                worksheet.set_column('A:A', 40)  # Financial item column
                worksheet.set_column('B:B', 12)  # Year column
                worksheet.set_column('C:C', 12)  # Date column
                worksheet.set_column('D:D', 15)  # Value column
        
        print(f"✅ Excel file saved successfully: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error saving Excel file: {e}")
        print("Make sure you have xlsxwriter installed: pip install xlsxwriter")
        return None

# Main execution
if __name__ == "__main__":
    # Get financial data
    financial_data = get_walmart_financial_statements()
    
    if financial_data:
        print(f"\n{'='*60}")
        print("DATA EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Extracted {len(financial_data)} financial items")
        
        # Create organized financial statements
        statements = create_financial_statements(financial_data)
        
        # Display formatted statements (console output)
        format_financial_display(statements)
        
        # Save to Excel file
        excel_filename = save_to_excel(statements, financial_data)
        
        if excel_filename:
            print(f"\n🎉 SUCCESS! Excel file created: {excel_filename}")
        else:
            print("\n⚠️ Console output available, but Excel export failed")
        
        # Summary of available data
        print(f"\n{'='*60}")
        print("SUMMARY OF EXTRACTED DATA:")
        print(f"{'='*60}")
        for item, data in financial_data.items():
            years_available = sorted(data['fy'].unique()) if not data.empty else []
            print(f"{item}: {len(years_available)} years - {years_available}")
            
    else:
        print("Failed to extract financial data")
        
    print("\nScript completed!")