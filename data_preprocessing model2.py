import pandas as pd

# Load the dataset
df = pd.read_csv('dataset.csv')

def clean_data(df):
    """
    Clean and preprocess the dataset.
    
    This function handles:
    - Dropping unnecessary columns.
    - Removing duplicate rows.
    - Filling missing values for quantitative and qualitative data.
    - Filtering out rows with unrealistic or outlier values.
    
    Parameters:
    - df: pandas DataFrame, the raw dataset.
    
    Returns:
    - df: pandas DataFrame, the cleaned dataset.
    """
  
    '''# Drop the 'District' column
    df = df.drop('District', axis=1)'''

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing values for quantitative data with median
    quantitative_columns = {
        'District': df['District'].mode()[0],
        'BathroomCount': df['BathroomCount'].median(),
        'BedroomCount': df['BedroomCount'].median(),
        'ConstructionYear': df['ConstructionYear'].median(),
        'LivingArea': df['LivingArea'].median(),
        'MonthlyCharges': df['MonthlyCharges'].median(),
        'NumberOfFacades': df['NumberOfFacades'].median(),
        'RoomCount': df['RoomCount'].median(),
        'ShowerCount': df['ShowerCount'].median(),
        'SurfaceOfPlot': df['SurfaceOfPlot'].median(),
        'ToiletCount': df['ToiletCount'].median(),
        'Price': df['Price'].median()
    }
    df = df.fillna(quantitative_columns)

    # Filter out rows with unrealistic or outlier values
    df = df[
        (df['Price'] >= 10000) & (df['Price'] <= 10000000) &
        (df['LivingArea'] >= 10) & (df['LivingArea'] <= 1000) &
        (df['SurfaceOfPlot'] >= 10) & (df['SurfaceOfPlot'] <= 10000) &
        (df['RoomCount'] >= 1) & (df['RoomCount'] <= 20) &
        (df['NumberOfFacades'] >= 1) & (df['NumberOfFacades'] <= 4) &
        (df['ShowerCount'] >= 1) & (df['ShowerCount'] <= 5) &
        (df['ToiletCount'] >= 1) & (df['ToiletCount'] <= 5) &
        (df['MonthlyCharges'] >= 0) & (df['MonthlyCharges'] <= 10000) &
        (df['ConstructionYear'] >= 1700) & (df['ConstructionYear'] <= 2024) &
        (df['BedroomCount'] >= 0) & (df['BedroomCount'] <= 20) &
        (df['BathroomCount'] >= 0) & (df['BathroomCount'] <= 20)
    ]

    # Fill missing values for qualitative data with the most frequent value
    qualitative_columns = {
        'Furnished': df['Furnished'].mode()[0],
        'Garden': df['Garden'].mode()[0],
        'SwimmingPool': df['SwimmingPool'].mode()[0],
        'Terrace': df['Terrace'].mode()[0],
        'Kitchen': df['Kitchen'].mode()[0],
        'PEB': df['PEB'].mode()[0],
        'Province': df['Province'].mode()[0],
        'StateOfBuilding': df['StateOfBuilding'].mode()[0]
    }
    df = df.fillna(qualitative_columns)

    return df

# Apply the cleaning function
df = clean_data(df)

# Save the cleaned dataset to a CSV file
save_path = 'cleaned_data.csv'
df.to_csv(save_path, index=False)
print(f"Data saved to {save_path}")
