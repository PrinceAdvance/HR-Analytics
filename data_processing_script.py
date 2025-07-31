import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class HRDataProcessor:
    """
    A comprehensive class for processing HR Analytics data
    """
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        
    def load_data(self):
        """Load the CSV data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Perform initial data exploration"""
        if self.df is None:
            print("Please load data first!")
            return
        
        print("="*50)
        print("DATA EXPLORATION REPORT")
        print("="*50)
        
        print("\n1. BASIC INFO:")
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        print("\n2. DATA TYPES:")
        print(self.df.dtypes)
        
        print("\n3. MISSING VALUES:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print("\n4. BASIC STATISTICS:")
        print(self.df.describe())
        
        print("\n5. CATEGORICAL VARIABLES:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts().head())
        
        return self.generate_data_quality_report()
    
    def generate_data_quality_report(self):
        """Generate a comprehensive data quality report"""
        report = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_records': self.df.duplicated().sum(),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        print("\n" + "="*50)
        print("DATA QUALITY REPORT")
        print("="*50)
        for key, value in report.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return report
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.df is None:
            print("Please load data first!")
            return
        
        print("\n" + "="*50)
        print("DATA CLEANING PROCESS")
        print("="*50)
        
        # Create a copy for processing
        self.processed_df = self.df.copy()
        
        # 1. Handle missing values
        print("\n1. Handling missing values...")
        initial_missing = self.processed_df.isnull().sum().sum()
        
        # Fill missing values based on column type
        for column in self.processed_df.columns:
            if self.processed_df[column].isnull().sum() > 0:
                if self.processed_df[column].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    self.processed_df[column].fillna(self.processed_df[column].median(), inplace=True)
                else:
                    # Fill categorical columns with mode
                    self.processed_df[column].fillna(self.processed_df[column].mode()[0], inplace=True)
        
        final_missing = self.processed_df.isnull().sum().sum()
        print(f"Missing values reduced from {initial_missing} to {final_missing}")
        
        # 2. Remove duplicates
        print("\n2. Removing duplicates...")
        initial_rows = len(self.processed_df)
        self.processed_df.drop_duplicates(inplace=True)
        final_rows = len(self.processed_df)
        print(f"Removed {initial_rows - final_rows} duplicate rows")
        
        # 3. Handle outliers (for salary and other numeric columns)
        print("\n3. Handling outliers...")
        numeric_columns = ['MonthlyIncome', 'DailyRate', 'HourlyRate', 'MonthlyRate']
        
        for col in numeric_columns:
            if col in self.processed_df.columns:
                Q1 = self.processed_df[col].quantile(0.25)
                Q3 = self.processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                self.processed_df[col] = np.where(self.processed_df[col] < lower_bound, lower_bound, self.processed_df[col])
                self.processed_df[col] = np.where(self.processed_df[col] > upper_bound, upper_bound, self.processed_df[col])
        
        # 4. Create derived features
        print("\n4. Creating derived features...")
        
        # Age groups
        if 'Age' in self.processed_df.columns:
            self.processed_df['AgeGroup'] = pd.cut(self.processed_df['Age'], 
                                                  bins=[0, 25, 35, 45, 55, 100], 
                                                  labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Salary slabs
        if 'MonthlyIncome' in self.processed_df.columns:
            self.processed_df['SalarySlab'] = pd.cut(self.processed_df['MonthlyIncome'], 
                                                    bins=[0, 5000, 10000, 15000, float('inf')],
                                                    labels=['Upto 5k', '5k-10k', '10k-15k', '15k+'])
        
        # Experience categories
        if 'TotalWorkingYears' in self.processed_df.columns:
            self.processed_df['ExperienceLevel'] = pd.cut(self.processed_df['TotalWorkingYears'],
                                                         bins=[0, 2, 10, 20, float('inf')],
                                                         labels=['Entry', 'Mid', 'Senior', 'Expert'])
        
        # Work-life balance categories
        if 'WorkLifeBalance' in self.processed_df.columns:
            balance_mapping = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
            self.processed_df['WorkLifeBalanceCategory'] = self.processed_df['WorkLifeBalance'].map(balance_mapping)
        
        print("Data cleaning completed successfully!")
        return self.processed_df
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        if self.processed_df is None:
            print("Please clean data first!")
            return
        
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # 1. Attrition Analysis
        print("\n1. ATTRITION ANALYSIS:")
        if 'Attrition' in self.processed_df.columns:
            attrition_rate = (self.processed_df['Attrition'] == 'Yes').mean() * 100
            print(f"Overall Attrition Rate: {attrition_rate:.2f}%")
            
            # Attrition by department
            if 'Department' in self.processed_df.columns:
                dept_attrition = self.processed_df.groupby('Department')['Attrition'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                ).sort_values(ascending=False)
                print("\nAttrition Rate by Department:")
                for dept, rate in dept_attrition.items():
                    print(f"  {dept}: {rate:.2f}%")
        
        # 2. Salary Analysis
        print("\n2. SALARY ANALYSIS:")
        if 'MonthlyIncome' in self.processed_df.columns:
            print(f"Average Salary: ${self.processed_df['MonthlyIncome'].mean():,.2f}")
            print(f"Median Salary: ${self.processed_df['MonthlyIncome'].median():,.2f}")
            print(f"Salary Range: ${self.processed_df['MonthlyIncome'].min():,.2f} - ${self.processed_df['MonthlyIncome'].max():,.2f}")
            
            # Salary by department
            if 'Department' in self.processed_df.columns:
                dept_salary = self.processed_df.groupby('Department')['MonthlyIncome'].mean().sort_values(ascending=False)
                print("\nAverage Salary by Department:")
                for dept, salary in dept_salary.items():
                    print(f"  {dept}: ${salary:,.2f}")
        
        # 3. Demographics
        print("\n3. DEMOGRAPHIC ANALYSIS:")
        demographic_cols = ['Gender', 'MaritalStatus', 'EducationField']
        for col in demographic_cols:
            if col in self.processed_df.columns:
                print(f"\n{col} Distribution:")
                distribution = self.processed_df[col].value_counts(normalize=True) * 100
                for category, percentage in distribution.items():
                    print(f"  {category}: {percentage:.1f}%")
        
        return self.processed_df
    
    def build_predictive_model(self):
        """Build a predictive model for attrition"""
        if self.processed_df is None or 'Attrition' not in self.processed_df.columns:
            print("Please clean data first and ensure Attrition column exists!")
            return
        
        print("\n" + "="*50)
        print("BUILDING PREDICTIVE MODEL")
        print("="*50)
        
        # Prepare features
        model_df = self.processed_df.copy()
        
        # Select relevant features
        feature_columns = []
        
        # Numeric features
        numeric_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'TotalWorkingYears',
                           'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                           'YearsWithCurrManager']
        
        for col in numeric_features:
            if col in model_df.columns:
                feature_columns.append(col)
        
        # Categorical features (encode them)
        categorical_features = ['Department', 'JobRole', 'MaritalStatus', 'Gender', 
                               'EducationField', 'BusinessTravel', 'OverTime']
        
        le = LabelEncoder()
        for col in categorical_features