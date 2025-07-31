import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    """Load and clean the HR dataset"""
    
    np.random.seed(42)
    n_employees = 1470
    
    data = {
        'EmpID': [f'RM{i:03d}' for i in range(1, n_employees + 1)],
        'Age': np.random.randint(18, 61, n_employees),
        'Attrition': np.random.choice(['Yes', 'No'], n_employees, p=[0.16, 0.84]),
        'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], 
                                         n_employees, p=[0.7, 0.2, 0.1]),
        'Department': np.random.choice(['Research & Development', 'Sales', 'Human Resources'], 
                                     n_employees, p=[0.65, 0.30, 0.05]),
        'DistanceFromHome': np.random.randint(1, 30, n_employees),
        'Education': np.random.randint(1, 6, n_employees),
        'EducationField': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'], 
                                         n_employees, p=[0.35, 0.25, 0.15, 0.15, 0.05, 0.05]),
        'Gender': np.random.choice(['Male', 'Female'], n_employees, p=[0.6, 0.4]),
        'JobLevel': np.random.randint(1, 6, n_employees),
        'JobRole': np.random.choice(['Research Scientist', 'Laboratory Technician', 'Sales Executive', 
                                   'Manufacturing Director', 'Healthcare Representative', 'Manager', 
                                   'Sales Representative', 'Human Resources', 'Research Director'], 
                                  n_employees, p=[0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05]),
        'JobSatisfaction': np.random.randint(1, 5, n_employees),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_employees, p=[0.4, 0.45, 0.15]),
        'MonthlyIncome': np.random.normal(6500, 4000, n_employees).astype(int),
        'OverTime': np.random.choice(['Yes', 'No'], n_employees, p=[0.3, 0.7]),
        'PerformanceRating': np.random.choice([3, 4], n_employees, p=[0.85, 0.15]),
        'TotalWorkingYears': np.random.randint(0, 41, n_employees),
        'WorkLifeBalance': np.random.randint(1, 5, n_employees),
        'YearsAtCompany': np.random.randint(0, 40, n_employees),
        'YearsInCurrentRole': np.random.randint(0, 19, n_employees),
        'YearsSinceLastPromotion': np.random.randint(0, 16, n_employees),
        'YearsWithCurrManager': np.random.randint(0, 18, n_employees),
    }
    
    df = pd.DataFrame(data)
    
    # Data cleaning
    df['MonthlyIncome'] = np.where(df['MonthlyIncome'] < 1000, 1000, df['MonthlyIncome'])
    df['MonthlyIncome'] = np.where(df['MonthlyIncome'] > 20000, 20000, df['MonthlyIncome'])
    
    # Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                           labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    
    # Create salary slabs
    df['SalarySlab'] = pd.cut(df['MonthlyIncome'], 
                             bins=[0, 5000, 10000, 15000, float('inf')],
                             labels=['Upto 5k', '5k-10k', '10k-15k', '15k+'])
    
    return df

def create_key_metrics(df):
    """Create key metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_employees = len(df)
    attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
    avg_salary = df['MonthlyIncome'].mean()
    avg_satisfaction = df['JobSatisfaction'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Total Employees</h3>
            <h2>{total_employees}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Attrition Rate</h3>
            <h2>{attrition_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Avg Salary</h3>
            <h2>${avg_salary:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⭐ Avg Satisfaction</h3>
            <h2>{avg_satisfaction:.1f}/5</h2>
        </div>
        """, unsafe_allow_html=True)

def create_attrition_analysis(df):
    """Create attrition analysis visualizations"""
    st.header("🔍 Attrition Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attrition by Department
        dept_attrition = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        dept_attrition_pct = dept_attrition.div(dept_attrition.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            x=dept_attrition_pct.index,
            y=dept_attrition_pct['Yes'],
            title="Attrition Rate by Department",
            labels={'x': 'Department', 'y': 'Attrition Rate (%)'},
            color=dept_attrition_pct['Yes'],
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Attrition by Age Group
        age_attrition = df.groupby(['AgeGroup', 'Attrition']).size().unstack(fill_value=0)
        age_attrition_pct = age_attrition.div(age_attrition.sum(axis=1), axis=0) * 100
        
        fig = px.line(
            x=age_attrition_pct.index,
            y=age_attrition_pct['Yes'],
            title="Attrition Rate by Age Group",
            labels={'x': 'Age Group', 'y': 'Attrition Rate (%)'},
            markers=True
        )
        fig.update_traces(line_color='#ff6b6b', marker_size=8)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_salary_analysis(df):
    """Create salary analysis visualizations"""
    st.header("💰 Salary Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary Distribution by Department
        fig = px.box(
            df, 
            x='Department', 
            y='MonthlyIncome',
            title="Salary Distribution by Department",
            color='Department'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Salary vs Performance
        fig = px.scatter(
            df,
            x='PerformanceRating',
            y='MonthlyIncome',
            color='Department',
            size='YearsAtCompany',
            title="Salary vs Performance Rating",
            hover_data=['JobRole', 'YearsAtCompany']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_demographic_analysis(df):
    """Create demographic analysis"""
    st.header("👥 Demographic Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Gender Distribution
        gender_counts = df['Gender'].value_counts()
        fig = px.pie(
            values=gender_counts.values,
            names=gender_counts.index.tolist(),  # Corrected line
            title="Gender Distribution",
            color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Education Field Distribution
        edu_counts = df['EducationField'].value_counts().head(6)
        fig = px.bar(
            x=edu_counts.values,
            y=edu_counts.index.tolist(), # Corrected line
            orientation='h',
            title="Top Education Fields",
            color=edu_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Marital Status Distribution
        marital_counts = df['MaritalStatus'].value_counts()
        fig = px.pie(
            values=marital_counts.values,
            names=marital_counts.index.tolist(), # Corrected line
            title="Marital Status Distribution",
            hole=0.4, # Use a donut chart instead of a pie chart
            color_discrete_sequence=['#ffcc99', '#ff9999', '#99ccff']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    st.header("🔗 Feature Correlations")
    
    # Select numeric columns
    numeric_cols = ['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'JobSatisfaction',
                   'MonthlyIncome', 'PerformanceRating', 'TotalWorkingYears', 'WorkLifeBalance',
                   'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def create_predictive_model(df):
    """Create and display predictive modeling results"""
    st.header("🤖 Attrition Prediction Model")
    
    # Prepare data for modeling
    model_df = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
                       'JobRole', 'MaritalStatus', 'OverTime']
    
    for col in categorical_cols:
        model_df[col + '_encoded'] = le.fit_transform(model_df[col])
    
    # Select features
    feature_cols = ['Age', 'DistanceFromHome', 'Education', 'JobLevel', 'JobSatisfaction',
                   'MonthlyIncome', 'PerformanceRating', 'TotalWorkingYears', 'WorkLifeBalance',
                   'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                   'YearsWithCurrManager'] + [col + '_encoded' for col in categorical_cols]
    
    X = model_df[feature_cols]
    y = (model_df['Attrition'] == 'Yes').astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Feature Importance",
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model performance metrics
        y_pred = rf_model.predict(X_test)
        accuracy = rf_model.score(X_test, y_test)
        
        st.metric("Model Accuracy", f"{accuracy:.3f}")
        st.metric("Training Samples", len(X_train))
        st.metric("Test Samples", len(X_test))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            title="Confusion Matrix",
            labels={'x': 'Predicted', 'y': 'Actual'},
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">HR Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive Analysis of Employee Data and Attrition Patterns")
    
    # Load data
    df = load_and_clean_data()
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    # Department filter
    departments = ['All'] + sorted(df['Department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Select Department", departments)
    
    # Age group filter
    age_groups = ['All'] + sorted(df['AgeGroup'].unique().tolist())
    selected_age = st.sidebar.selectbox("Select Age Group", age_groups)
    
    # Apply filters
    if selected_dept != 'All':
        df = df[df['Department'] == selected_dept]
    
    if selected_age != 'All':
        df = df[df['AgeGroup'] == selected_age]
    
    # Display key metrics
    create_key_metrics(df)
    
    st.markdown("---")
    
    # Create different analysis sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Attrition Analysis", "Salary Analysis", "Demographics", "Correlations", "Predictive Model"])
    
    with tab1:
        create_attrition_analysis(df)
    
    with tab2:
        create_salary_analysis(df)
    
    with tab3:
        create_demographic_analysis(df)
    
    with tab4:
        create_correlation_heatmap(df)
    
    with tab5:
        create_predictive_model(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>HR Analytics Dashboard | Built with Streamlit & Plotly</p>
        <p>Data insights for better workforce management</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
