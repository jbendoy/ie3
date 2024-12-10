import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules


@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_survey_results.csv')
    return data

data = load_data()
st.markdown(
    """
    <style>
    .streamlit-table {
        width: 100%;
        margin: 0 auto;
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title('Main Menu')
section = st.sidebar.radio("Go to", [
    "Introduction",
    "Data Preparation",
    "K-means Clustering",
    "Linear Regression",
    "Descriptive Statistics & Visualization",
    "Apriori Algorithm"
])

# Introduction Section
if section == "Introduction":
    st.image('IE3_Cover.png', use_column_width=True)
    st.title('Stack Overflow Developer Survey Analysis Report')
    
    st.header('Introduction')
    st.write("""
    This report explores the Stack Overflow Developer Survey dataset, a comprehensive resource containing responses from developers worldwide. The dataset offers detailed insights into the global tech industry, focusing on job roles, skills, technologies used, and salary information. This dataset serves as a valuable tool for understanding the dynamics of the tech job market and analyzing trends across various developer demographics.
    Key attributes covered in the dataset include:

    - Demographic Information: Age, education level, country of residence, employment status, and job satisfaction.
    - Work Experience and Job Roles: Years of coding experience, job responsibilities, tools, and technologies used in professional settings.
    - Salary Information: Total compensation, job satisfaction ratings, and factors influencing earnings.
    - Technology Usage: Programming languages, platforms, databases, tools, and operating systems.
    - Artificial Intelligence (AI) Engagement: Developers’ usage of AI tools, challenges in AI adoption, and ethical considerations.

    By analyzing this data, we aim to explore how factors like job roles, geographic locations, years of experience, and technology usage impact developer salaries and satisfaction.
    """)

    st.markdown("---")
    
    # Checkbox to toggle visibility
    if st.checkbox("Show Data Analysis Technique"):
        st.title("Data Analysis Techniques")
        
        st.markdown("""
        ## Overview
        This project applies several data analysis techniques to extract meaningful insights from the Stack Overflow Developer Survey dataset. Each technique is tailored to address specific research questions and explore trends in developer demographics, job roles, and compensation.
        """)

        # Expanders for detailed techniques
        with st.expander("a. Clustering (K-Means Clustering)"):
            st.markdown("""
            - **Purpose**: Group the data based on shared characteristics.
            - **Variables**:
              - Years of Experience: YearsCode, YearsCodePro
              - Job Role and Type: DevType, OrgSize, Industry
              - Salary Information: CompTotal
              - Technology Usage: Language, Database, Platform, ToolsTech, MiscTech
            - **Outcome**: Identify patterns and group developers with similar attributes for deeper analysis.
            """)

        with st.expander("b. Linear Regression"):
            st.markdown("""
            - **Purpose**: Predict developers’ salaries (CompTotal) based on independent variables.
            - **Key Variables**:
              - Years of Experience: YearsCodePro, YearsCode
              - Education Level: EdLevel
              - Industry: Industry
              - Job Satisfaction: JobSat, JobSatPoints
              - Technology Usage: Tools, platforms, and languages
            - **Outcome**: Quantify the influence of each factor on salary and identify key correlating aspects.
            """)

        with st.expander("c. Descriptive Statistics & Visualization"):
            st.markdown("""
            - **Purpose**: Summarize and visualize the data.
            - **Analysis**:
              - Salary distributions across regions, job roles, and experience levels.
              - Summary statistics for demographic variables (e.g., average age, coding experience, and job satisfaction).
            - **Visualizations**:
              - Histograms
              - Boxplots
              - Scatter plots
            """)

        with st.expander("d. Apriori Algorithm"):
            st.markdown("""
            - **Purpose**: Explore relationships between technology usage.
            - **Examples**:
              - Do developers using certain languages (e.g., Python, JavaScript) also use specific tools or frameworks?
              - Which technologies (e.g., cloud computing, databases) are frequently used together in job roles?
            - **Outcome**: Uncover patterns of technology adoption and tool combinations.
            """)

# Data Preparation Section
if section == "Data Preparation":
    st.title("Data Preparation")
    st.write("""
    In this section, we clean and prepare the data for analysis. This includes handling missing values, 
    dropping columns with excessive missing data, and imputing missing values where appropriate.
    """)

    # Show the first few rows of the original dataset
    st.subheader("Original Dataset Preview")
    st.write(data.head())

    # Display column information
    column_info = pd.DataFrame({
        'Column Name': data.columns,
        'Missing Values': data.isnull().sum(),
        'Data Type': data.dtypes
    }).reset_index(drop=True)

    st.subheader("Column Information")
    st.write(column_info)

    # Calculate the percentage of missing values
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    high_missing_cols = missing_percentage[missing_percentage > 50]

    # Show remaining columns after cleaning
    st.subheader("Remaining Columns After Cleaning")
    remaining_columns = data_cleaned.columns
    st.write(remaining_columns)

    # Fill missing numerical values with the median
    numerical_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
    data_cleaned[numerical_cols] = data_cleaned[numerical_cols].fillna(data_cleaned[numerical_cols].median())

    # Fill missing categorical values with the mode
    categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
    data_cleaned[categorical_cols] = data_cleaned[categorical_cols].fillna(data_cleaned[categorical_cols].mode().iloc[0])

    # Display missing values after imputation
    st.subheader("Missing Values After Imputation")
    missing_after_imputation = {
        'Numerical Columns': data_cleaned[numerical_cols].isnull().sum(),
        'Categorical Columns': data_cleaned[categorical_cols].isnull().sum()
    }
    st.write(missing_after_imputation)

    # Display the cleaned data preview
    st.subheader("Cleaned Dataset Preview")
    st.write(data_cleaned.head())

    # Save the cleaned dataset
    data_cleaned.to_csv('cleaned_survey_results.csv', index=False)

# K-means Clustering Section
if section == "K-means Clustering":
    st.title("K-means Clustering")
    
    # Brief Explanation
    st.write("K-means clustering is applied to group respondents based on their education level, coding experience, and professional coding experience. The optimal number of clusters is determined using the Elbow Method, and the results are visualized.")

    # Import libraries
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load the cleaned dataset
    data = load_data()  # Assumes load_data() is defined above

    # Columns to drop based on irrelevance or redundancy
    columns_to_drop = ['ResponseId', 'Unnamed: 17', 'Currency']
    data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

    # Selecting relevant columns for clustering
    selected_columns = ['EdLevel', 'YearsCode', 'YearsCodePro']
    data_numeric = data_cleaned[selected_columns]

    # Handling missing values using mean imputation for numerical columns
    imputer = SimpleImputer(strategy='most_frequent')  # 'most_frequent' is used for categorical data
    data_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)

    # Converting categorical columns to numerical using Label Encoding
    label_encoders = {}
    for column in ['EdLevel', 'YearsCode', 'YearsCodePro']:
        label_encoders[column] = LabelEncoder()
        data_imputed[column] = label_encoders[column].fit_transform(data_imputed[column])

    # Scaling the data for clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    # Finding the optimal number of clusters using the Elbow Method
    inertia = []
    range_k = range(1, 11)
    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow Curve to determine the optimal number of clusters
    plt.figure(figsize=(8, 5))
    plt.plot(range_k, inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    st.pyplot(plt)

    # Applying K-Means with the chosen number of clusters (optimal_k = 4)
    optimal_k = 4  # Adjust based on elbow curve results
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Adding the cluster labels to the dataset
    data_imputed['Cluster'] = clusters

    # Visualizing the clusters using a pair plot
    data_imputed['Cluster'] = data_imputed['Cluster'].astype(str)  # Convert cluster labels to strings for visualization
    sns.pairplot(data_imputed, hue='Cluster', diag_kind='kde', corner=True)
    st.pyplot(plt)

    # Saving the clustered dataset to a new CSV file
    data_imputed.to_csv('clustered_survey_data.csv', index=False)

# Linear Regression Section
if section == "Linear Regression":
    st.title("Linear Regression")

    # Display the explanation
    st.write("""
    This linear regression model predicts developers' salaries (CompTotal) using key factors like years of experience (YearsCode, YearsCodePro), education level (EdLevel), and job role (DevType). 
    The data is cleaned by converting categorical experience values to numeric and encoding categorical features using one-hot encoding. 
    The model is trained on 80% of the data and tested on 20% to evaluate performance using Mean Squared Error (MSE) and R-squared (R²). 
    The most influential features are identified, highlighting which factors have the strongest impact on salary, providing insights into how experience, education, and role affect developer compensation.
    """)

    try:
        # Data Cleaning (already done but included here for clarity)
        data = data.copy()

        # Handle 'YearsCode' and 'YearsCodePro' conversion
        data['YearsCode'] = data['YearsCode'].replace({'Less than 1 year': 0.5, 'More than 50 years': 50}).apply(pd.to_numeric, errors='coerce')
        data['YearsCodePro'] = data['YearsCodePro'].replace({'Less than 1 year': 0.5, 'More than 50 years': 50}).apply(pd.to_numeric, errors='coerce')

        # Drop rows with missing target or independent variables
        data = data.dropna(subset=['CompTotal', 'YearsCode', 'YearsCodePro', 'EdLevel', 'DevType'])

        # One-hot encode categorical columns
        data = pd.get_dummies(data, columns=['EdLevel', 'DevType'], drop_first=True)

        # Select relevant columns for the regression
        X = data[['YearsCode', 'YearsCodePro'] + [col for col in data.columns if 'EdLevel_' in col or 'DevType_' in col]]
        y = data['CompTotal']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model's performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Get the coefficients and their corresponding features
        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', key=abs, ascending=False)

        # Display the results
        st.subheader("Model Evaluation")
        st.write(f'Mean Squared Error (MSE): {mse:.2f}')
        st.write(f'R-squared (R²): {r2:.2f}')

        st.subheader("Top 10 Most Influential Features")
        st.write(coefficients.head(10))

    except Exception as e:
        st.error(f"Error loading or processing the data: {e}")

# Descriptive Statistics & Visualization Section
if section == "Descriptive Statistics & Visualization":
    st.title("Descriptive Statistics & Visualization")
    
    # Brief Explanation
    st.write("This section provides an overview of the dataset's descriptive statistics and visualizations, including distributions of coding experience, age, salary, and country-specific data.")

    # Load the cleaned dataset (assuming load_data() is defined above)
    data = load_data()

    # Display basic information about the dataset
    st.write("Displaying first few rows of the dataset:")
    st.write(data.head())

    # Plot the top 10 countries by respondent count
    if 'Country' in data.columns:
        country_counts = data['Country'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=country_counts.index, y=country_counts.values, palette="Blues_d")
        plt.title('Top 10 Countries by Respondents')
        plt.xlabel('Country')
        plt.ylabel('Number of Respondents')
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        st.write("No 'Country' column found in the dataset.")

    # Handle missing or invalid values for 'YearsCode' and 'YearsCodePro' and convert them to numeric values
    data = data.dropna(subset=['YearsCode', 'YearsCodePro'])
    data['YearsCode'] = pd.to_numeric(data['YearsCode'], errors='coerce')
    data['YearsCodePro'] = pd.to_numeric(data['YearsCodePro'], errors='coerce')
    data = data.dropna(subset=['YearsCode', 'YearsCodePro'])

    # Plot distribution of coding experience (YearsCode) and professional experience (YearsCodePro)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data['YearsCode'], kde=True, color='skyblue', bins=20)
    plt.title('Distribution of Years of Coding Experience')
    plt.xlabel('Years of Coding Experience')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    sns.histplot(data['YearsCodePro'], kde=True, color='salmon', bins=20)
    plt.title('Distribution of Professional Coding Experience')
    plt.xlabel('Years of Professional Coding Experience')
    plt.ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(plt)

    # Plot Age Distribution of Respondents
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
    plt.title('Age Distribution of Respondents')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

    # Boxplot for Age Distribution by Top 15 Countries
    top_countries = data['Country'].value_counts().head(15).index
    filtered_data = data[data['Country'].isin(top_countries)]
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=filtered_data, x='Country', y='Age')
    plt.title('Age Distribution by Top 15 Countries')
    plt.xlabel('Country')
    plt.ylabel('Age')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

    # Bar Chart for Average Salary by Education Level
    avg_salary_by_education = data.groupby('EdLevel')['CompTotal'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    avg_salary_by_education.plot(kind='bar', color='orange')
    plt.title('Average Salary by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Average Salary')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    st.pyplot(plt)

    # Cleaning 'YearsCodePro' and 'CompTotal' columns for better analysis
    data['YearsCodePro'] = data['YearsCodePro'].replace({
        'Less than 1 year': 0.5,
        'More than 50 years': 50,
        'NA': np.nan
    }).astype(float)

    data['CompTotal'] = data['CompTotal'].replace('NA', np.nan)  # Replace 'NA' with NaN
    data['CompTotal'] = pd.to_numeric(data['CompTotal'], errors='coerce')  # Convert to numeric

    # Remove outliers in 'CompTotal' and plot relationship with 'YearsCodePro'
    q99 = data['CompTotal'].quantile(0.99)  # 99th percentile for outlier removal
    data_cleaned = data[(data['CompTotal'] <= q99) & (data['CompTotal'] > 0)].dropna(subset=['YearsCodePro', 'CompTotal'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_cleaned, x='YearsCodePro', y='CompTotal', alpha=0.6)
    sns.regplot(data=data_cleaned, x='YearsCodePro', y='CompTotal', scatter=False, color='red')
    plt.title('Salary vs. Professional Years of Experience')
    plt.xlabel('Professional Years of Experience')
    plt.ylabel('Salary')
    plt.grid(True)
    st.pyplot(plt)

    # Heatmap for Numeric Correlations between Age, YearsCodePro, CompTotal, and YearsCode
    numeric_cols = ['Age', 'YearsCodePro', 'CompTotal', 'YearsCode']
    corr_matrix = data[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix for Age, Years of Experience, and Salary')
    st.pyplot(plt)

#----------------------------------------------------------------------------------------------------------------------

# Apriori Algorithm Section
if section == "Apriori Algorithm":
    st.title("Apriori Algorithm")

    # Brief Explanation
    st.write(
        "The Apriori algorithm is used to find frequent itemsets and generate association rules, "
        "helping identify patterns in how developers use various technologies. It reveals common "
        "combinations of tools and technologies, providing insights into how certain technologies "
        "are grouped together in real-world scenarios."
    )

    # Load the cleaned dataset
    df_cleaned_apriori = load_data()

    # Define the columns to process (update based on your data)
    columns_to_encode = [
        'LanguageHaveWorkedWith',
        'DatabaseHaveWorkedWith',
        'WebframeHaveWorkedWith',
        'ToolsTechHaveWorkedWith',
        'DevType'
    ]

    # Create binary matrix
    binary_df = pd.DataFrame()

    for col in columns_to_encode:
        if col in df_cleaned_apriori.columns:
            split_data = df_cleaned_apriori[col].str.get_dummies(sep=';')
            binary_df = pd.concat([binary_df, split_data], axis=1)

    # Convert binary matrix to boolean type
    binary_df_bool = binary_df.astype(bool)

    # Apply the Apriori algorithm
    frequent_itemsets = apriori(binary_df_bool, min_support=0.05, use_colnames=True)

    # Generate association rules
    num_itemsets = len(frequent_itemsets)
    rules = association_rules(frequent_itemsets, num_itemsets=num_itemsets, metric="lift", min_threshold=1.0)

    # Sort and display top rules
    rules = rules.sort_values(by='lift', ascending=False)
    top_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)

    # Display the top association rules table in Streamlit using st.table
    st.write("Top 10 association rules between tools (sorted by lift):")
    st.table(top_rules)

    # ------------------------------------------------------------------------------------------------------------------------------

    # Exploring relationships between employment factors and technology usage
    employment_columns = ['Employment', 'RemoteWork', 'OrgSize']
    tech_columns = [
        'LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith',
        'WebframeHaveWorkedWith', 'ToolsTechHaveWorkedWith'
    ]

    # Convert employment data to binary
    binary_employment = pd.get_dummies(df_cleaned_apriori[employment_columns], prefix=employment_columns).astype(bool)
    binary_tech = pd.DataFrame()

    for col in tech_columns:
        if col in df_cleaned_apriori.columns:
            split_data = df_cleaned_apriori[col].str.get_dummies(sep=';').astype(bool)
            binary_tech = pd.concat([binary_tech, split_data], axis=1)

    # Combine employment and tech binary data
    binary_data = pd.concat([binary_employment, binary_tech], axis=1)

    # Apply Apriori algorithm
    frequent_itemsets = apriori(binary_data, min_support=0.05, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, num_itemsets=num_itemsets, metric="lift", min_threshold=1.0)

    # Filter and sort the rules
    rules = rules.sort_values(by='lift', ascending=False)
    top_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)

    # Display the filtered and sorted rules table in Streamlit using st.table
    st.write("Top 10 association rules for Employment and Technology:")
    st.table(top_rules)
    # 