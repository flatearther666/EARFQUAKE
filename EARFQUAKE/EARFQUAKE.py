#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#######################
# Page configuration
st.set_page_config(
    page_title="EARFQUAKE Analysis",  # Updated Project Title
    page_icon="🌍",  # Updated with an Earth emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:
    # Sidebar Title
    st.title('EARFQUAKE')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("""
    1. Aguas, Yñikko Arzee Neo
    2. Almandres, Villy Joel
    3. Macabales, Carl Emmanuel
    4. Macatangay, Robin Jairic
    5. Perico, Frederick Lemuel 
    """)

#######################
# Data

# Load data once to avoid redundancy
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

dataset_path = "EARFQUAKE/earthquakes.csv"
try:
    df = load_data(dataset_path)
except FileNotFoundError:
    st.error(f"File not found at path: {dataset_path}")
    st.stop()

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown("""
    This is a Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to analyze the **Global Earthquake Data** dataset.

    #### Pages
    1. `Dataset` - Brief description of the Global Earthquake Data dataset used in this dashboard. 
    2. `EDA` - Exploratory Data Analysis of the Global Earthquake Data dataset.
    3. `Data Cleaning / Pre-processing` - Cleaning, preprocessing, and feature engineering to ensure data quality and model suitability.
    4. `Machine Learning` - Training and evaluating supervised machine learning models to predict earthquake occurrences and related factors.
    5. `Prediction` - Utilizing trained models to make predictions on new, unseen data.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.
    """)

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.markdown("""
    The `Global Earthquake Data` dataset was uploaded by **Shreya Sur**. This dataset contains detailed information on **1,137 earthquakes** worldwide, including **magnitude**, **depth**, **location**, **time**, and seismic measurements. It's ideal for analyzing global seismic activity, developing machine learning models, and studying the impact of earthquakes.
    
    ### Key Features:
    1. Comprehensive coverage of global earthquakes.
    2. Detailed location information.
    3. Seismic measurements and intensity data.
    4. Tsunami and alert information.

    ### Content:        
    The dataset contains **1,137 rows** with **43** primary attributes related to earthquake events. 
        
    The essential columns are as follows: 
    - **Magnitude**
    - **Depth**
    - **Latitude**
    - **Longitude**  
    - **Date**  
    - **Time**    
    - **Type**     
    - **Location**       
    - **Continent**       
    - **Country**         
    - **Tsunami Presence**
        
    Additionally, it includes attributes detailing the characteristics and locations of each earthquake event. 
    This dataset provides comprehensive information for analyzing earthquake patterns and effects across various regions.

    `Dataset Link:` [Kaggle - Recent Earthquakes](https://www.kaggle.com/datasets/shreyasur965/recent-earthquakes/data)                
    """)

    # Display data overview sections
    st.header("Dataset Overview")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Data Summary")
    st.dataframe(df.describe())

    st.subheader("Null Values")
    st.dataframe(df.isnull().sum())

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    plt.style.use('dark_background')  # Set dark background for all plots

    # Create three columns for better layout
    cols = st.columns((3, 3, 3), gap='medium')

    # Magnitude Distribution
    with cols[0]:
        st.markdown('#### Magnitude Distribution')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.kdeplot(df['magnitude'], shade=True, ax=ax, color='skyblue')
        ax.set_title("Magnitude Distribution (KDE)", color='white')
        ax.set_xlabel("Magnitude", color='white')
        ax.set_ylabel("Density", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Magnitude vs Depth
    with cols[1]:
        st.markdown('#### Magnitude vs Depth')
        fig, ax = plt.subplots(figsize=(8, 5))
        hb = ax.hexbin(df['magnitude'], df['depth'], gridsize=30, cmap="YlGnBu")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Frequency")
        ax.set_title("Magnitude vs Depth Hexbin Plot")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Depth (km)")
        st.pyplot(fig)
        plt.close(fig)


    # Depth Distribution
    with cols[2]:
        st.markdown('#### Depth Distribution')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.violinplot(y=df['depth'], ax=ax, color='lightgreen')
        ax.set_title("Depth Distribution (Violin Plot)", color='white')
        ax.set_ylabel("Depth (km)", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Earthquake Locations
    with cols[0]:
        st.markdown('#### Earthquake Locations')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.kdeplot(x=df['longitude'], y=df['latitude'], cmap="Reds", shade=True, thresh=0.05, ax=ax)
        ax.set_title("Earthquake Location Density Plot", color='white')
        ax.set_xlabel("Longitude", color='white')
        ax.set_ylabel("Latitude", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Earthquake Magnitude by Continent
    with cols[1]:
        st.markdown('#### Magnitude by Continent')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.swarmplot(x='continent', y='magnitude', data=df, ax=ax, palette="viridis")
        ax.set_title("Magnitude Distribution by Continent", color='white')
        ax.set_xlabel("Continent", color='white')
        ax.set_ylabel("Magnitude", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Magnitude by Tsunami Presence
    with cols[2]:
        st.markdown('#### Magnitude by Tsunami Presence')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.histplot(data=df, x='magnitude', hue='tsunami', multiple="stack", bins=30, ax=ax, palette="magma")
        ax.set_title("Magnitude Distribution by Tsunami Presence", color='white')
        ax.set_xlabel("Magnitude", color='white')
        ax.set_ylabel("Frequency", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Hour of Earthquake Occurrence
    with cols[0]:
        st.markdown('#### Hour of Earthquake Occurrence')
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['hour'] = df['time'].dt.hour
        hours = df['hour'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(polar=True))  # Consistent size
        theta = np.linspace(0.0, 2 * np.pi, len(hours), endpoint=False)
        values = hours.values
        bars = ax.bar(theta, values, width=0.3, bottom=0.0, color='teal', alpha=0.7)

        ax.set_xticks(theta)
        ax.set_xticklabels(hours.index, color='white')
        ax.set_title("Earthquake Occurrences by Hour (Polar Plot)", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Magnitude and Depth by Type
    with cols[1]:
        st.markdown('#### Magnitude and Depth by Type')
        pairplot = sns.pairplot(df, vars=['magnitude', 'depth'], hue='type', palette="husl")
        pairplot.fig.suptitle("Magnitude and Depth by Earthquake Type", y=1.02, color='white')
        st.pyplot(pairplot.fig)
        plt.close(pairplot.fig)

    # Magnitude vs Distance from Epicenter
    with cols[2]:
        st.markdown('#### Magnitude vs Distance from Epicenter')
        fig, ax = plt.subplots(figsize=(8, 5))  # Consistent size
        sns.regplot(x='distanceKM', y='magnitude', data=df, scatter_kws={'alpha': 0.5}, ax=ax, color='orange')
        ax.set_title("Magnitude vs. Distance from Epicenter", color='white')
        ax.set_xlabel("Distance from Epicenter (KM)", color='white')
        ax.set_ylabel("Magnitude", color='white')
        st.pyplot(fig)
        plt.close(fig)

    # Insights Section
    st.header("💡 Insights")

    # Re-displaying the graphs without columns
    st.markdown('#### Magnitude Distribution')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(df['magnitude'], shade=True, ax=ax, color='skyblue')
    ax.set_title("Magnitude Distribution (KDE)", color='white')
    ax.set_xlabel("Magnitude", color='white')
    ax.set_ylabel("Density", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: A KDE (Kernel Density Estimate) plot, showing the smooth probability density of earthquake magnitudes. 
             * Interpretation: Shows which earthquake magnitudes are most common. Peaks indicate typical magnitudes, while lower regions reveal rarer magnitudes. Helps to identify the usual intensity range of earthquakes.
             """)

    st.markdown('#### Magnitude vs Depth')
    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(df['magnitude'], df['depth'], gridsize=30, cmap="YlGnBu")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Frequency")
    ax.set_title("Magnitude vs Depth Hexbin Plot", color='white')
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Depth (km)")
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Hexbin plot showing the distribution of earthquake depth and magnitude.
            * Interpretation: Displays how magnitude changes with depth. Darker hexagons indicate more earthquakes with similar magnitudes and depths, helping to pinpoint common depth levels for high-intensity earthquakes.
            """)

    st.markdown('#### Depth Distribution')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(y=df['depth'], ax=ax, color='lightgreen')
    ax.set_title("Depth Distribution (Violin Plot)", color='white')
    ax.set_ylabel("Depth (km)", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Violin plot showing the distribution of earthquake depths.
             * Interpretation: Highlights the spread of earthquake depths. Wider areas in the plot show the depths where earthquakes are more frequent, helping us understand the most common depth range.
             """)

    st.markdown('#### Earthquake Locations')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(x=df['longitude'], y=df['latitude'], cmap="Reds", shade=True, thresh=0.05, ax=ax)
    ax.set_title("Earthquake Location Density Plot", color='white')
    ax.set_xlabel("Longitude", color='white')
    ax.set_ylabel("Latitude", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Density plot showing areas with the highest concentration of earthquakes on a geographical scale.
             * Interpretation: Highlights the regions with the highest concentration of earthquakes. Brighter areas indicate frequent earthquake activity, identifying regions that may be more seismically active.
             """)

    st.markdown('#### Magnitude by Continent')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.swarmplot(x='continent', y='magnitude', data=df, ax=ax, palette="viridis")
    ax.set_title("Magnitude Distribution by Continent", color='white')
    ax.set_xlabel("Continent", color='white')
    ax.set_ylabel("Magnitude", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Swarm plot showing individual earthquake magnitudes across continents.
             * Interpretation: Shows the magnitude range of earthquakes by continent. Densely packed points indicate frequent magnitudes, helping to identify if certain continents experience stronger earthquakes more often.
             """)

    st.markdown('#### Magnitude by Tsunami Presence')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='magnitude', hue='tsunami', multiple="stack", bins=30, ax=ax, palette="magma")
    ax.set_title("Magnitude Distribution by Tsunami Presence", color='white')
    ax.set_xlabel("Magnitude", color='white')
    ax.set_ylabel("Frequency", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Polar plot showing earthquake occurrences based on the time of day.
             * Interpretation: Compares earthquake magnitudes with tsunami occurrence. Higher bars show the magnitude range where tsunamis are more common, indicating a relationship between higher magnitudes and tsunami generation.
            """)
    
    st.markdown('#### Hour of Earthquake Occurrence')
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['hour'] = df['time'].dt.hour
    hours = df['hour'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(polar=True))
    theta = np.linspace(0.0, 2 * np.pi, len(hours), endpoint=False)
    values = hours.values
    bars = ax.bar(theta, values, width=0.3, bottom=0.0, color='teal', alpha=0.7)

    ax.set_xticks(theta)
    ax.set_xticklabels(hours.index, color='white')
    ax.set_title("Earthquake Occurrences by Hour (Polar Plot)", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""* Graph: Polar plot showing earthquake occurrences based on the time of day.
            * Interpretation: Displays earthquake occurrences by hour in a circular format. Peaks at certain hours indicate the time when earthquakes are more frequent, showing if there are preferred times for earthquake activity.
             """)
    
    st.markdown('#### Magnitude and Depth by Type')
    pairplot = sns.pairplot(df, vars=['magnitude', 'depth'], hue='type', palette="husl")
    pairplot.fig.suptitle("Magnitude and Depth by Earthquake Type", y=1.02, color='white')
    st.pyplot(pairplot.fig)
    plt.close(pairplot.fig)
    st.write("""
             * Graph: Pair plot comparing magnitude, depth, and earthquake type.
             * Interpretation: Compares depth and magnitude across earthquake types. Clusters of points indicate typical depth-magnitude ranges for each type, showing if specific types tend to occur at certain depths or magnitudes.
             """)

    st.markdown('#### Magnitude vs Distance from Epicenter')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(x='distanceKM', y='magnitude', data=df, scatter_kws={'alpha': 0.5}, ax=ax, color='orange')
    ax.set_title("Magnitude vs. Distance from Epicenter", color='white')
    ax.set_xlabel("Distance from Epicenter (KM)", color='white')
    ax.set_ylabel("Magnitude", color='white')
    st.pyplot(fig)
    plt.close(fig)
    st.write("""
             * Graph: Scatter plot showing the relationship between magnitude and distance from the epicenter, with a regression line.
             * Interpretation: Shows if earthquake magnitudes change with distance from the epicenter. The regression line reveals any trend—higher or lower magnitudes—relative to their distance from the origin point, indicating whether location affects intensity.
             """)


# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")


# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Add your machine learning code here
#Actual vs. Predicted Magnitude Plot
    import streamlit as st
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Assuming df is defined and available

    regression_features = ["latitude", "longitude", "depth", "felt", "mmi", "gap", "rms"]
    df_cleaned = df[regression_features + ["magnitude"]].dropna()

    X = df_cleaned[regression_features]  # Features
    y = df_cleaned["magnitude"]  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)

    y_pred = linear_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.markdown('## Actual vs. Predicted Magnitude Plot')
    st.subheader('#### Model Evaluation')
    st.write(f"1. **Mean Squared Error (MSE): {mse:.4f}**")
    st.write("   - `MSE` represents the average squared difference between actual and predicted magnitudes.")
    st.write("   - A `lower MSE` indicates better performance.\n")

    st.write(f"2. **R² Score: {r2:.4f}**")
    st.write("   - `R²` measures the proportion of the variance in the target variable explained by the model.")
    st.write("   - R² ranges from `0 to 1`, where 1 indicates perfect predictions.\n")

    st.write(f"3. **Mean Absolute Error (MAE): {mae:.4f}**")
    st.write("   - `MAE` represents the average absolute difference between actual and predicted magnitudes.")
    st.write("   - Like MSE, a `lower MAE` indicates better performance.\n")

    # Visualizing the Actual vs Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Actual vs Predicted Magnitudes')
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Random Forest Classifier to Predict Tsunami
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st

    classification_features = ["latitude", "longitude", "depth", "felt", "mmi", "gap", "rms"]
    df_classification = df[classification_features + ["tsunami"]].dropna()

    X_class = df_classification[classification_features]
    y_class = df_classification["tsunami"]
    X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_class_train, y_class_train)

    y_class_pred = classifier.predict(X_class_test)

    accuracy = accuracy_score(y_class_test, y_class_pred)
    class_report = classification_report(y_class_test, y_class_pred)

    st.markdown('## Random Forest Classifier to Predict Tsunami')
    st.markdown('### Tsunami Prediction Model Evaluation')
    st.write(f"1. **Accuracy: {accuracy:.4f}**")
    st.write("   - `Accuracy` indicates the proportion of correct predictions out of the total predictions.\n")
    st.write("**2. Classification Report:**")
    st.text(class_report)

    # Confusion Matrix Visualization
    conf_matrix = confusion_matrix(y_class_test, y_class_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix for Tsunami Prediction')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display the confusion matrix in Streamlit
    st.pyplot(plt)

    
    # K-Nearest Neighbors Classifier for Alert Prediction
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # Assuming `data` is defined and available

    data_alert = df.dropna(subset=["alert"])
    X_alert = data_alert[["latitude", "longitude", "depth", "mmi", "gap"]]
    y_alert = data_alert["alert"]

    le = LabelEncoder()
    y_alert = le.fit_transform(y_alert)

    X_alert_train, X_alert_test, y_alert_train, y_alert_test = train_test_split(X_alert, y_alert, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_alert_train, y_alert_train)

    y_alert_pred = knn.predict(X_alert_test)

    alert_accuracy = accuracy_score(y_alert_test, y_alert_pred)
    alert_report = classification_report(y_alert_test, y_alert_pred, target_names=le.classes_)

    st.markdown('## K-Nearest Neighbors Classifier for Alert Prediction')
    st.markdown('#### Alert Prediction Model Evaluation')
    st.write(f"**1. Accuracy: {alert_accuracy:.4f}**")
    st.write("   - `Accuracy` indicates the proportion of correct predictions out of the total predictions.\n")
    st.write("**2. Classification Report:**")
    st.text(alert_report)

    # Confusion Matrix Visualization
    conf_matrix_alert = confusion_matrix(y_alert_test, y_alert_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_alert, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix for Alert Prediction')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display the confusion matrix in Streamlit
    st.pyplot(plt)

    # Means Clustering to Find Patterns
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import streamlit as st

    # Assuming `data` is defined and available

    clustering_features = ["latitude", "longitude", "depth", "magnitude"]
    data_clustering = df[clustering_features].dropna()

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data_clustering)
    data_clustering_with_clusters = data_clustering.copy()
    data_clustering_with_clusters["Cluster"] = clusters

    st.markdown('## Means Clustering to Find Patterns')
    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(data_clustering_with_clusters["latitude"], data_clustering_with_clusters["longitude"], 
                c=data_clustering_with_clusters["Cluster"], cmap="viridis", alpha=0.6)
    plt.title("Earthquake Clusters based on Location")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.colorbar(label="Cluster")

    # Display the plot in Streamlit
    st.pyplot(plt)


    # Principal Component Analysis (PCA) for Dimensionality Reduction
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import streamlit as st

    # Preprocessing for PCA
    pca_features = ["latitude", "longitude", "depth", "felt", "mmi", "gap", "rms"]
    df_pca = df[pca_features].dropna()

    # Applying PCA to reduce to 2 principal components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_pca)

    # Evaluate PCA
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    # Display the results in Streamlit
    st.markdown('## Principal Component Analysis (PCA) for Dimensionality Reduction')
    st.markdown('#### PCA Evaluation')
    st.write("Explained Variance Ratio for each `Principal Component`:")
    for i, var in enumerate(explained_variance, 1):
        st.write(f"  **Principal Component {i}: {var:.4f}**")

    st.write(f"\n**Cumulative Explained Variance for the first 2 components: {cumulative_variance[1]:.4f}**")
    st.write("   - Measures how much of the `total variance` is retained by the first two components.\n")

    # Visualizing PCA results
    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.6)
    plt.title("PCA of Earthquake Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Display the plot in Streamlit
    st.pyplot(plt)
# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Add your prediction code here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

# Footer
st.markdown("""
--- 
**Project by Group 3**
""")
