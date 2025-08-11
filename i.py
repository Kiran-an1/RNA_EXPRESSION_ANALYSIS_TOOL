import streamlit as st
import mysql.connector
import pandas as pd # handling and analyzing tabular data
import plotly.express as px #interactive visulization for scater plot and heatmap
from sklearn.decomposition import PCA #perform PCA for easier visualization and reduce dimenionality
from sklearn.preprocessing import StandardScaler #standarize the data before PCA

# Function to load external CSS
def load_css():
    # Load custom CSS from an external file
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Default styles applied.")

# Inject external CSS into the app
load_css()

# Title and App Description
st.title("Tissue-Specific RNA Expression Analysis Tool")
st.write("Select genes and tissues to analyze RNA expression levels and generate visualizations.")

# Function to connect to the MySQL database
def connect_to_database():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",  
            password="12345",  
            database="gene_expression_data"  
        )
    except mysql.connector.Error as err:
        st.error(f"Error connecting to the database: {err}")
        return None

# Function to fetch unique genes and tissue columns from the database
def get_genes_and_tissues():
    conn = connect_to_database()
    if not conn:
        return [], []

    cursor = conn.cursor()
    try:
        # Fetch distinct gene names
        cursor.execute("SELECT DISTINCT Name FROM gene_expression_data")
        genes = [row[0] for row in cursor.fetchall()]

        # Fetch tissue columns
        cursor.execute("SHOW COLUMNS FROM gene_expression_data")
        tissues = [row[0] for row in cursor.fetchall() if row[0] not in ["Name", "Description"]]
    except mysql.connector.Error as err:
        st.error(f"Error fetching data: {err}")
        genes, tissues = [], []
    finally:
        conn.close()

    return genes, tissues

# Function to normalize data
def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

# Function to fetch and normalize selected data
def fetch_and_normalize(selected_genes, selected_tissues):
    conn = connect_to_database()
    if not conn:
        return pd.DataFrame()

    cursor = conn.cursor()
    try:
        # Dynamically build the query
        query = f"SELECT Name, {', '.join(selected_tissues)} FROM gene_expression_data WHERE Name IN ({', '.join(['%s'] * len(selected_genes))})"
        cursor.execute(query, selected_genes)

        # Convert data to a pandas DataFrame
        data = pd.DataFrame(cursor.fetchall(), columns=["Name"] + selected_tissues)
        conn.close()

        # Normalize the data
        normalized_data = normalize_data(data.set_index("Name"))
        return normalized_data
    except mysql.connector.Error as err:
        st.error(f"Error fetching or processing data: {err}")
        conn.close()
        return pd.DataFrame()

# Function to identify tissue-specific genes Finds the tissue where each gene is most highly expressed.
def identify_tissue_specific_genes(df):
    return df.idxmax(axis=1)

# Function to perform PCA and visualize results
def plot_pca(df):
    # Standardize the data  and then scale it to transform before PCA 
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Apply PCA, captures in which direction, data captures the most variation
    pca = PCA(n_components=2)  # We will use the first two components
    pca_result = pca.fit_transform(scaled_data)

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    pca_df["Gene"] = df.index

    # Plot the PCA result
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Gene",  # Color by gene
        title="PCA of Gene Expression Data",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        hover_data={"Gene": True},  # Show gene name on hover
        text=None  # Remove gene name from the chart
    )

    # Adjust marker size and appearance
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')))

    # Add hover template for better interactivity
    fig.update_traces(hovertemplate="<b>Gene:</b> %{customdata[0]}<br>PC1: %{x}<br>PC2: %{y}<extra></extra>")

    # Display the PCA plot
    st.plotly_chart(fig)

    # Explain PCA
    st.write("**PCA Explanation**:")
    st.write("PCA reduces the dimensionality of the gene expression data by finding the principal components (PC1, PC2) that capture the most variance.")
    st.write("PC1 (Principal Component 1) is the direction in which the data varies the most, and PC2 (Principal Component 2) captures the second most variance.")
    st.write("This allows us to visualize high-dimensional data in 2D while retaining important patterns in the data.")


# Function to create and display an interactive heatmap
def plot_interactive_heatmap(df):
    fig = px.imshow(
        df,
        labels={'x': 'Tissues', 'y': 'Genes', 'color': 'Expression'},
        color_continuous_scale='Blues',
        title="Interactive Heatmap of Gene Expression"
    )

    # Update layout for better readability
    fig.update_layout(
        width=1000,
        height=800,
        coloraxis_colorbar=dict(title="Expression Level")
    )

    st.plotly_chart(fig)

# Main Application Workflow
genes, tissues = get_genes_and_tissues()

if genes and tissues:
    # User selections
    selected_genes = st.multiselect("Select Genes", genes)
    selected_tissues = st.multiselect("Select Tissues", tissues)

    if selected_genes and selected_tissues:
        # Fetch and normalize data
        normalized_data = fetch_and_normalize(selected_genes, selected_tissues)

        if not normalized_data.empty:
            # Display normalized data
            st.write("### Normalized Data")
            st.dataframe(normalized_data)

            # Identify tissue-specific genes
            tissue_specific_genes = identify_tissue_specific_genes(normalized_data)
            st.write("### Tissue-Specific Genes")
            st.write(tissue_specific_genes)

            # Generate visualizations
            st.write("### Interactive Heatmap")
            plot_interactive_heatmap(normalized_data)

            st.write("### PCA Visualization")
            plot_pca(normalized_data)

            # Option to download normalized data as CSV
            csv_data = normalized_data.reset_index().to_csv(index=False)
            st.download_button("Download Results as CSV", csv_data, "normalized_gene_expression.csv", "text/csv")
else:
    st.warning("No genes or tissues found in the database. Please ensure the database is set up correctly.")
