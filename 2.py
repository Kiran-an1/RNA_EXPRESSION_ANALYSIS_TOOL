import streamlit as st
import mysql.connector
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Title of the App
st.title("Tissue-Specific RNA Expression Analysis Tool")
st.write("Select genes and tissues to analyze RNA expression levels.")

# Set scientific theme
def set_scientific_theme():
    st.markdown(
        
        <style>
        .stApp {
            background-color: #f0f8ff;  /* Light blue background */
            color: #2c3e50;  /* Dark gray text for readability */
        }
        .stButton, .stSelectbox, .stMultiselect, .stTextInput {
            background-color: #58a4b0;  /* Soft teal button color */
            color: white;
        }
        .stButton:hover, .stSelectbox:hover, .stMultiselect:hover, .stTextInput:hover {
            background-color: #2c3e50;  /* Darker teal on hover */
        }
        .stMarkdown, .stText {
            color: #2c3e50;  /* Dark text for markdown and other text */
        }
        .stDataFrame {
            background-color: #ffffff;  /* White background for dataframes */
        }
        .stTable {
            background-color: #ffffff;  /* White background for tables */
        }
        .stPlotlyChart {
            background-color: #ffffff;  /* White background for charts */
        }
        </style>
        
        unsafe_allow_html=True,
    )

# Apply the scientific theme
set_scientific_theme()

# Function to connect to the MySQL database
def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MySQL username
        password="12345",  # Replace with your MySQL password
        database="gene_expression_data"  # Replace with your database name
    )

# Function to fetch genes and tissues from the database
def get_genes_and_tissues():
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Name FROM gene_expression_data")
    genes = [row[0] for row in cursor.fetchall()]
    cursor.execute("SHOW COLUMNS FROM gene_expression_data")
    tissues = [row[0] for row in cursor.fetchall() if row[0] not in ["Name", "Description"]]
    conn.close()
    return genes, tissues

# Function to normalize data
def normalize_data(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

# Function to fetch and normalize data based on user selections
def fetch_and_normalize(selected_genes, selected_tissues):
    conn = connect_to_database()
    cursor = conn.cursor()

    # Build query dynamically
    query = f"SELECT Name, {', '.join(selected_tissues)} FROM gene_expression_data WHERE Name IN ({', '.join(['%s'] * len(selected_genes))})"
    cursor.execute(query, selected_genes)

    # Convert to DataFrame
    data = pd.DataFrame(cursor.fetchall(), columns=["Name"] + selected_tissues)
    conn.close()

    # Normalize
    normalized_data = normalize_data(data.set_index("Name"))
    return normalized_data

# Function to identify tissue-specific genes
def identify_tissue_specific_genes(df):
    specific_genes = df.idxmax(axis=1)
    return specific_genes

# Function to perform PCA and visualize the results
def plot_pca(df):
    # Standardize the data before PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Apply PCA
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

# Visualization: Heatmap with Plotly
def plot_interactive_heatmap(df):
    # Create the heatmap with Plotly
    fig = px.imshow(
        df,
        labels={'x': 'Tissues', 'y': 'Genes', 'color': 'Expression'},
        color_continuous_scale='Blues',  # Blue color scheme for expression levels
        title="Interactive Heatmap of Gene Expression"
    )

    # Update x and y axis labels and font size
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))  # Rotate x-axis labels and reduce font size
    fig.update_yaxes(tickfont=dict(size=8))  # Reduce font size for y-axis labels

    # Adjust layout for better presentation
    fig.update_layout(
        width=1200,  # Set width of the heatmap
        height=900,  # Set height of the heatmap
        title_font=dict(size=16),  # Set font size for title
        coloraxis_colorbar=dict(title="Expression", tickfont=dict(size=8))  # Color bar font size
    )

    # Add hover information for better interactivity
    fig.update_traces(
        hovertemplate="<b>%{y}</b> in <b>%{x}</b><br>Expression: %{z}<extra></extra>",
        hoverlabel=dict(font_size=10)
    )

    # Display the interactive heatmap
    st.plotly_chart(fig)

# Main Application
genes, tissues = get_genes_and_tissues()

selected_genes = st.multiselect("Select Genes", genes)
selected_tissues = st.multiselect("Select Tissues", tissues)

if selected_genes and selected_tissues:
    # Fetch and normalize data
    normalized_data = fetch_and_normalize(selected_genes, selected_tissues)

    # Display normalized data
    st.write("Normalized Data:")
    st.dataframe(normalized_data)

    # Identify tissue-specific genes
    tissue_specific_genes = identify_tissue_specific_genes(normalized_data)
    st.write("Tissue-Specific Genes:")
    st.write(tissue_specific_genes)

    # Generate visualizations
    st.write("Interactive Heatmap of Gene Expression:")
    plot_interactive_heatmap(normalized_data)

    st.write("PCA of Gene Expression Data:")
    plot_pca(normalized_data)

    # Option to download results
    csv_data = normalized_data.reset_index().to_csv(index=False)
    st.download_button("Download Results as CSV", csv_data, "results.csv", "text/csv")
