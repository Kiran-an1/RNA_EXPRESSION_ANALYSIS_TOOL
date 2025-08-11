import streamlit as st
import mysql.connector
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
# Title of the App
st.title("Tissue-Specific RNA Expression Analysis Tool")
st.write("Select genes and tissues to analyze RNA expression levels.")

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

# Visualization: Heatmap with Plotly
def plot_interactive_heatmap(df):
    # Apply log transformation if needed (optional)
    # df = df.apply(lambda x: np.log(x + 1))  # Uncomment if you want to log-transform

    # Create the heatmap with Plotly
    fig = px.imshow(
        df,
        labels={'x': 'Tissues', 'y': 'Genes', 'color': 'Expression'},
        color_continuous_scale='Pinkyl',  # Change color scheme to pink shades
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

# Visualization: Violin Plot
# Visualization: Violin Plot
def plot_violinplot(df):
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df, palette="muted")  # Ensure this is the correct function
    plt.title("Violin Plot of Gene Expression Across Tissues", fontsize=16)
    plt.ylabel("Expression Level", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    st.pyplot(plt)


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

    st.write("Violin Plot of Gene Expression Across Tissues:")
    plot_violinplot(normalized_data)

    # Option to download results
    csv_data = normalized_data.reset_index().to_csv(index=False)
    st.download_button("Download Results as CSV", csv_data, "results.csv", "text/csv")
