Tissue-Specific RNA Expression Analysis Tool
Description
This project is a Streamlit-based web application designed to analyze RNA expression levels across various tissues.
It connects to a MySQL database containing gene expression data, lets users select specific genes and tissues, and provides interactive visualizations such as:

Interactive Heatmap

PCA (Principal Component Analysis) Plot

The tool also identifies tissue-specific genes — i.e., the tissue where each gene is most highly expressed.

🚀 Features
Connects directly to a MySQL database with gene expression data.

Allows selection of multiple genes and tissues.

Normalizes expression data for easier comparison.

Identifies tissue-specific genes.

Generates:

📊 Interactive Heatmap (via Plotly)

📉 PCA Visualization (via scikit-learn + Plotly)

Option to download results as CSV.

Styled via custom CSS for better UI.

🛠 Tech Stack
Python 3.8+

Streamlit – Web app framework

MySQL – Database backend

Pandas – Data manipulation

Plotly – Interactive visualizations

scikit-learn – PCA & data scaling


👩‍💻 Author
Kiran Naseer
