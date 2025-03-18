# PlotDex

## About

PlotDex is an intelligent tool for automatically identifying and classifying visualizations and plots in scientific documents. Using a ResNet-based deep learning model trained on a diverse dataset of scientific visualizations, PlotDex can recognize various chart types including line plots, bar charts, scatter plots, heatmaps, and more.

This tool helps researchers, students, and data scientists quickly analyze the visual content of papers, enabling efficient literature review, meta-analysis, and data extraction from published research. PlotDex is especially valuable for visually impaired users, providing an accessible way to understand the visual elements in scientific publications through detailed descriptions and classifications of charts and graphs.

### Key Features

- Fast and accurate recognition of common scientific visualization types
- Multiple trained models to choose from, each with different training characteristics
- Simple web interface built with Gradio for easy interaction
- Accessibility features to help visually impaired users understand scientific visualizations
- Support for local processing on CPU, CUDA, or MPS devices
- Open-source implementation for research transparency and community improvement

### Use Cases

- Enhancing accessibility of scientific literature for visually impaired researchers and students
- Providing text-based descriptions of visual content in academic papers
- Automating literature reviews by classifying figures in bulk
- Extracting structured data from publications
- Teaching visualization best practices by analyzing example charts
- Building meta-analysis studies of visualization usage across research domains

### Accessibility Vision

PlotDex aims to make scientific visualizations more accessible by not only identifying plot types but also extracting key information that can be conveyed through screen readers and other assistive technologies. Future development will focus on generating rich, descriptive summaries of visualizations that communicate the essential information typically gleaned visually, making scientific research more inclusive and accessible to all.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/JOTELLECHEA/PlotDex.git

# Navigate to the repository directory
cd PlotDex

# Install required dependencies
# file will be uploaded soon
pip install -r requirements.txt
```

# Running the Application

## Start the Gradio web interface

```bash
# Run the Gradio web app
python app.py
```

Once the application is running, open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860) to access the PlotDex interface.
