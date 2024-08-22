# Quora Question Similarity
By *Alonge Daniel* and Team NIK Avengers on Innomatics Internship 

- GitHub: [Alonge 9500](https://github.com/Alonge9500)
- LinkedIn: [Alonge Daniel](https://www.linkedin.com/in/alonge-daniel-27b4b4139/)
- Email: [Alonge Daniel](mailto:alongedaniel19@gmail.com)

## Note
* This work is a group work and my area of focus are
* Modelling
* Deployment(Flask)
I' will appreciate any comment and correction on this work 

This repository contains code for a Quora question similarity project. The goal of this project is to determine the similarity between pairs of questions asked on Quora. By identifying similar questions, it can help improve user experience by providing relevant answers and reducing duplicate content.

## Problem Statement

Quora is a platform where users can ask questions and get answers from the community. However, many questions asked on Quora are similar or duplicate in nature. The aim of this project is to build a machine learning model that can accurately identify question pairs that are semantically similar. This can be useful in various applications such as question recommendation, duplicate question detection, and information retrieval.

## Data Description

The project uses a dataset containing pairs of questions from Quora. The dataset includes the following columns:

- question1: First question in the pair
- question2: Second question in the pair
- is_duplicate: Flag indicating whether the question pair is duplicate (1) or not (0)

## Getting Started

To run the code in this repository, follow these steps:

1. Clone the repository: `git clone https://github.com/Alonge9500/quora_deployment`
2. Install the required libraries: `pip install -r requirements.txt`
3. Download the dataset file (quora_questions.csv) and place it in the project directory.
4. Run the Jupyter notebook: `jupyter notebook`
5. Open the notebook `EDA(1).ipynb` and execute the code cells sequentially.

## Exploratory Data Analysis

The notebook includes exploratory data analysis of the question similarity dataset. It analyzes the distribution of question pairs, explores the distribution of duplicate vs. non-duplicate questions, and visualizes the most common words in question pairs.

## Text Preprocessing

The notebook performs text preprocessing tasks on the question pairs. It applies techniques such as tokenization, removing stopwords, and stemming/lemmatization to transform the raw text data into a suitable format for modeling.

## Modeling with MLOps

The notebook builds a machine learning model to predict question similarity. It uses various algorithms such as TF-IDF (Term Frequency-Inverse Document Frequency), Word2Vec, or BERT (Bidirectional Encoder Representations from Transformers). The model is trained on the labeled dataset and evaluated using appropriate evaluation metrics. MLOps practices, such as versioning the model, tracking experiments, and automating the training pipeline, are implemented using libraries like MLflow and TensorFlow Extended (TFX).

## Deployment with Flask

The repository also includes code for deploying the trained model using Flask. The Flask web application provides an interface where users can enter two questions and get a similarity score as the output. The model is loaded and used to make predictions in real-time.

## Conclusion

The Quora question similarity project aims to identify semantically similar question pairs using machine learning techniques. By performing exploratory analysis, text preprocessing, and modeling with MLOps, it provides a comprehensive solution for question similarity detection. The deployment with Flask enables real-time inference and integration with other applications.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to explore the code and adapt it to your requirements.
