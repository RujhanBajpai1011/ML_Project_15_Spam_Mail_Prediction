# 📧 Spam Mail Prediction

This project implements a Spam Mail Prediction system using Logistic Regression. It classifies incoming emails as either "ham" (legitimate) or "spam" (unsolicited/junk) based on their textual content. The project involves text preprocessing using TF-IDF vectorization, model training, and evaluation.

## 📊 Dataset

The dataset used is `mail_data.csv`, which contains two main columns:

- **Category**: Indicates whether the mail is 'ham' or 'spam'. This is the target variable.
- **Message**: The actual text content of the email.

The dataset contains 5572 entries and 2 columns, and importantly, it has no missing values after initial handling.

## ✨ Features

**Data Loading and Initial Inspection**: Loads the `mail_data.csv` into a pandas DataFrame and displays initial rows and overall shape.

**Missing Value Handling**: Replaces any potential null values in the DataFrame with an empty string to ensure clean text data for processing.

**Categorical to Numerical Conversion**: Converts the 'Category' column from text labels ('spam', 'ham') to numerical representations (0 for 'spam', 1 for 'ham').

**Data Separation**: Divides the dataset into features (X, the 'Message' content) and the target variable (Y, the numerical 'Category').

**Data Splitting**: Splits the data into training and testing sets (80% training, 20% testing) to evaluate the model's performance on unseen data.

**Text to Feature Vectors (TF-IDF)**: Uses TfidfVectorizer to transform the raw text messages into numerical feature vectors. This process:
- Converts text into a matrix of TF-IDF features.
- Considers words that appear in at least min_df = 1 document.
- Removes common English stop_words (e.g., "the", "a", "is").
- Converts all text to lowercase.

**Logistic Regression Model Training**: Trains a Logistic Regression model on the TF-IDF transformed training data. Logistic Regression is a suitable choice for binary classification problems like spam detection.

**Model Evaluation**: Calculates and prints the accuracy score of the trained model on both the training and test datasets, indicating how well the model classifies emails.

**Predictive System**: Includes a practical example demonstrating how to use the trained model to predict whether a new, unseen email message is "Ham mail" or "Spam mail".

## 🛠️ Technologies Used

- **Python**
- **pandas**: For efficient data loading and manipulation.
- **numpy**: For numerical operations.
- **scikit-learn**: For core machine learning functionalities, specifically:
  - **train_test_split**: For dividing the dataset.
  - **TfidfVectorizer**: For text feature extraction.
  - **LogisticRegression**: The core classification algorithm used.
  - **accuracy_score**: For evaluating model performance.

## 📦 Requirements

To run this project, you will need the following Python libraries:

- pandas
- numpy
- scikit-learn

## 🚀 Getting Started

Follow these steps to get a copy of this project up and running on your local machine:

### Installation

1. Clone the repository (if applicable):
```bash
git clone <repository_url>
cd <repository_name>
```

2. Install the required Python packages:
```bash
pip install pandas numpy scikit-learn
```

### Usage

1. **Place the dataset**: Ensure the `mail_data.csv` file is located in the same directory as the Jupyter notebook (`Spam_Mail_Prediction.ipynb`).

2. **Run the Jupyter Notebook**: Open and execute all the cells in the `Spam_Mail_Prediction.ipynb` notebook using a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, Google Colab).

The notebook will:
- Load and preprocess the email data.
- Convert text messages into numerical features.
- Train the Logistic Regression model.
- Output the model's accuracy on training and test data.
- Demonstrate a prediction for a sample email message.

## 📈 Results

The notebook outputs the accuracy scores for the Logistic Regression model on both the training and test datasets. These scores indicate the percentage of correctly classified emails.

- **Accuracy on Training Data**: Approximately 0.9677 (96.77%)
- **Accuracy on Test Data**: Approximately 0.9668 (96.68%)

These high accuracy scores suggest that the Logistic Regression model performs exceptionally well in distinguishing between spam and ham emails, demonstrating strong generalization capabilities to new, unseen messages.

## 🧑‍💻 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit pull requests.

## 📄 License

This project is open-source.
