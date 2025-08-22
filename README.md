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

- pand
