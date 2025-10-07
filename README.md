# AG News Dataset Preprocessing & EDA

This project focuses on preparing and analyzing the AG News dataset to build a clean, 
reliable dataset for machine learning models that classify news articles into categories.

---

## 🎯 Project Objectives
- Perform data cleaning and preprocessing  
- Apply 6 preprocessing techniques  
- Conduct Exploratory Data Analysis (EDA) with visualizations  
- Select important features using feature selection / dimensionality reduction  
- Prepare the final processed dataset for modeling  

---

## 👥 Group Members & Responsibilities

| Member | IT Number  | Preprocessing Technique | Description |
| ------- | ---------- | ----------------------- | ------------ |
| 1 | IT24102989 | Handling Missing Values | Checked for null values in text and labels, filled missing text with empty strings, and dropped rows with missing labels. |
| 2 | IT24102743 | Encoding Categorical Variables | Converted text labels (World, Sports, Business, Sci/Tech) into numeric form using label encoding. |
| 3 | **IT24102783 (Mevinu Methdam)** | **Outlier Handling** | Used IQR on word count to remove extremely short or long articles. |
| 4 | IT24102329 | Normalization (Min-Max Scaling) | Applied Min-Max scaling to numeric features. |
| 5 | IT24102879 | Dimensionality Reduction | Applied Truncated SVD for latent semantic analysis. |
| 6 | IT24102665 | Feature Selection | Used Chi-Square (SelectKBest) to retain the most informative TF-IDF features. |

---

## 💻 My Contribution (By: **IT24102783 – Mevinu Methdam**)

### 🧩 Technique: Outlier Handling

I was responsible for detecting and removing **outliers** in the AG News dataset, ensuring the text data was clean and consistent before feature extraction.

---

#### 🧠 Objective:
Outliers occur when some news articles are **too short** (like titles only) or **too long** (such as entire paragraphs).  
These extreme cases can mislead machine learning models by introducing irregular text lengths.  
To fix this, I used the **Interquartile Range (IQR)** method based on **word counts**.

---

#### ⚙️ Steps I Followed:
1. Created a new column `word_count` using the number of words in each article.  
2. Calculated **Q1**, **Q3**, and **IQR (Q3 - Q1)**.  
3. Defined lower and upper limits to detect outliers.  
4. Removed records outside these limits.  
5. Saved the cleaned dataset for the next stages (scaling, normalization, etc.).

---

#### 🧮 Example Code:
```python
# Create a word count column
df['word_count'] = df['Description'].apply(lambda x: len(str(x).split()))

# Calculate IQR
Q1 = df['word_count'].quantile(0.25)
Q3 = df['word_count'].quantile(0.75)
IQR = Q3 - Q1

# Define limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Remove outliers
df_outlier_removed = df[(df['word_count'] >= lower_limit) & (df['word_count'] <= upper_limit)]

print("Before Removal:", df.shape)
print("After Removal:", df_outlier_removed.shape)
