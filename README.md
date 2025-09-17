# 🍲 Food Recommendation Based on Collaborative Filtering

**Author:** [Naufal Fajar Revanda](https://www.linkedin.com/in/naufalrevanda/) · [GitHub](https://github.com/nrevanda) · [LinkedIn](https://www.linkedin.com/in/naufalrevanda/)

[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-IBCF-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)

> Personalized food product recommendations learned from user reviews. Built with **Item‑Based Collaborative Filtering** (Cosine Similarity) on the **Amazon Fine Food Reviews** dataset, served via a **Streamlit** app.

---

## 📚 Introduction

### Context

In today’s highly competitive e‑commerce landscape, delivering a personalized shopping experience has become essential for both retaining customers and driving sales growth. One of the most powerful ways to achieve this is through **recommendation systems**, which guide users toward products they are likely to enjoy. Beyond enhancing convenience, such systems give platforms a competitive edge by **encouraging cross‑sell** and **up‑sell opportunities**.

### Problem Statement

The **Amazon Fine Food Reviews** dataset contains **500k+** product reviews. With so many choices, **users struggle to find items that truly match their tastes**. Meanwhile, platforms must recommend relevant items even when there is limited purchase history. This **information overload** demands intelligent solutions to bridge the gap between the vast volume of product data and each user’s unique preferences.

### Goals

* **Build an Item‑Based Collaborative Filtering model** that identifies product similarity from user rating patterns.
* **Deliver relevant recommendations**: Provide a Top‑N list of products users are likely to enjoy, inferred from their interactions with other items.

---

## 🗂 Dataset & Features

**Source:** [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data)

| Feature                    | Description                                                         |
| -------------------------- | ------------------------------------------------------------------- |
| **Id**                     | Row identifier                                                      |
| **ProductId**              | Unique identifier for the product                                   |
| **UserId**                 | Unique identifier for the user                                      |
| **ProfileName**            | Display name of the reviewer                                        |
| **HelpfulnessNumerator**   | Number of users who found the review helpful                        |
| **HelpfulnessDenominator** | Number of users who indicated whether the review was helpful or not |
| **Score**                  | Rating (1–5)                                                        |
| **Time**                   | Review timestamp (Unix)                                             |
| **Summary**                | Short title/summary of the review                                   |
| **Text**                   | Full review text                                                    |

---
## 🎥 Streamlit Interface

<img width="3017" height="1482" alt="image" src="https://github.com/user-attachments/assets/4a5cee61-c459-4bb3-a3db-1aeca4bdf253" />

---

## 🏗 Project Structure

```
.
├── app.py                                
├── requirements.txt
├── models/
│   └── item_similarity_matrix.joblib    
├── notebooks/
│   └── recommendation_system_ibcf.ipynb        
├── data/
│   └── Reviews.csv                       
└── README.md
```

---


## 📊 Results (Example)

* The recommender returns **Top‑N** similar items ordered by cosine similarity.
* Streamlit view includes a styled table (scores with gradient) and a bar chart.
* Example artifacts (add your visuals): `assets/demo.png`.

---

## ✅ Conclusion

We successfully built a functional **Item‑Based Collaborative Filtering** system on **Amazon Fine Food Reviews**, uncovering patterns in user ratings and recommending items similar to those a user prefers. Using **Cosine Similarity**, the model provides relevant suggestions that can support product discovery and **potentially boost cross/upsell** for e‑commerce.

---

## 🔧 Recommendations / Future Work

* **Cold Start**: Combine CF with **content‑based** signals (product metadata, review **sentiment/topic**) or popularity‑based priors.
* **Advanced Models**: Explore **Matrix Factorization** (e.g., SVD) to learn latent factors for improved accuracy.
* **Hybridization**: Enrich with NLP features from review text for **context‑aware** recommendations.
* **UX Enhancements**: Map `ProductId → Product Title/Brand` in the UI; add explanations (“customers who liked X also liked Y”).

---


## 🧾 TL;DR (GitHub Overview)

**One‑liner:** Food recommendation system using Item‑Based Collaborative Filtering on Amazon Fine Food Reviews — interactive Streamlit app for Top‑N similar products.

**Key points:**

* Learns from user ratings to compute item–item similarity (Cosine)
* Dropdown select → Top‑N recommendations (styled table + chart)
* Precomputed matrix for fast inference (`.joblib`)
* Clear limitations & future work (cold start, MF/SVD, hybrid with text)

---

## 📬 Contact

**Naufal Fajar Revanda** · [LinkedIn](https://www.linkedin.com/in/naufalrevanda/) · [GitHub](https://github.com/nrevanda)
