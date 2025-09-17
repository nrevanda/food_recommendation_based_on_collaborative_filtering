# # app.py
# import streamlit as st
# import pandas as pd
# import joblib

# # 1. Load the Saved Model
# @st.cache_resource
# def load_model():
#     try:
#         return joblib.load('models/item_similarity_matrix.joblib')
#     except FileNotFoundError:
#         st.error("Model file not found. Make sure 'item_similarity_matrix.joblib' is in the same directory.")
#         st.stop()

# item_similarity_df = load_model()

# # 2. Define the Recommendation Function
# def get_recommendations(product_id, similarity_matrix, top_n=5):
#     if product_id not in similarity_matrix.columns:
#         return pd.DataFrame()

#     similarity_scores = similarity_matrix[product_id]
#     sorted_scores = similarity_scores.sort_values(ascending=False)[1:top_n+1]

#     recommendations_df = pd.DataFrame(sorted_scores)
#     recommendations_df.columns = ['Similarity Score']
#     recommendations_df = recommendations_df.reset_index()
#     recommendations_df.rename(columns={'index': 'Recommended Product ID'}, inplace=True)
    
#     return recommendations_df

# # 3. User Interface (UI) Design
# st.title('Food Recommendation System Based on Collaborative Filtering🍲')
# st.markdown("Select a product from the list below to get recommendations for the most similar products.")

# product_list = sorted(item_similarity_df.columns.tolist())
# selected_product = st.selectbox('Select a Product:', product_list)

# if st.button('Get Recommendations'):
#     recommendations = get_recommendations(selected_product, item_similarity_df)
    
#     st.subheader('Recommendations for You:')
#     if recommendations.empty:
#         st.info("No recommendations were found for this product.")
#     else:
#         st.dataframe(recommendations)

# # 4. Add dataset information
# st.markdown("---")
# st.markdown("You can download this dataset directly from the Kaggle")
# st.markdown('Download link: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data)')

# # 5. Add Author Information at the Bottom of the Page
# st.markdown("---")
# st.markdown("Created by **Naufal Revanda**")
# st.markdown("Reach me at [GitHub](https://github.com/nrevanda) or [LinkedIn](https://www.linkedin.com/in/naufalrevanda/)")



# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= Page Config & Styles =================
st.set_page_config(page_title="Food Recommender", page_icon="🍲", layout="wide")

st.markdown("""
<style>
footer {visibility: hidden;}
.header-band {
  background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 40%, #e0f2fe 100%);
  border: 1px solid rgba(0,0,0,0.06);
  padding: 22px 20px; border-radius: 18px; margin-bottom: 12px;
}
.kpi-card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px; padding: 14px 16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.65), rgba(255,255,255,0.35));
}
.kpi-title { font-size: 0.85rem; color: #64748b; margin-bottom: 6px; }
.kpi-value { font-weight: 700; font-size: 1.3rem; color: #0f172a; }
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:#eef2ff; color:#3730a3; font-size:0.78rem; font-weight:600; margin-right:6px;
  border:1px solid rgba(55,48,163,0.15);
}
.data-note { font-size:0.9rem; color:#475569; }
.styled-subheader { margin-top: 4px; padding: 8px 12px; border-left: 4px solid #3b82f6;
  background: #f1f5f9; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ================= Load Model =================
@st.cache_resource(show_spinner=False)
def load_similarity_matrix():
    sim = joblib.load("item_similarity_matrix.joblib")
    sim.index = sim.index.astype(str)
    sim.columns = sim.columns.astype(str)
    return sim

with st.spinner("Loading similarity model…"):
    try:
        item_similarity_df = load_similarity_matrix()
    except FileNotFoundError:
        st.error("Model file not found. Make sure **item_similarity_matrix.joblib** is in the app folder.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ================= Helpers =================
@st.cache_data(show_spinner=False)
def get_product_list(sim_df: pd.DataFrame):
    return sorted(sim_df.columns.tolist())

def get_recommendations(product_id: str, similarity_matrix: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Returns Top-N recommended products based on item similarity.
    Ensures self-similarity is removed so the selected product never appears in the results.
    """
    if (product_id not in similarity_matrix.columns) or (product_id not in similarity_matrix.index):
        return pd.DataFrame()

    similarity_scores = similarity_matrix.loc[:, product_id].drop(labels=[product_id], errors='ignore')
    top_scores = similarity_scores.sort_values(ascending=False).head(top_n)
    if top_scores.empty:
        return pd.DataFrame()

    # Robust naming: pastikan kolom selalu 'Recommended Product ID' dan 'Similarity Score'
    recs = top_scores.rename_axis('Recommended Product ID').reset_index(name='Similarity Score')
    return recs

def style_recommendations(df: pd.DataFrame):
    if df.empty: return df
    return (df.style
              .format({"Similarity Score": "{:.3f}"})
              .background_gradient(subset=["Similarity Score"], cmap="Blues"))

# ================= Header =================
st.markdown("""
<div class="header-band">
  <h1 style="margin:0;">🍲 Food Recommendation System</h1>
  <div class="data-note">Item-Based Collaborative Filtering with Cosine Similarity</div>
  <div style="margin-top:8px;">
    <span class="badge">Collaborative Filtering</span>
    <span class="badge">Cosine Similarity</span>
    <span class="badge">Top-N Recommendations</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ================= Sidebar =================
with st.sidebar:
    st.header("⚙️ Controls")
    TOP_N = st.slider("Top-N Recommendations", min_value=3, max_value=20, value=5,
                      help="Number of similar items to display.")
    st.markdown("---")
    st.caption("📦 About")
    st.write("- Precomputed item–item similarities")
    st.write("- Cosine metric on user rating patterns")
    st.write("- Self-similarity removed")

# ================= Main Body =================
product_list = get_product_list(item_similarity_df)

# KPI Row (removed 'Similarity Matrix Size')
k1, k2 = st.columns(2)
with k1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Products in Model</div>'
        f'<div class="kpi-value">{item_similarity_df.shape[0]:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Top-N (current)</div>'
        f'<div class="kpi-value">{TOP_N}</div></div>', unsafe_allow_html=True)

# Selection (dropdown) + button placed BELOW the dropdown (full width)
st.markdown("### Select a Product")
selected_product = st.selectbox("Product ID", product_list, index=0 if product_list else None,
                                placeholder="Pick a product…")
run = st.button("✨ Get Recommendations", type="primary", use_container_width=True)

st.markdown('<div class="styled-subheader"><b>Recommendations</b></div>', unsafe_allow_html=True)

# Actions
if run and selected_product:
    with st.spinner("Finding similar products…"):
        recs = get_recommendations(selected_product, item_similarity_df, top_n=TOP_N)

    if recs.empty:
        st.info(f"No recommendations were found for **{selected_product}**.")
    else:
        st.success(f"Top {len(recs)} similar products for **{selected_product}**")
        st.dataframe(style_recommendations(recs), use_container_width=True)

        chart_df = recs.set_index("Recommended Product ID")["Similarity Score"]
        st.bar_chart(chart_df)

        csv = recs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download recommendations (CSV)",
            data=csv,
            file_name=f"recs_{selected_product}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Persist last results if user scrolls/interacts
if run:
    st.session_state["last_result"] = recs if 'recs' in locals() else pd.DataFrame()

if "last_result" in st.session_state and not run:
    st.caption("Last results")
    st.dataframe(style_recommendations(st.session_state["last_result"]), use_container_width=True)

# ================= Extra Info =================
with st.expander("ℹ️ How it works"):
    st.markdown("""
- **Item-Based Collaborative Filtering** computes similarity between products using user rating patterns.
- We use **Cosine Similarity**; the Top-N highest scores are returned as recommendations.
- Self-similarity is explicitly removed so the selected product never appears in its own recommendations.
""")

st.markdown("---")
st.markdown("**Dataset**: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data)")
st.markdown("**Created by**: **Naufal Revanda** · [GitHub](https://github.com/nrevanda) · [LinkedIn](https://www.linkedin.com/in/naufalrevanda/)")

