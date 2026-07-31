import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="MovieIQ", layout="wide")

# =========================================================
# STAGE 1: DATA PREPARATION
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    def parse_genres(genre_str):
        try:
            return [g["name"] for g in ast.literal_eval(genre_str)]
        except (ValueError, SyntaxError):
            return []

    df["genre_list"] = df["genres"].apply(parse_genres)
    df["primary_genre"] = df["genre_list"].apply(lambda g: g[0] if g else "Unknown")
    return df

# Allow reviewers without the file to upload their own
uploaded = st.sidebar.file_uploader("Upload movies.csv (optional)", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df["success"] = (df["revenue"] > df["budget"]).astype(int)
    df["genre_list"] = df["genres"].apply(lambda x: [g["name"] for g in ast.literal_eval(x)] if pd.notna(x) else [])
    df["primary_genre"] = df["genre_list"].apply(lambda g: g[0] if g else "Unknown")
else:
    df = load_data()

# =========================================================
# STAGE 4: TRAIN MODEL (cached so it only trains once)
# =========================================================
@st.cache_resource
def train_model(df):
    df_model = pd.get_dummies(df, columns=["primary_genre"], drop_first=True)
    feature_cols = ["budget", "popularity", "runtime", "vote_average"] + \
                   [c for c in df_model.columns if c.startswith("primary_genre_")]
    X = df_model[feature_cols]
    y = df_model["success"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return clf, feature_cols, metrics, importances

clf, feature_cols, metrics, importances = train_model(df)
all_genres = sorted(df["primary_genre"].unique())

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Filters")
selected_genres = st.sidebar.multiselect("Genre", options=all_genres, default=all_genres)
min_vote = st.sidebar.slider("Minimum vote average", 0.0, 10.0, 0.0, 0.1)

filtered_df = df[(df["primary_genre"].isin(selected_genres)) & (df["vote_average"] >= min_vote)]

# =========================================================
# HEADER + KPI CARDS
# =========================================================
st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.caption("A movie is labeled **successful** when its revenue exceeds its budget.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Movies (filtered)", f"{len(filtered_df):,}")
col2.metric("Success Rate", f"{filtered_df['success'].mean():.1%}" if len(filtered_df) else "N/A")
col3.metric("Avg Budget", f"${filtered_df['budget'].mean():,.0f}" if len(filtered_df) else "N/A")
col4.metric("Avg Revenue", f"${filtered_df['revenue'].mean():,.0f}" if len(filtered_df) else "N/A")

st.divider()

# =========================================================
# STAGE 2: EDA CHARTS
# =========================================================
st.header("Exploratory Data Analysis")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Budget vs Revenue", "Genre Trends", "Feature Comparisons", "Correlation Heatmap",
    "Top 10 by Revenue", "Runtime Distribution", "Genre Share", "Budget by Genre", "Prediction Confidence"
])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x="budget", y="revenue", hue="success",
                     palette={0: "red", 1: "green"}, alpha=0.6, ax=ax)
    ax.set_title("Budget vs Revenue")
    st.pyplot(fig)
    st.caption("Higher budgets loosely trend toward higher revenue, but with wide scatter — "
               "budget alone doesn't guarantee success.")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 5))
        filtered_df["primary_genre"].value_counts().plot(kind="bar", color="steelblue", ax=ax)
        ax.set_title("Movies per Genre")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 5))
        filtered_df.groupby("primary_genre")["success"].mean().sort_values(ascending=False).plot(
            kind="bar", color="seagreen", ax=ax)
        ax.set_title("Success Rate by Genre")
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab3:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
        sns.boxplot(data=filtered_df, x="success", y=col, ax=ax)
        ax.set_title(f"{col} by Success")
    st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots(figsize=(7, 6))
    numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    corr = filtered_df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.caption("Budget and revenue are strongly correlated (0.76). Revenue correlates with "
               "success by definition, so it is excluded from the model to avoid data leakage.")

with tab5:
    top10 = filtered_df.nlargest(10, "revenue")[["title", "revenue", "primary_genre"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=top10, y="title", x="revenue", hue="primary_genre", dodge=False, ax=ax)
    ax.set_title("Top 10 Movies by Revenue")
    ax.set_xlabel("Revenue ($)")
    st.pyplot(fig)

with tab6:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df["runtime"], bins=20, kde=True, color="mediumpurple", ax=ax)
    ax.set_title("Runtime Distribution")
    ax.set_xlabel("Runtime (minutes)")
    st.pyplot(fig)

with tab7:
    fig, ax = plt.subplots(figsize=(6, 6))
    genre_counts = filtered_df["primary_genre"].value_counts()
    ax.pie(genre_counts, labels=genre_counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Share of Movies by Genre")
    st.pyplot(fig)

with tab8:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = filtered_df.groupby("primary_genre")["budget"].median().sort_values(ascending=False).index
    sns.boxplot(data=filtered_df, x="primary_genre", y="budget", order=order, ax=ax)
    ax.set_title("Budget Distribution by Genre")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab9:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_probs = clf.predict_proba(
        pd.get_dummies(df, columns=["primary_genre"], drop_first=True).reindex(columns=feature_cols, fill_value=0)
    )[:, 1]
    sns.histplot(all_probs, bins=20, color="teal", ax=ax)
    ax.set_title("Distribution of Predicted Success Probabilities (all movies)")
    ax.set_xlabel("Predicted probability of success")
    st.pyplot(fig)
    st.caption("Shows how confident the model is across the whole dataset — a spike near 1.0 "
               "reflects the class imbalance discussed above.")

st.divider()

# =========================================================
# STAGE 3: STATISTICAL TESTS
# =========================================================
st.header("Statistical Testing")

c1, c2 = st.columns(2)
with c1:
    st.subheader("T-Test: Popularity vs Success")
    succ = df[df["success"] == 1]["popularity"]
    fail = df[df["success"] == 0]["popularity"]
    t_stat, p_val = stats.ttest_ind(succ, fail, equal_var=False)
    st.write(f"t-statistic: `{t_stat:.4f}`")
    st.write(f"p-value: `{p_val:.4f}`")
    if p_val < 0.05:
        st.success("Statistically significant: popularity differs between successful and unsuccessful movies.")
    else:
        st.info("Not statistically significant.")

with c2:
    st.subheader("Chi-Square: Genre vs Success")
    contingency = pd.crosstab(df["primary_genre"], df["success"])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
    st.write(f"chi2 statistic: `{chi2:.4f}`")
    st.write(f"p-value: `{p_chi:.4f}`")
    if p_chi < 0.05:
        st.success("Statistically significant: genre is associated with success.")
    else:
        st.info("Not statistically significant: genre and success look independent.")

st.divider()

# =========================================================
# STAGE 4: MODEL PERFORMANCE
# =========================================================
st.header("Model Performance (Random Forest)")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Precision", f"{metrics['precision']:.1%}")
c3.metric("Recall", f"{metrics['recall']:.1%}")

st.warning(
    "Accuracy looks high, but the dataset is imbalanced (~81% success). Check the confusion "
    "matrix below — the model under-predicts failures. Accuracy alone can be misleading here."
)

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Success"], yticklabels=["Fail", "Success"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with c2:
    fig, ax = plt.subplots(figsize=(6, 4))
    importances.head(8).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Top Feature Importances")
    st.pyplot(fig)

st.divider()

# =========================================================
# STAGE 5: LIVE PREDICTION
# =========================================================
st.header("Predict Success for a New Movie")

with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        input_budget = st.number_input("Budget ($)", min_value=0, value=50_000_000, step=1_000_000)
        input_popularity = st.slider("Popularity", 0.0, 100.0, 50.0)
    with c2:
        input_runtime = st.number_input("Runtime (minutes)", min_value=0, value=120)
        input_vote = st.slider("Vote Average", 0.0, 10.0, 6.0)
    input_genre = st.selectbox("Primary Genre", options=all_genres)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_row = {col: 0 for col in feature_cols}
    input_row["budget"] = input_budget
    input_row["popularity"] = input_popularity
    input_row["runtime"] = input_runtime
    input_row["vote_average"] = input_vote
    genre_col = f"primary_genre_{input_genre}"
    if genre_col in input_row:
        input_row[genre_col] = 1

    input_df = pd.DataFrame([input_row])[feature_cols]
    prediction = clf.predict(input_df)[0]
    probability = clf.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Predicted: **SUCCESS** (probability: {probability:.1%})")
    else:
        st.error(f"Predicted: **NOT SUCCESSFUL** (probability of success: {probability:.1%})")