import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import statsmodels.api as sm
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ignore FutureWarnings from seaborn/pandas potentially related to CategoricalDtype
warnings.simplefilter(action='ignore', category=FutureWarning)
# --- Define Exact Column Names from excel ---
age_col = 'What is your age group?'
gender_col = 'What is your gender?'
shop_freq_col = 'How often do you shop online?'
spending_col = 'How much do you typically spend on online shopping per month?'
platforms_col = 'Which platforms do you use the most for online shopping? (Select all that apply)'
product_type_col = 'What type of products do you purchase the most online?'
influence_col = 'What influences your decision to buy online the most?'
trust_col_orig = 'On a scale of 1-5, how much do you trust online product reviews?\n(1 = Not at all, 5 = Completely)'
ads_purchase_col = 'Have you ever purchased a product due to an online advertisement?'
discovery_col = 'How do you prefer to discover new products?'
ad_click_freq_col = 'On a scale of 1-5, how often do you click on online ads while shopping?\n(1 = Never, 5 = Very often)'
marketing_eff_col = 'Which type of online marketing do you find most effective?'
influencer_purchase_col = 'Have you ever made a purchase from an influencer recommendation?'
influencer_impact_col = 'On a scale of 1-5, how much do influencer recommendations impact your buying decision?   (1 = Not at all, 5 = A lot)'
trust_reviews_yn_col = 'Do you trust online reviews when making a purchase?'
content_influence_col = 'What type of content influences your buying decision the most?'
shipping_importance_col = 'On a scale of 1-5, how important is free shipping when purchasing online?\n(1 = Not important, 5 = Very important)'
cart_abandon_col = 'On a scale of 1-5, how often do you abandon your cart due to high shipping costs?   (1 = Never, 5 = Very often)'
livestream_col = 'Would you be interested in shopping through live-stream shopping events'
personalized_ads_feel_col = 'How do you feel about personalized ads based on your browsing history?'
data_comfort_col = 'On a scale of 1-5, how comfortable are you with sharing personal data for personalized shopping experiences?\n(1 = Not comfortable, 5 = Very comfortable)'
chatbot_used_col = 'Have you used AI chatbots for online shopping assistance?'
chatbot_rating_col = 'On a scale of 1-5, how would you rate your overall experience with AI-powered shopping assistants?\n(1 = Poor, 5 = Excellent)'
voice_search_col = 'How likely are you to use voice search (Alexa, Google Assistant, etc.) for online shopping?'
sustainability_col = 'How important is sustainability when choosing an online brand to shop from?\n(1 = Not important, 5 = Very important)'
discount_influence_col = 'On a scale of 1-5, how much do discounts or coupon codes influence your purchase?\n(1 = Not at all, 5 = A lot)'
flash_sale_col = 'Have you ever participated in a flash sale for online shopping?'
social_engage_col = 'How often do you engage with brands via social media before making a purchase?\n(1 = Never, 5 = Very often)'
platform_pref_col = 'Do you prefer shopping on a brand’s website or through third-party platforms (Amazon, Flipkart, etc.)?'
comments_col_orig = 'What improvements would you like to see in online shopping experiences? (Open-ended)'
comments_col_new = 'comments'
# --- Configuration ---
st.set_page_config(
    page_title="Online Shopping Behavior Survey Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load excel file and return a DataFrame."""
    try:
        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()
        st.success(f"Successfully loaded data from {path}")
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at '{path}'. Please check the path.")
        return None
    except Exception as e:
        st.error(f"Error loading excel file: {e}")
        return None

@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by handling missing values, data types, and categorical orders."""
    if df is None:
        st.warning("Input DataFrame is None. Skipping cleaning.")
        return None

    df_cleaned = df.copy()
    st.info(f"Starting cleaning process with {len(df_cleaned)} rows.")

    # --- Data Cleaning Steps ---
    required_cols = [age_col, gender_col]
    missing_essentials = [col for col in required_cols if col not in df_cleaned.columns]
    if missing_essentials:
        st.error(f"Essential columns missing from data: {', '.join(missing_essentials)}. Cannot proceed with cleaning.")
        return None

    original_rows = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=required_cols)
    rows_dropped = original_rows - len(df_cleaned)
    if rows_dropped > 0:
        st.info(f"Dropped {rows_dropped} rows due to missing Age or Gender.")

    # Convert numerical columns (1-5 scales) to numeric type
    numerical_cols_to_convert = [
        trust_col_orig, ad_click_freq_col, influencer_impact_col,
        shipping_importance_col, cart_abandon_col, data_comfort_col,
        chatbot_rating_col, voice_search_col, sustainability_col,
        discount_influence_col, social_engage_col
    ]
    for col in numerical_cols_to_convert:
        if col in df_cleaned.columns:
            original_dtype = df_cleaned[col].dtype
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                st.warning(f"Column '{col}' could not be converted to numeric (original type: {original_dtype}). Check data for non-numeric entries.")
            else:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        else:
            st.warning(f"Numerical column '{col}' not found in data. Skipping conversion.")

    # Encode binary columns (fix case sensitivity)
    binary_cols = {
        ads_purchase_col: {'Yes': 1, 'No': 0},  # Match dataset case
        influencer_purchase_col: {'Yes': 1, 'No': 0},  # Match dataset case
        livestream_col: {'No': 0, 'Yes': 1},
        flash_sale_col: {'No': 0, 'Yes': 1}
    }
    for col, mapping in binary_cols.items():
        if col in df_cleaned.columns:
            # Convert values to string and strip whitespace
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            df_cleaned[col] = df_cleaned[col].map(mapping)
            if df_cleaned[col].isna().any():
                st.warning(f"After mapping, column '{col}' contains NaN values. Unmapped values: {df_cleaned[col][df_cleaned[col].isna()].index.tolist()}")
                df_cleaned[col].fillna(0, inplace=True)

    # Handle comments column
    if comments_col_orig in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={comments_col_orig: comments_col_new})
        df_cleaned[comments_col_new] = df_cleaned[comments_col_new].fillna('Not provided')
    elif comments_col_new not in df_cleaned.columns:
        st.warning(f"Comments column '{comments_col_orig}' not found. Creating empty '{comments_col_new}' column.")
        df_cleaned[comments_col_new] = 'Not provided'

    # Categorical columns with ordered categories
    if age_col in df_cleaned.columns:
        df_cleaned[age_col] = df_cleaned[age_col].astype(str).str.strip()
        age_order = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55+']
        present_categories = [cat for cat in age_order if cat in df_cleaned[age_col].unique()]
        if present_categories:
            df_cleaned[age_col] = pd.Categorical(df_cleaned[age_col], categories=present_categories, ordered=True)
        else:
            st.warning(f"Could not establish ordered categories for '{age_col}'. Found values: {df_cleaned[age_col].unique()}")
            df_cleaned[age_col] = df_cleaned[age_col].astype('category')

    if gender_col in df_cleaned.columns:
        df_cleaned[gender_col] = df_cleaned[gender_col].astype('category')

    if shop_freq_col in df_cleaned.columns:
        df_cleaned[shop_freq_col] = df_cleaned[shop_freq_col].astype(str).str.strip()
        freq_order = ['Never', 'Rarely', 'Occasionally', 'Monthly', 'Weekly', 'Daily']
        present_freq = [freq for freq in freq_order if freq.lower() in df_cleaned[shop_freq_col].str.lower().unique()]
        if present_freq:
            df_cleaned[shop_freq_col] = pd.Categorical(df_cleaned[shop_freq_col], categories=present_freq, ordered=True)
        else:
            st.warning(f"Could not establish ordered categories for '{shop_freq_col}'. Found values: {df_cleaned[shop_freq_col].unique()}")
            df_cleaned[shop_freq_col] = df_cleaned[shop_freq_col].astype('category')

    if spending_col in df_cleaned.columns:
        df_cleaned[spending_col] = df_cleaned[spending_col].astype(str).str.strip()
        spend_order = [
            'Less than Rs.500', 'Rs.0-500',
            'Rs.500-1000',
            'Rs.1000-2000',
            'Rs.2000 and above', 'Rs.2000+'
        ]
        present_spend = [s for s in spend_order if s in df_cleaned[spending_col].unique()]
        if present_spend:
            df_cleaned[spending_col] = pd.Categorical(df_cleaned[spending_col], categories=present_spend, ordered=True)
        else:
            st.warning(f"Could not establish ordered categories for '{spending_col}'. Found values: {df_cleaned[spending_col].unique()}")
            df_cleaned[spending_col] = df_cleaned[spending_col].astype('category')

    # Other categorical columns
    categorical_cols = [
        platforms_col, product_type_col, influence_col, ads_purchase_col,
        discovery_col, marketing_eff_col, influencer_purchase_col,
        trust_reviews_yn_col, content_influence_col, livestream_col,
        personalized_ads_feel_col, chatbot_used_col, flash_sale_col,
        platform_pref_col, gender_col
    ]
    for col in categorical_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype('category')

    # Drop duplicates
    original_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    rows_dropped = original_rows - len(df_cleaned)
    if rows_dropped > 0:
        st.info(f"Dropped {rows_dropped} duplicate rows.")

    # Drop unused columns
    for col in ['Timestamp', 'E-mail']:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[col])

    st.success(f"Cleaning finished. Returning {len(df_cleaned)} rows.")
    return df_cleaned
def encode_variables_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables for correlation and regression analysis."""
    df_encoded = df.copy()

    # Define mappings
    shop_freq_map = {'Never': 1, 'Rarely': 2, 'Occasionally': 3, 'Monthly': 4, 'Weekly': 5, 'Daily': 6}
    spending_map = {
        'Less than Rs.500': 1, 'Rs.0-500': 1,
        'Rs.500-1000': 2,
        'Rs.1000-2000': 3,
        'Rs.2000 and above': 4, 'Rs.2000+': 4
    }
    voice_search_map = {
        'Very Unlikely': 1, 'Unlikely': 2, 'Neutral': 3, 'Likely': 4, 'Very Likely': 5
    }
    platform_preference_map = {
        'third-party platforms': 0, 'brand’s website': 1
    }

    # Apply mappings
    mappings = {
        shop_freq_col: shop_freq_map,
        spending_col: spending_map,
        voice_search_col: voice_search_map,
        platform_pref_col: platform_preference_map
    }

    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            # Map the categorical values to numerical values
            df_encoded[col] = df_encoded[col].map(mapping)
            # Convert the column to a numerical type (float) to avoid Categorical dtype issues
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
            # Fill missing values with 0
            df_encoded[col].fillna(0, inplace=True)

    return df_encoded

def plot_correlation_matrix(df: pd.DataFrame, title: str):
    """Generate and display a correlation matrix heatmap."""
    corr_columns = [
        shop_freq_col,
        spending_col,
        ads_purchase_col,
        influencer_purchase_col,
        cart_abandon_col,
        trust_col_orig,
        ad_click_freq_col,
        influencer_impact_col,
        shipping_importance_col,
        data_comfort_col,
        chatbot_rating_col,
        voice_search_col,
        discount_influence_col,
        social_engage_col
    ]

    # Check for missing columns
    missing_cols = [col for col in corr_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns for correlation analysis: {', '.join(missing_cols)}")
        return

    df_encoded = encode_variables_for_correlation(df)
    df_corr = df_encoded[corr_columns].dropna()

    if df_corr.empty:
        st.warning("No data available for correlation matrix after removing missing values.")
        return

    corr_matrix = df_corr.corr().round(2)
    short_labels = [
        'Shop Freq', 'Spending', 'Purchase Ad', 'Purchase Influencer',
        'Cart Abandon', 'Trust Reviews', 'Click Ads', 'Influencer Impact',
        'Free Shipping', 'Comfort Data', 'AI Experience', 'Voice Search',
        'Discount Influence', 'Social Engage'
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        vmin=-1, vmax=1, center=0,
        cmap='viridis',
        square=True,
        annot=True,
        fmt=".2f",
        ax=ax,
        xticklabels=short_labels,
        yticklabels=short_labels
    )
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write("Correlation Matrix Table:")
    st.dataframe(corr_matrix)

def run_linear_regression(df: pd.DataFrame):
    """Run linear regression to predict monthly spending (Q4)."""
    predictors = [
        ad_click_freq_col, influencer_impact_col, trust_col_orig,
        discount_influence_col, chatbot_rating_col, data_comfort_col
    ]
    target = spending_col

    # Check if all columns exist
    missing_cols = [col for col in predictors + [target] if col not in df.columns]
    if missing_cols:
        st.error(
            f"Missing columns for linear regression: {', '.join(missing_cols)}\n"
            f"Expected columns: {', '.join(predictors + [target])}\n"
            f"Available columns in dataset: {', '.join(df.columns.tolist())}"
        )
        return

    # Encode spending for regression
    df_encoded = encode_variables_for_correlation(df)
    X = df_encoded[predictors]
    y = df_encoded[target]

    # Drop rows with NaN in predictors or target
    data = pd.concat([X, y], axis=1).dropna()
    if data.empty:
        st.warning("No complete data available for linear regression.")
        return

    X = data[predictors]
    y = data[target]
    X = sm.add_constant(X)

    # Check for multicollinearity using Variance Inflation Factor (VIF)
    st.subheader("Multicollinearity Check (VIF)")
    vif_data = pd.DataFrame()
    vif_data["Predictor"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data)
    st.write("**Note**: VIF > 5 indicates potential multicollinearity.")

    # Fit model
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        st.error(f"Error fitting linear regression: {e}")
        return

    # Display results
    st.subheader("Linear Regression Results: Predicting Monthly Spending")
    st.write(f"**R²**: {model.rsquared:.3f} (proportion of variance explained)")
    st.write(f"**Adjusted R²**: {model.rsquared_adj:.3f}")

    # Create results table
    results_df = pd.DataFrame({
        'Predictor': ['Constant'] + predictors,
        'Coefficient': model.params.values,
        'Std Error': model.bse.values,
        'p-value': model.pvalues.values,
        'Significant': ['Yes' if p < 0.05 else 'No' for p in model.pvalues]
    })
    results_df['Coefficient'] = results_df['Coefficient'].round(3)
    results_df['Std Error'] = results_df['Std Error'].round(3)
    results_df['p-value'] = results_df['p-value'].round(3)
    st.dataframe(results_df)

    # Plot significant predictors with shorter labels
    short_labels = {
        ad_click_freq_col: 'Click Ads',
        influencer_impact_col: 'Influencer Impact',
        trust_col_orig: 'Trust Reviews',
        discount_influence_col: 'Discount Importance',
        chatbot_rating_col: 'AI Experience',
        data_comfort_col: 'Comfort Data'
    }
    sig_results = results_df[results_df['Significant'] == 'Yes'][results_df['Predictor'] != 'Constant'].copy()
    if not sig_results.empty:
        sig_results['Predictor'] = sig_results['Predictor'].map(short_labels)
        num_predictors = len(sig_results)
        fig_height = max(2, num_predictors * 0.8)
        fig, ax = plt.subplots(figsize=(8, fig_height))
        sns.barplot(
            x='Coefficient',
            y='Predictor',
            data=sig_results,
            palette=sns.color_palette("viridis", n_colors=num_predictors),
            width=0.5,
            ax=ax
        )
        ax.grid(True, axis='both', linestyle='--', alpha=0.7)
        ax.set_title('Significant Predictors of Monthly Spending', fontsize=12)
        ax.set_xlabel('Coefficient', fontsize=10)
        ax.set_ylabel('Predictor', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Interpretation
    st.markdown("**Interpretation**:")
    for _, row in results_df.iterrows():
        if row['Predictor'] != 'Constant' and row['Significant'] == 'Yes':
            short_name = short_labels.get(row['Predictor'], row['Predictor'])
            st.write(f"- **{short_name}**: A one-unit increase is associated with a {row['Coefficient']:.3f} unit change in monthly spending (p={row['p-value']:.3f}).")

def run_logistic_regression_ad_purchase(df: pd.DataFrame):
    """Run logistic regression to predict purchases due to online ads."""
    predictors = [
        shop_freq_col, ad_click_freq_col, influencer_impact_col,
        trust_col_orig, discount_influence_col, data_comfort_col
    ]
    target = ads_purchase_col

    # Check for missing columns
    missing_cols = [col for col in predictors + [target] if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns for logistic regression (Ad Purchase): {', '.join(missing_cols)}")
        return

    # Prepare data
    df_encoded = encode_variables_for_correlation(df)
    df_model = df_encoded[predictors + [target]].dropna()

    if df_model.empty:
        st.warning("No data available for logistic regression (Ad Purchase) after removing missing values.")
        return

    X = df_model[predictors]
    y = df_model[target]

    # Check for variation in the target variable
    if y.nunique() < 2:
        st.warning("Target variable has no variation (all values are the same). Cannot perform logistic regression.")
        st.write(f"Target value counts:\n{y.value_counts()}")
        return

    # Multicollinearity check (VIF)
    st.markdown("**Multicollinearity Check (VIF)**")
    st.markdown("*Note: VIF > 5 indicates potential multicollinearity.*")
    try:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = predictors
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}))

        # Check for high VIF
        high_vif = vif_data[vif_data["VIF"] > 5]
        if not high_vif.empty:
            st.warning("High multicollinearity detected. Consider removing or combining predictors with VIF > 5.")
    except Exception as e:
        st.warning(f"Error computing VIF: {e}")
        return

    # Add constant for statsmodels
    X = sm.add_constant(X)

    # Fit logistic regression
    try:
        logit_model = sm.Logit(y, X).fit(disp=0)
        st.markdown("**Logistic Regression Results (Predicting Purchase from Ad)**")
        st.write(logit_model.summary())

        # Display odds ratios
        odds_ratios = np.exp(logit_model.params)
        odds_df = pd.DataFrame({
            "Predictor": odds_ratios.index,
            "Odds Ratio": odds_ratios.values,
            "p-value": logit_model.pvalues
        })
        st.markdown("**Odds Ratios**")
        st.dataframe(odds_df.style.format({"Odds Ratio": "{:.3f}", "p-value": "{:.3f}"}))
    except Exception as e:
        st.warning(f"Error fitting logistic regression: {e}")
def run_logistic_regression_influencer_purchase(df: pd.DataFrame):
    """Run logistic regression to predict purchases due to influencer recommendations."""
    predictors = [
        ad_click_freq_col, influencer_impact_col, trust_col_orig,
        discount_influence_col, chatbot_rating_col, data_comfort_col
    ]
    target = influencer_purchase_col

    # Check for missing columns
    missing_cols = [col for col in predictors + [target] if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns for logistic regression (Influencer Purchase): {', '.join(missing_cols)}")
        return

    # Filter data to include only rows where chatbot_rating_col is non-missing
    df_filtered = df[df[chatbot_rating_col].notna()]
    if df_filtered.empty:
        st.warning("No data available for logistic regression (Influencer Purchase) after filtering for non-missing chatbot ratings.")
        return

    # Prepare data
    df_encoded = encode_variables_for_correlation(df_filtered)
    df_model = df_encoded[predictors + [target]].dropna()

    if df_model.empty:
        st.warning("No data available for logistic regression (Influencer Purchase) after removing missing values.")
        return

    X = df_model[predictors]
    y = df_model[target]

    # Multicollinearity check (VIF)
    st.markdown("**Multicollinearity Check (VIF)**")
    st.markdown("*Note: VIF > 5 indicates potential multicollinearity.*")
    try:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = predictors
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}))

        # Check for high VIF
        high_vif = vif_data[vif_data["VIF"] > 5]
        if not high_vif.empty:
            st.warning("High multicollinearity detected. Consider removing or combining predictors with VIF > 5.")
    except Exception as e:
        st.warning(f"Error computing VIF: {e}")
        return

    # Add constant for statsmodels
    X = sm.add_constant(X)

    # Fit logistic regression
    try:
        logit_model = sm.Logit(y, X).fit(disp=0)
        st.markdown("**Logistic Regression Results (Predicting Purchase from Influencer)**")
        st.write(logit_model.summary())

        # Display odds ratios
        odds_ratios = np.exp(logit_model.params)
        odds_df = pd.DataFrame({
            "Predictor": odds_ratios.index,
            "Odds Ratio": odds_ratios.values,
            "p-value": logit_model.pvalues
        })
        st.markdown("**Odds Ratios**")
        st.dataframe(odds_df.style.format({"Odds Ratio": "{:.3f}", "p-value": "{:.3f}"}))
    except Exception as e:
        st.warning(f"Error fitting logistic regression: {e}")
def plot_word_cloud(text_series, title):
    """Generate and display a word cloud."""
    if text_series.isnull().all() or text_series.astype(str).str.strip().eq('').all():
        st.warning(f"No text data available for '{title}' Word Cloud.")
        return

    text = ' '.join(review for review in text_series.astype(str).dropna() if review.lower() not in ['nan', 'na', '.', '-', 'not provided', 'none', 'nothing'])
    if not text.strip():
        st.warning(f"Text data for '{title}' Word Cloud is empty after filtering stopwords/placeholders.")
        return

    stopwords = set(STOPWORDS)
    try:
        wordcloud = WordCloud(
            stopwords=stopwords,
            background_color="white",
            width=800,
            height=400,
            colormap='viridis'
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(title, fontsize=16)
        st.pyplot(fig)
        plt.close(fig)
    except ValueError as ve:
        if "zero area" in str(ve):
            st.warning(f"Could not generate word cloud for '{title}'. Not enough words after filtering.")
        else:
            st.error(f"Error generating word cloud for '{title}': {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred during word cloud generation for '{title}': {e}")

# --- Main App ---

def main():
    st.title("🛒📊 Online Shopping Behavior Survey Dashboard")
    st.markdown("Analyze responses from the online shopping survey.")

    st.sidebar.header("Configuration")
    default_path = "C:\\Project\\Data_Analysis_2\\daman responce.xlsx"
    data_path = st.sidebar.text_input("excel file path", value=default_path)

    if not data_path:
        st.error("Please provide a valid excel file path in the sidebar.")
        st.stop()

    # --- Load and Clean Data ---
    df_raw = load_data(data_path)
    if df_raw is None:
        st.warning("Failed to load data. Cannot proceed.")
        st.stop()

    if st.sidebar.checkbox("Show Raw Data (Before Cleaning)", False):
        st.subheader("Raw Survey Data (Before Cleaning)")
        st.dataframe(df_raw)
        st.write(f"Raw data shape: {df_raw.shape}")
        st.write("Raw data columns:", df_raw.columns.tolist())

    df = clean_data(df_raw)
    if df is None or df.empty:
        st.error("Dataframe is empty after cleaning or cleaning failed. Cannot generate dashboard.")
        st.stop()

    if st.sidebar.checkbox("Show Cleaned Data", False):
        st.subheader("Cleaned Survey Data")
        st.dataframe(df)
        st.write(f"Cleaned data shape: {df.shape}")
        st.write("Cleaned data columns:", df.columns.tolist())

    # --- Define Column Names ---
    age_col = 'What is your age group?'
    gender_col = 'What is your gender?'
    shop_freq_col = 'How often do you shop online?'
    spending_col = 'How much do you typically spend on online shopping per month?'
    platforms_col = 'Which platforms do you use the most for online shopping? (Select all that apply)'
    product_type_col = 'What type of products do you purchase the most online?'
    influence_col = 'What influences your decision to buy online the most?'
    trust_col_orig = 'On a scale of 1-5, how much do you trust online product reviews?\n(1 = Not at all, 5 = Completely)'
    shipping_importance_col = 'On a scale of 1-5, how important is free shipping when purchasing online?\n(1 = Not important, 5 = Very important)'
    cart_abandon_col = 'On a scale of 1-5, how often do you abandon your cart due to high shipping costs?   (1 = Never, 5 = Very often)'
    comments_col_new = 'comments'

    # --- Dashboard Layout ---
    st.markdown("---")
    st.subheader("📊 Key Metrics Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Valid Responses", len(df))

    if age_col in df.columns and not df[age_col].isnull().all():
        if pd.api.types.is_categorical_dtype(df[age_col]) or not df[age_col].mode().empty:
            most_frequent_age = df[age_col].mode()
            col2.metric("Most Frequent Age Group", most_frequent_age.iloc[0] if not most_frequent_age.empty else "N/A")
        else:
            col2.metric("Most Frequent Age Group", "N/A (Check Data)")
    else:
        col2.metric("Most Frequent Age Group", "N/A (Column Missing)")

    if trust_col_orig in df.columns and df[trust_col_orig].notna().any():
        avg_trust = df[trust_col_orig].mean()
        col3.metric("Average Trust Score (1-5)", f"{avg_trust:.2f}" if pd.notna(avg_trust) else "N/A")
    else:
        col3.metric("Average Trust Score (1-5)", "N/A (Column Missing or Empty)")

    # --- Demographics ---
    st.markdown("---")
    st.subheader("📋 Demographics")
    col_demo1, col_demo2 = st.columns(2)

    with col_demo1:
        st.markdown("#### Gender Distribution")
        if gender_col in df.columns and df[gender_col].notna().any():
            gender_counts = df[gender_col].value_counts()
            if not gender_counts.empty:
                fig_gender, ax_gender = plt.subplots(figsize=(3, 3))
                viridis_colors = sns.color_palette("viridis", n_colors=len(gender_counts))
                ax_gender.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
                              startangle=90, textprops={'fontsize': 9}, pctdistance=0.85,
                              colors=viridis_colors)
                ax_gender.axis('equal')
                centre_circle = plt.Circle((0,0),0.70,fc='white')
                fig_gender.gca().add_artist(centre_circle)
                st.pyplot(fig_gender)
                plt.close(fig_gender)
            else:
                st.warning("No data available for gender distribution.")
        else:
            st.warning(f"Gender column ('{gender_col}') not found or empty.")

    with col_demo2:
        st.markdown("#### Age Group Distribution")
        if age_col in df.columns and df[age_col].notna().any():
            if pd.api.types.is_categorical_dtype(df[age_col]) and df[age_col].cat.ordered:
                age_counts = df[age_col].value_counts().sort_index()
            else:
                age_counts = df[age_col].value_counts()
            if not age_counts.empty:
                st.bar_chart(age_counts)
            else:
                st.warning("No data available for age group distribution.")
        else:
            st.warning(f"Age Group column ('{age_col}') not found or empty.")

    # --- Shopping Habits ---
    st.markdown("---")
    st.subheader("🛍️ Shopping Habits")
    col_habit1, col_habit2 = st.columns(2)

    with col_habit1:
        st.markdown("#### Shopping Frequency")
        if shop_freq_col in df.columns and df[shop_freq_col].notna().any():
            if pd.api.types.is_categorical_dtype(df[shop_freq_col]) and df[shop_freq_col].cat.ordered:
                freq_counts = df[shop_freq_col].value_counts().sort_index()
            else:
                freq_counts = df[shop_freq_col].value_counts()
            if not freq_counts.empty:
                st.bar_chart(freq_counts)
            else:
                st.warning("No data available for shopping frequency distribution.")
        else:
            st.warning(f"Shopping Frequency column ('{shop_freq_col}') not found or empty.")

    with col_habit2:
        st.markdown("#### Typical Monthly Spending")
        if spending_col in df.columns and df[spending_col].notna().any():
            if pd.api.types.is_categorical_dtype(df[spending_col]) and df[spending_col].cat.ordered:
                spend_counts = df[spending_col].value_counts().sort_index()
            else:
                spend_counts = df[spending_col].value_counts()
            if not spend_counts.empty:
                st.bar_chart(spend_counts)
            else:
                st.warning("No data available for monthly spending distribution.")
        else:
            st.warning(f"Monthly Spending column ('{spending_col}') not found or empty.")

    st.markdown("#### Most Purchased Product Types")
    if product_type_col in df.columns and df[product_type_col].notna().any():
        product_counts = df[product_type_col].value_counts().head(10)
        if not product_counts.empty:
            st.bar_chart(product_counts)
        else:
            st.warning("No data available for product types.")
    else:
        st.warning(f"Product Type column ('{product_type_col}') not found or empty.")

    # --- Purchase Influences & Trust ---
    st.markdown("---")
    st.subheader("🤔 Purchase Influences & Trust")
    col_infl1, col_infl2, col_infl3 = st.columns(3)

    with col_infl1:
        st.markdown("#### Trust in Online Reviews (1-5 Scale)")
        if trust_col_orig in df.columns and df[trust_col_orig].notna().any():
            fig_trust, ax_trust = plt.subplots(figsize=(6, 4))
            sns.histplot(df[trust_col_orig].dropna(), bins=5, kde=False, ax=ax_trust, discrete=True, stat="count", color=sns.color_palette("viridis", 1)[0])
            ax_trust.set_title('Distribution of Trust Scores')
            ax_trust.set_xlabel('Trust Score (1=Not at all, 5=Completely)')
            ax_trust.set_ylabel('Number of Responses')
            ax_trust.set_xticks(range(1, 6))
            st.pyplot(fig_trust)
            plt.close(fig_trust)
        else:
            st.warning(f"Trust score column ('{trust_col_orig}') not found or empty.")

    with col_infl2:
        st.markdown("#### Average Trust Score by Gender")
        if gender_col in df.columns and trust_col_orig in df.columns:
            plot_data = df[[gender_col, trust_col_orig]].dropna()
            if not plot_data.empty and plot_data[gender_col].nunique() > 0:
                fig_tg, ax_tg = plt.subplots(figsize=(6, 4))
                sns.boxplot(x=gender_col, y=trust_col_orig, data=plot_data, ax=ax_tg, palette="viridis")
                ax_tg.set_title('Trust Score Distribution by Gender')
                ax_tg.set_xlabel('Gender')
                ax_tg.set_ylabel('Trust Score (1-5)')
                st.pyplot(fig_tg)
                plt.close(fig_tg)
            else:
                st.warning("Not enough data or distinct groups to plot Trust by Gender.")
        else:
            st.warning(f"Gender ('{gender_col}') or Trust ('{trust_col_orig}') column missing for 'Trust by Gender' plot.")

    with col_infl3:
        st.markdown("#### Average Trust in Reviews by Age Group")
        if age_col in df.columns and trust_col_orig in df.columns:
            trust_by_age = df.groupby(age_col, observed=False)[trust_col_orig].mean().dropna()
            if not trust_by_age.empty:
                fig_age_trust, ax_age_trust = plt.subplots(figsize=(6, 4))
                viridis_single = sns.color_palette("viridis", n_colors=1)[0]
                trust_by_age.plot(
                    kind='line',
                    marker='o',
                    ax=ax_age_trust,
                    color=viridis_single
                )
                ax_age_trust.grid(True, axis='both', linestyle='--', alpha=0.7)
                ax_age_trust.set_title("Trust in Reviews by Age Group")
                ax_age_trust.set_xlabel("Age Group")
                ax_age_trust.set_ylabel("Average Trust Score (1-5)")
                ax_age_trust.set_ylim(bottom=0)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_age_trust)
                plt.close(fig_age_trust)
            else:
                st.warning("No data for trust score by age group.")
        else:
            st.warning(f"Required columns missing for Trust by Age Group.")

    st.markdown("#### What Influences Purchase Decisions Most?")
    if influence_col in df.columns and df[influence_col].notna().any():
        influence_counts = df[influence_col].value_counts().head(15)
        if not influence_counts.empty:
            st.bar_chart(influence_counts)
            st.caption("Note: Shows counts of combined selections.")
        else:
            st.warning("No data available for purchase influence factors.")
    else:
        st.warning(f"Influence factor column ('{influence_col}') not found or empty.")

    # --- Shipping & Costs ---
    st.markdown("---")
    st.subheader("🚚 Shipping & Costs")
    col_ship1, col_ship2 = st.columns(2)

    with col_ship1:
        st.markdown("#### Importance of Free Shipping (1-5 Scale)")
        if shipping_importance_col in df.columns and df[shipping_importance_col].notna().any():
            fig_ship, ax_ship = plt.subplots(figsize=(6, 4))
            sns.histplot(df[shipping_importance_col].dropna(), bins=5, kde=False, ax=ax_ship, discrete=True, stat="count", color=sns.color_palette("viridis", 1)[0])
            ax_ship.set_title('Importance of Free Shipping')
            ax_ship.set_xlabel('Importance (1=Not important, 5=Very important)')
            ax_ship.set_ylabel('Number of Responses')
            ax_ship.set_xticks(range(1, 6))
            st.pyplot(fig_ship)
            plt.close(fig_ship)
        else:
            st.warning(f"Free shipping importance column ('{shipping_importance_col}') not found or empty.")

    with col_ship2:
        st.markdown("#### Cart Abandonment due to Shipping Cost (1-5 Scale)")
        if cart_abandon_col in df.columns and df[cart_abandon_col].notna().any():
            fig_abandon, ax_abandon = plt.subplots(figsize=(6, 4))
            sns.histplot(df[cart_abandon_col].dropna(), bins=5, kde=False, ax=ax_abandon, discrete=True, stat="count", color=sns.color_palette("viridis", 1)[0])
            ax_abandon.set_title('Frequency of Cart Abandonment')
            ax_abandon.set_xlabel('Frequency (1=Never, 5=Very often)')
            ax_abandon.set_ylabel('Number of Responses')
            ax_abandon.set_xticks(range(1, 6))
            st.pyplot(fig_abandon)
            plt.close(fig_abandon)
        else:
            st.warning(f"Cart abandonment column ('{cart_abandon_col}') not found or empty.")

    # --- Correlation Analysis ---
    st.markdown("---")
    st.subheader("🔗 Correlation Analysis")
    st.markdown("Explore relationships between shopping behavior and marketing influence variables.")
    if st.checkbox("Show Correlation Matrix Heatmap", False):
        plot_correlation_matrix(df, "Correlation Matrix of Shopping and Marketing Variables")

    # --- Regression Analysis ---
    st.markdown("---")
    st.subheader("📉 Regression Analysis")
    st.markdown("Analyze factors influencing online shopping behavior.")
    if st.checkbox("Show Linear Regression Results (Predicting Monthly Spending)", False):
        run_linear_regression(df)
    if st.checkbox("Show Logistic Regression Results (Predicting Purchase from Ad)", False):
        run_logistic_regression_ad_purchase(df)
    if st.checkbox("Show Logistic Regression Results (Predicting Purchase from Influencer)", False):
        run_logistic_regression_influencer_purchase(df)

    # --- Improvements & Feedback ---
    st.markdown("---")
    st.subheader("💡 Improvements & Feedback")
    if comments_col_new in df.columns:
        st.markdown("#### Word Cloud: Suggested Improvements")
        plot_word_cloud(df[comments_col_new], "Suggested Improvements")
    else:
        st.warning(f"Comments column ('{comments_col_new}') not found for Word Cloud.")

    if st.sidebar.checkbox("Show Descriptive Statistics", False):
        st.subheader("Descriptive Statistics of Cleaned Data")
        try:
            cols_to_describe = df.select_dtypes(include=['number', 'category']).columns.tolist()
            st.write(df[cols_to_describe].describe(include='all'))
        except Exception as e:
            st.warning(f"Could not generate descriptive statistics: {e}")

# --- Run the App ---
if __name__ == "__main__":
    main()