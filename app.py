import streamlit as st
import pandas as pd
import plotly.express as px

# Safe ML import
try:
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

st.set_page_config(layout="wide")

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center; color:#2E8B57; font-weight:bold;'>
🐦 Bird Species Observation Analysis
</h1>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("cleaned_data.csv")

# ---------------- MONTH MAP ----------------
month_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

df["Month"] = df["Month"].astype("Int64")
df["Month_Name"] = df["Month"].map(month_map)

month_order = list(month_map.values())

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filters")

# Habitat
habitat = st.sidebar.multiselect(
    "🌳 Habitat",
    df["Habitat"].unique(),
    default=df["Habitat"].unique()
)

# Month dropdown
selected_month = st.sidebar.selectbox(
    "📅 Select Month",
    ["All"] + month_order
)

# Quick seasonal checkboxes
st.sidebar.markdown("### ⚡ Quick Season Filter")
may = st.sidebar.checkbox("May")
june = st.sidebar.checkbox("June")
july = st.sidebar.checkbox("July")

# ---------------- FILTER LOGIC ----------------
filtered = df.copy()

if habitat:
    filtered = filtered[filtered["Habitat"].isin(habitat)]

# Dropdown filter
if selected_month != "All":
    filtered = filtered[filtered["Month_Name"] == selected_month]

# Checkbox override (summer focus)
season_months = []
if may: season_months.append("May")
if june: season_months.append("June")
if july: season_months.append("July")

if season_months:
    filtered = filtered[filtered["Month_Name"].isin(season_months)]

# ---------------- KPI STYLE (FIXED VISIBILITY) ----------------
def kpi(card, value, label):
    card.markdown(f"""
        <div style='text-align:center; padding:20px;
        background:#ffffff; border-radius:12px;
        border:1px solid #ddd;'>
            <div style='font-size:32px; font-weight:bold; color:#000;'>{value}</div>
            <div style='font-size:14px; color:#333; margin-top:5px;'>{label}</div>
        </div>
    """, unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "🧬 Species", 
    "⏳ Time", 
    "🌦️ Environment", 
    "🧭 Behavior", 
    "🌍 Hotspots"
])

# ================= OVERVIEW =================
with tab1:
    col1, col2, col3 = st.columns(3)

    kpi(col1, len(filtered), "Total Observations")
    kpi(col2, filtered["Common_Name"].nunique(), "Total Species")
    kpi(col3, round(filtered["Temperature"].mean(), 2), "Avg Temperature")

    habitat_dist = filtered["Habitat"].value_counts().reset_index()
    habitat_dist.columns = ["Habitat", "Count"]

    fig = px.bar(
        habitat_dist,
        x="Habitat",
        y="Count",
        text="Count",
        color="Habitat",
        title="🌳 Observation Distribution by Habitat"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Habitat", yaxis_title="Observations")

    st.plotly_chart(fig, use_container_width=True)

# ================= SPECIES =================
with tab2:
    top_species = filtered["Common_Name"].value_counts().head(10).reset_index()
    top_species.columns = ["Species", "Count"]

    fig = px.bar(
        top_species,
        x="Count",
        y="Species",
        orientation="h",
        text="Count",
        color="Count",
        title="🧬 Top Bird Species"
    )
    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

# ================= TIME =================
with tab3:
    monthly = filtered.groupby("Month_Name").size().reset_index(name="Count")

    monthly["Month_Name"] = pd.Categorical(
        monthly["Month_Name"],
        categories=month_order,
        ordered=True
    )

    monthly = monthly.sort_values("Month_Name")

    fig = px.bar(
        monthly,
        x="Month_Name",
        y="Count",
        text="Count",
        color="Count",
        title="📅 Monthly Bird Activity"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Month", yaxis_title="Observations")

    st.plotly_chart(fig, use_container_width=True)

# ================= ENVIRONMENT =================
with tab4:
    fig = px.scatter(
        filtered,
        x="Temperature",
        y="Humidity",
        color="Habitat",
        title="🌦️ Weather Impact on Bird Activity"
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= BEHAVIOR =================
with tab5:
    if "Distance" in filtered.columns:
        dist = filtered["Distance"].value_counts().reset_index()
        dist.columns = ["Distance", "Count"]

        fig = px.bar(
            dist,
            x="Distance",
            y="Count",
            text="Count",
            color="Count",
            title="🧭 Distance Detection Analysis"
        )
        fig.update_traces(textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

# ================= HOTSPOTS =================
with tab6:
    hotspot = filtered["Plot_Name"].value_counts().reset_index()
    hotspot.columns = ["Location", "Count"]

    fig = px.bar(
        hotspot.head(15),
        x="Location",
        y="Count",
        text="Count",
        color="Count",
        title="🌍 Top Bird Activity Locations"
    )
    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

# ================= ML =================
if ML_AVAILABLE:
    st.header("🤖 Prediction Model")

    model_df = filtered.dropna(subset=["Temperature", "Humidity", "Habitat"])

    model_df["Habitat"] = model_df["Habitat"].astype("category").cat.codes
    model_df["Species_Code"] = model_df["Common_Name"].astype("category").cat.codes

    X = model_df[["Temperature", "Humidity", "Habitat"]]
    y = model_df["Species_Code"]

    model = RandomForestClassifier()
    model.fit(X, y)

    temp = st.slider("Temperature", float(df["Temperature"].min()), float(df["Temperature"].max()))
    hum = st.slider("Humidity", float(df["Humidity"].min()), float(df["Humidity"].max()))
    hab = st.selectbox("Habitat", ["Forest", "Grassland"])

    hab_code = 0 if hab == "Forest" else 1

    pred = model.predict([[temp, hum, hab_code]])

    species_map = dict(enumerate(model_df["Common_Name"].astype("category").cat.categories))

    st.success(f"Predicted Species: {species_map[pred[0]]}")