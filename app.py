
import joblib, pandas as pd, streamlit as st

FEATURES = ["Age", "LeadTime", "SMS_received", "Scholarship", "Hipertension",
            "Diabetes", "Alcoholism", "Handcap", "GenderNum", "SameDay"]
BLOCKED = ["phone", "email", "address", "personal reason", "name"]

model = joblib.load("model_rf3.pkl")

@st.cache_data
def load():
    d = pd.read_csv("schedule_features.csv", parse_dates=["Date"])
    d["prob"] = model.predict_proba(d[FEATURES])[:, 1]
    d["risk"] = pd.cut(d["prob"], [-1, 0.30, 0.35, 2],
                       labels=["Low", "Medium", "High"], right=False)
    d["action"] = d["risk"].map({"High": "Call + waitlist",
                                 "Medium": "Extra SMS",
                                 "Low": "Normal reminder"}).astype(str)
    d.loc[d["ProviderAvailable"] == 0, "action"] = "Reschedule: provider not available"
    return d

df = load()
st.title("Hospital No-Show Daily Brief")

date = st.selectbox("Date", sorted(df["Date"].dt.date.unique()))
day = df[df["Date"].dt.date == date]

st.write("Total:", len(day))
st.write(day["action"].value_counts())
st.dataframe(day[(day["risk"] == "High") & (day["ProviderAvailable"] == 1)]
             .nlargest(10, "prob")[["AppointmentID", "AppointmentTime", "ProviderID", "prob"]])

q = st.text_input("Ask a question")
if q:
    st.write("The dataset does not contain that information."
             if any(w in q.lower() for w in BLOCKED) else "See the brief above.")
