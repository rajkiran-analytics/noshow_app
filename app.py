import joblib
import pandas as pd
import streamlit as st

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

st.subheader("Agent: Ask a Question")
q = st.text_input("Type 'high risk', 'capacity', or enter an AppointmentID:")

if q:
    q_lower = q.lower()
    
    # 1. Privacy Fail-safe
    if any(w in q_lower for w in BLOCKED):
        st.warning("Privacy Alert: The dataset does not contain personal information like name, phone, or email.")
        
    # 2. Daily Optimization Brief (High Risk Count)
    elif "high" in q_lower or "risk" in q_lower:
        high_count = len(day[day["risk"] == "High"])
        st.info(f"There are {high_count} high-risk appointments scheduled for {date}. We recommend calling them.")
        
    # 3. Schedule Capacity Check
    elif "capacity" in q_lower or "slot" in q_lower:
        total_capacity = day["SlotCapacity"].sum()
        total_booked = day["BookedSlots"].sum()
        st.info(f"Today's total capacity is {total_capacity} slots. Currently {total_booked} slots are booked.")
        
    # 4. Model-Grounded Explanation (Why flagged?)
    elif q.isdigit():
        appt = day[day["AppointmentID"] == int(q)]
        if not appt.empty:
            prob = appt.iloc[0]['prob']
            lead = appt.iloc[0]['LeadTime']
            age = appt.iloc[0]['Age']
            st.success(f"Appointment ID {q} has a {(prob*100):.1f}% no-show risk.")
            st.write(f"**Why was this flagged?** Evidence shows LeadTime is {lead} days and Age is {age}. Higher lead times usually increase no-show risk.")
        else:
            st.error("Appointment ID not found for this date. Please check the ID.")
            
    # Fallback
    else:
        st.write("I am a simple agent. Try asking about 'high risk', 'capacity', or type an exact AppointmentID from the table above.")
