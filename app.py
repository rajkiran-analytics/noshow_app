import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hospital No-Show Daily Brief", layout="wide")

# ---------------------------------------------------------------- settings
FEATURES = ["Age", "LeadTime", "SMS_received", "Scholarship", "Hipertension",
            "Diabetes", "Alcoholism", "Handcap", "GenderNum", "SameDay"]

# Two policies. (low_cut, high_cut): Low < low_cut <= Medium < high_cut <= High
POLICIES = {
    "Adjusted policy (High >= 0.35) - used in the project report": (0.30, 0.35),
    "Strict policy (High >= 0.60) - assignment example": (0.30, 0.60),
}

ACTION_CALL = "Reminder call + offer earlier slot / waitlist backup"
ACTION_SMS = "Extra SMS reminder"
ACTION_NORMAL = "Standard reminder"
ACTION_RESCHEDULE = "Reschedule: provider not available"

PRIVACY_MESSAGE = ("The dataset does not contain that information. "
                   "I only use appointment data (ID, time, provider, model risk) "
                   "and never suggest contacting a patient using details that are not in the dataset.")
BLOCKED = re.compile(r"\b(phone|mobile|telephone|whatsapp|email|e-mail|address|name|surname|"
                     r"personal reasons?|contact details?)\b", re.I)

model = joblib.load("model_rf3.pkl")
IMPORTANCE = pd.Series(model.feature_importances_, index=FEATURES)


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    d = pd.read_csv("schedule_features.csv", parse_dates=["Date"])
    d["prob"] = model.predict_proba(d[FEATURES])[:, 1]
    return d


def apply_policy(d, low_cut, high_cut):
    d = d.copy()
    d["risk"] = np.select([d["prob"] >= high_cut, d["prob"] >= low_cut],
                          ["High", "Medium"], default="Low")
    # slot is already full and the patient is high risk -> overbooking is worth considering
    d["overbook"] = (d["risk"] == "High") & (d["ProviderAvailable"] == 1) & \
                    (d["BookedSlots"] >= d["SlotCapacity"])

    def pick(r):
        if r["ProviderAvailable"] == 0:          # provider check always comes first
            return ACTION_RESCHEDULE
        return {"High": ACTION_CALL, "Medium": ACTION_SMS}.get(r["risk"], ACTION_NORMAL)

    d["action"] = d.apply(pick, axis=1)
    return d


# ---------------------------------------------------------------- explanation
def explain(row, typical):
    """What-if explanation from the real model: put each factor at its typical value
    and see how much the model's no-show probability changes."""
    base = row[FEATURES].astype(float)
    groups = {f: [f] for f in FEATURES if f not in ("LeadTime", "SameDay")}
    groups["LeadTime (with SameDay)"] = ["LeadTime", "SameDay"]

    variants = [base.values]
    for cols in groups.values():
        x = base.copy()
        x[cols] = typical[cols]
        variants.append(x.values)
    probs = model.predict_proba(pd.DataFrame(variants, columns=FEATURES))[:, 1]
    p = probs[0]

    out = []
    for (name, cols), p_typical in zip(groups.items(), probs[1:]):
        out.append({
            "Factor": name,
            "This appointment": ", ".join(f"{base[c]:g}" for c in cols),
            "Typical value": ", ".join(f"{typical[c]:g}" for c in cols),
            "Model importance (%)": round(IMPORTANCE[cols].sum() * 100, 1),
            "Effect on risk (points)": round((p - p_typical) * 100, 1),
        })
    table = pd.DataFrame(out).sort_values("Effect on risk (points)", key=abs, ascending=False)
    return p, table.reset_index(drop=True)


def explanation_text(appt_id, row, table):
    top = table[table["Effect on risk (points)"].abs() >= 0.5].head(3)
    if top.empty:
        why = "No single factor moves the risk by more than 0.5 points; the score is close to typical."
    else:
        parts = []
        for _, t in top.iterrows():
            way = "raises" if t["Effect on risk (points)"] > 0 else "lowers"
            parts.append(f"**{t['Factor']}** = {t['This appointment']} (typical {t['Typical value']}) "
                         f"{way} the risk by {abs(t['Effect on risk (points)']):.1f} points")
        why = "; ".join(parts) + "."
    return (f"Appointment {appt_id}: model no-show probability **{row['prob'] * 100:.1f}%** "
            f"(risk level **{row['risk']}**). Main drivers: {why}")


# ---------------------------------------------------------------- agent Q&A
def answer(q, day, full, date, typical):
    """Returns (message, optional table). Every answer comes from the data or the model."""
    ql = q.lower()

    # fail-safe 1: asks for information that is not in the dataset
    long_number = re.search(r"\d{10,}", re.sub(r"[ \-]", "", q))
    if BLOCKED.search(q) or long_number:
        return PRIVACY_MESSAGE, None

    # a specific appointment ID anywhere in the question
    m = re.search(r"\b\d{5,9}\b", q)
    if m:
        appt_id = int(m.group())
        hit = full[full["AppointmentID"] == appt_id]
        if hit.empty:
            return f"Appointment ID {appt_id} is not in the dataset.", None
        row = hit.iloc[0]
        _, table = explain(row, typical)
        msg = explanation_text(appt_id, row, table)
        msg += f"\n\nPlanned action under the selected policy: **{row['action']}**."
        if row["overbook"]:
            msg += " The slot is already full, so overbooking one patient can be considered."
        return msg, table

    cols = ["AppointmentID", "AppointmentTime", "ProviderID", "prob", "risk", "action"]
    if "overbook" in ql:
        ob = day[day["overbook"]]
        return (f"{len(ob)} slot(s) on {date} are full and hold a high-risk patient, "
                f"so one extra booking could be considered there."), ob[cols].sort_values("prob", ascending=False)
    if "reschedul" in ql or "unavailable" in ql:
        rs = day[day["ProviderAvailable"] == 0]
        return f"{len(rs)} appointment(s) on {date} need rescheduling because the provider is not available.", rs[cols]
    if "capacity" in ql or "slot" in ql:
        return (f"On {date}: total capacity {int(day['SlotCapacity'].sum())} slots, "
                f"booked {int(day['BookedSlots'].sum())}."), None
    if "high" in ql or "risk" in ql:
        hi = day[(day["risk"] == "High") & (day["ProviderAvailable"] == 1)]
        if hi.empty:
            return (f"No appointments on {date} reach High risk under the selected policy "
                    f"(highest probability that day: {day['prob'].max() * 100:.1f}%)."), None
        return (f"{len(hi)} high-risk appointment(s) on {date}. "
                f"Recommended: {ACTION_CALL.lower()}."), hi.sort_values("prob", ascending=False)[cols]

    return ("I can answer: 'why is appointment <ID> flagged?', 'high risk', 'overbook', "
            "'reschedule' and 'capacity'."), None


# ---------------------------------------------------------------- page
raw = load()
typical = raw[FEATURES].median()
typical["SameDay"] = float(typical["LeadTime"] == 0)

st.title("Hospital No-Show Daily Brief")
st.caption("Random Forest risk model (AUC 0.72 on the held-out test days). "
           "Every number below comes from the model or the schedule file.")

policy_name = st.sidebar.radio("Risk policy", list(POLICIES))
low_cut, high_cut = POLICIES[policy_name]
df = apply_policy(raw, low_cut, high_cut)

st.sidebar.markdown(f"**Thresholds:** Low < {low_cut}, Medium {low_cut} to {high_cut}, High >= {high_cut}")
n_strict_high = int((raw["prob"] >= 0.60).sum())
st.sidebar.info(f"The model's highest probability is {raw['prob'].max():.2f}. "
                f"Appointments at or above 0.60: {n_strict_high}. "
                "That is why the adjusted policy is available.")

date = st.selectbox("Date", sorted(df["Date"].dt.date.unique()))
day = df[df["Date"].dt.date == date]

# ---- daily brief
st.subheader(f"Daily brief for {date}")
c = st.columns(5)
c[0].metric("Appointments", len(day))
c[1].metric("High risk", int((day["risk"] == "High").sum()))
c[2].metric("Medium risk", int((day["risk"] == "Medium").sum()))
c[3].metric("Reschedule", int((day["ProviderAvailable"] == 0).sum()))
c[4].metric("Overbook candidates", int(day["overbook"].sum()))

st.write("**Actions planned**")
st.dataframe(day["action"].value_counts().rename("appointments").to_frame())

st.write("**Top 10 riskiest appointments (provider available)**")
show = ["AppointmentID", "AppointmentTime", "ProviderID", "prob", "risk", "action"]
top10 = day[day["ProviderAvailable"] == 1].nlargest(10, "prob")[show]
st.dataframe(top10.assign(prob=top10["prob"].round(3)), hide_index=True)

# ---- agent
st.subheader("Agent: ask a question")
st.caption("Try: 'why is appointment 5698989 flagged?', 'high risk', 'overbook', 'reschedule', 'capacity'. "
           "Questions about phone, email, address or names are refused.")
q = st.text_input("Your question")
if q:
    msg, table = answer(q, day, df, date, typical)
    if msg == PRIVACY_MESSAGE:
        st.warning(msg)
    else:
        st.info(msg)
    if table is not None:
        st.dataframe(table, hide_index=True)
        if "Effect on risk (points)" in table.columns:
            st.caption("Effect on risk: how many percentage points of no-show probability the model removes "
                       "when that factor is set to its typical value. LeadTime and SameDay are changed together "
                       "because they describe the same thing.")
