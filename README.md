# Hospital Appointment No-Show & Scheduling Agent

An agent that predicts which outpatient appointments are likely to be missed, decides an action for each one, and produces a daily schedule brief for clinic staff. Every answer comes from the model or the schedule data, never from invented reasons.

**Live app:** _add your Streamlit link here_

## What it does

| Part | Description |
|---|---|
| Risk model | Random Forest that returns a no-show probability for each appointment |
| Decision policy | Maps risk to an action (reminder call with earlier-slot or waitlist offer, extra SMS, standard reminder). If the provider is unavailable, the appointment is marked for rescheduling first |
| Daily brief | Scores every appointment of a chosen day, applies the policy and summarises counts, actions and the top 10 riskiest appointments |
| Overbooking hint | Flags a full slot that holds a high-risk patient as an overbooking candidate (a project assumption) |
| Explanations | "Why is appointment X flagged?" is answered with what-if effects from the real model |
| Privacy fail-safe | Refuses questions about phone, email, address or names. The app shows only AppointmentID, time, provider, risk probability and action |

## Risk policies

The sidebar lets you switch between two policies.

| Risk level | Adjusted policy (used in the report) | Strict policy (assignment example) |
|---|---|---|
| Low | below 0.30 | below 0.30 |
| Medium | 0.30 to 0.35 | 0.30 to 0.60 |
| High | 0.35 or more | 0.60 or more |

The model's highest probability is about 0.45, so the strict 0.60 cutoff never produces a High risk appointment. The adjusted policy exists for that reason. The thresholds were chosen after looking at the test data.

## Data

- **Primary:** Kaggle "Medical Appointment No Shows" (Brazil), 110,527 appointments from 2016-04-29 to 2016-06-08. After removing 11 invalid rows: 110,516 appointments, 20.19% no-show.
- **Simulated schedule:** 10 providers with 16 slots each per day (160 appointments per day), covering the 5 test days from 2016-06-02 to 2016-06-08 (800 appointments). It includes slot capacity, booked slots and provider availability, so the agent has a schedule to optimise.

## Key findings

- **Lead time is the strongest driver.** No-show rises from 4.7% for same-day bookings to 33.0% after 30+ days.
- **SMS looks harmful overall but is not.** SMS patients have longer lead times. Inside the same lead-time group they miss fewer appointments, by 1.7 to 7.3 percentage points.
- **Age:** no-show is highest at 13 to 17 years (26.6%) and lowest at 60+ (15.2%).

## Model and results

- Random Forest, 10 features (Age, LeadTime, SMS_received, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, GenderNum, SameDay), `n_estimators=100`, `min_samples_leaf=50`.
- Time-based split: training before 2016-06-02 (88,531 rows), testing from 2016-06-02 onward (21,985 rows).
- **ROC-AUC 0.724.** At cutoff 0.30: precision 0.329, recall 0.462, **F1 0.384**.
- Real no-show rate by risk band on the test set (adjusted policy): Low 13.5%, Medium 29.8%, High 36.7%, against 18.5% overall.
- On the 800-appointment simulated schedule, agent decisions matched the policy in 800 of 800 cases. This checks consistency with the rules, not clinical effectiveness.
- Projected reduction in no-show idle slots: 8% to 11%, **assuming** a confirmation call saves 20 to 30% of no-shows and an extra SMS saves 10%. These success rates are assumptions, not measured results.

## Run locally

```bash
git clone https://github.com/rajkiran-analytics/noshow_app.git
cd noshow_app
pip install -r requirements.txt
streamlit run app.py
```

Then pick a date in the app to see that day's brief. Try questions such as:

- `why is appointment 5698989 flagged?`
- `high risk`
- `overbook`
- `reschedule`
- `capacity`

## Repository files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app: daily brief, policy switch, question box |
| `model_rf3.pkl` | Trained Random Forest model |
| `schedule_features.csv` | Simulated schedule with model features |
| `requirements.txt` | Python dependencies |

## Limitations

- AUC of about 0.72 is modest. Many real no-shows fall in the Low band and receive no extra action.
- The schedule is simulated and small, so results are indicative.
- Actions and their success rates are assumptions.
- The privacy filter is keyword based, so cleverly worded questions could slip through.
- Explanations are what-if estimates that change one factor at a time and ignore interactions.

## Built with

Python, pandas, scikit-learn, Streamlit.

## Author

Raj Kiran, [rajkiran-analytics](https://github.com/rajkiran-analytics)
