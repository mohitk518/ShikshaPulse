# ShikshaPulse – MVP (Prasunex Innovathon 2026, Team Code Crusaders)
Early-warning system for school students who are falling behind or may drop out, built as an add-on to a school ERP.

## What the MVP does
1. Teacher login (demo: any email / password `demo123`)
2. Upload ERP Excel/CSV export (sample: `sample_students.csv`) – dashboard updates instantly
3. ML risk score 0-100 (logistic regression trained on 6,000 synthetic students, `train_model.py`) with top-3 reasons from model coefficients
4. Class dashboard + student page with attendance/marks trend charts
5. Weak topic, 3-step teacher plan, Hindi WhatsApp parent message draft (teacher approves before sending)

## CSV columns
`student_id, name, class, week, attendance_pct, test_marks, homework_pct, fee_pending_days, weak_topic (optional)`

## Model results (synthetic data)
AUC 0.747; top-10% list catches 28% of at-risk students (about 2.8x random). Real accuracy will be measured on anonymised school data in the pilot.

## Run
Open `index.html` in a browser (no install). To retrain: `pip install scikit-learn numpy && python train_model.py`, then paste `model.json` into `index.html`.

## Next
ERP API connector, LLM practice sheets, WhatsApp Business API, Supabase auth, principal dashboard, school pilot.
