
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------------------
# AI STUDENT PERFORMANCE ANALYZER
# ------------------------------------------

st.set_page_config(
    page_title="AI Student Performance Analyzer",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------------------
# PROFESSIONAL DASHBOARD DESIGN
# ------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown {
        color: #cbd5e1;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #172554, #172033);
        border: 1px solid #334155;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stMetricLabel"] {
        color: #93c5fd !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 1.8rem;
        white-space: normal !important;
        overflow-wrap: anywhere;
    }

    div[data-testid="stForm"] {
        background-color: #111b2e;
        border: 1px solid #334155;
        padding: 24px;
        border-radius: 18px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button,
    div.stDownloadButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 700;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover,
    div.stDownloadButton > button:hover {
        border: 1px solid #93c5fd;
        color: white;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #334155;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# HEADER
# ------------------------------------------

st.title("🎓 AI Student Performance Analyzer")

st.write(
    "Analyze student performance and estimate final marks "
    "using a machine-learning model."
)

st.info(
    "Educational demonstration: predictions use synthetic "
    "training data and are not validated real-world predictions."
)

# ------------------------------------------
# CREATE DEMONSTRATION TRAINING DATA
# ------------------------------------------

@st.cache_resource
def train_model():
    data = []

    for attendance in range(40, 101, 10):
        for study_hours in range(1, 9):
            for previous_marks in range(35, 96, 15):
                for assignment_marks in (40, 60, 80, 100):
                    for internal_marks in (35, 55, 75, 95):

                        final_marks = (
                            0.20 * attendance
                            + 0.20 * study_hours * 10
                            + 0.30 * previous_marks
                            + 0.15 * assignment_marks
                            + 0.15 * internal_marks
                        )

                        final_marks = max(
                            0, min(100, final_marks)
                        )

                        data.append([
                            attendance,
                            study_hours,
                            previous_marks,
                            assignment_marks,
                            internal_marks,
                            final_marks
                        ])

    df = pd.DataFrame(
        data,
        columns=[
            "Attendance",
            "Study Hours",
            "Previous Marks",
            "Assignment Marks",
            "Internal Marks",
            "Final Marks"
        ]
    )

    feature_columns = [
        "Attendance",
        "Study Hours",
        "Previous Marks",
        "Assignment Marks",
        "Internal Marks"
    ]

    X = df[feature_columns]
    y = df["Final Marks"]

    model = LinearRegression()
    model.fit(X, y)

    return model


model = train_model()

# ------------------------------------------
# STUDENT INPUT FORM
# ------------------------------------------

st.subheader("📝 Enter Student Details")

with st.form("student_form"):
    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student name"
    )

    subject = st.selectbox(
        "Subject",
        [
            "Python Programming",
            "Data Structures",
            "Software Engineering",
            "Feature Engineering",
            "Database Management Systems",
            "Other"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider(
            "Attendance (%)", 0, 100, 75
        )

        study_hours = st.slider(
            "Study Hours per Day", 0, 12, 3
        )

        previous_marks = st.number_input(
            "Previous Exam Marks (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

    with col2:
        assignment_marks = st.number_input(
            "Assignment Marks (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        internal_marks = st.number_input(
            "Internal Marks (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

    submitted = st.form_submit_button(
        "🔍 Analyze Performance",
        use_container_width=True
    )

# ------------------------------------------
# PREDICTION AND RESULTS
# ------------------------------------------

if submitted:
    if not student_name.strip():
        st.error("Please enter the student's name.")

    else:
        input_data = pd.DataFrame(
            [[
                attendance,
                study_hours,
                previous_marks,
                assignment_marks,
                internal_marks
            ]],
            columns=[
                "Attendance",
                "Study Hours",
                "Previous Marks",
                "Assignment Marks",
                "Internal Marks"
            ]
        )

        prediction = float(model.predict(input_data)[0])
        prediction = max(0, min(100, prediction))
        prediction = round(prediction, 2)

        # Grade and performance category
        if prediction >= 90:
            grade = "A+"
            category = "Excellent"
        elif prediction >= 80:
            grade = "A"
            category = "Very Good"
        elif prediction >= 70:
            grade = "B"
            category = "Good"
        elif prediction >= 60:
            grade = "C"
            category = "Average"
        elif prediction >= 40:
            grade = "D"
            category = "Needs Improvement"
        else:
            grade = "F"
            category = "At Risk"

        st.divider()
        st.subheader("📊 Performance Analysis")

        st.write(f"**Student:** {student_name}")
        st.write(f"**Subject:** {subject}")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Estimated Final Marks",
            f"{prediction}/100"
        )

        c2.metric("Predicted Grade", grade)
        c3.metric("Performance Category", category)

        st.progress(int(prediction))

        # ----------------------------------
        # ACADEMIC PERFORMANCE CHART
        # ----------------------------------

        st.subheader("📈 Academic Performance Breakdown")

        chart_data = pd.DataFrame({
            "Assessment": [
                "Previous Exam",
                "Assignment",
                "Internal",
                "Estimated Final"
            ],
            "Marks": [
                previous_marks,
                assignment_marks,
                internal_marks,
                prediction
            ]
        })

        st.bar_chart(
            chart_data,
            x="Assessment",
            y="Marks",
            y_label="Marks (out of 100)",
            color="#3B82F6",
            height=350
        )

        # ----------------------------------
        # PERSONALIZED SUGGESTIONS
        # ----------------------------------

        st.subheader("💡 Personalized Suggestions")

        suggestions = []

        if attendance < 75:
            suggestions.append(
                "Improve attendance and attend classes regularly."
            )

        if study_hours < 3:
            suggestions.append(
                "Try maintaining a consistent daily study routine."
            )

        if previous_marks < 50:
            suggestions.append(
                "Revise fundamental concepts and solve practice questions."
            )

        if assignment_marks < 60:
            suggestions.append(
                "Spend more time completing assignments and reviewing mistakes."
            )

        if internal_marks < 60:
            suggestions.append(
                "Practice internal-test questions and clarify difficult topics."
            )

        if not suggestions:
            suggestions.append(
                "Maintain your study routine and continue revising regularly."
            )

        for suggestion in suggestions:
            st.write(f"• {suggestion}")

        # ----------------------------------
        # DOWNLOADABLE STUDENT REPORT
        # ----------------------------------

        st.divider()
        st.subheader("📥 Download Student Report")

        report_data = pd.DataFrame([{
            "Student Name": student_name.strip(),
            "Subject": subject,
            "Attendance (%)": attendance,
            "Study Hours per Day": study_hours,
            "Previous Exam Marks": previous_marks,
            "Assignment Marks": assignment_marks,
            "Internal Marks": internal_marks,
            "Estimated Final Marks": prediction,
            "Predicted Grade": grade,
            "Performance Category": category,
            "Suggestions": " | ".join(suggestions)
        }])

        csv_report = report_data.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download Student Report (CSV)",
            data=csv_report,
            file_name="student_performance_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "This model uses synthetic demonstration data. "
            "Its predictions are illustrative estimates, not "
            "validated forecasts of actual academic performance. "
            "Grade bands are illustrative."
        )

# ------------------------------------------
# FOOTER
# ------------------------------------------

st.divider()

st.caption(
    "AI Student Performance Analyzer | "
    "Python • Streamlit • Pandas • Scikit-learn"
)
