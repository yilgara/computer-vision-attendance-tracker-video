import streamlit as st
from views import employee_management, process_attendance, view_records

st.set_page_config(page_title="Employee Attendance System", layout="wide")
st.title("🏢 Employee Attendance System")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", 
                            ["Employee Management", "Process Attendance", "View Records"])

if page == "Employee Management":
    employee_management.show()
elif page == "Process Attendance":
    process_attendance.show()
elif page == "View Records":
    view_records.show()
