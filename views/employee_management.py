import streamlit as st
from employee import get_all_employees, save_employee_data, delete_employee
from embeddings import compute_embedding, load_embeddings_for_recognition
from datetime import datetime
import os


def show():
    st.header("👥 Employee Management")
    
    tab1, tab2 = st.tabs(["Add Employee", "Manage Employees"])
    
    with tab1:
        st.subheader("Add New Employee")
        st.info("💡 Tip: Upload 2-4 photos with different angles and lighting for better recognition")

      
        employee_name = st.text_input("Employee Name")
        uploaded_photos = st.file_uploader("Upload Employee Photos", 
                                           type=['jpg','jpeg','png'], accept_multiple_files=True)
        if st.button("Add Photos") and employee_name and uploaded_photos:
            success_count = 0
            error_count = 0
            with st.spinner("Processing photos and computing embeddings..."):
                for uploaded_photo in uploaded_photos:
                  
                    photo_dir = "employee_photos"
                    os.makedirs(photo_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    photo_path = os.path.join(photo_dir, f"{employee_name}_{timestamp}_{uploaded_photo.name}")
                  
                    with open(photo_path, "wb") as f:
                        f.write(uploaded_photo.getbuffer())
                      
                    embedding = compute_embedding(photo_path)
                  
                    if embedding:
                        save_employee_data(employee_name, photo_path, embedding)
                        success_count += 1
                        st.write(f"✅ Processed {uploaded_photo.name}")
                    else:
                        error_count += 1
                        os.remove(photo_path)
                        st.write(f"❌ Skipped {uploaded_photo.name}")
                      
            if success_count:
                st.success(f"✅ Added {success_count} photos for {employee_name}")
                if error_count:
                    st.warning(f"⚠️ {error_count} photos skipped")
                embeddings_data = load_embeddings_for_recognition()
              
                st.info(f"🚀 Total embeddings: {len(embeddings_data['embeddings'])}")
                st.rerun()
            else:
                st.error("❌ No valid faces detected.")

    with tab2:
        st.subheader("Manage Employees")
        employees = get_all_employees()
        if employees:
            for emp_id, name, main_photo, photo_count, all_photos in employees:
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                with col1:
                    st.write(f"**{name}**")
                    st.write(f"📸 Photos: {photo_count}")
                with col2:
                    if main_photo and os.path.exists(main_photo):
                        st.image(main_photo, width=100, caption="Main photo")
                with col3:
                    if st.button("View All", key=f"view_{emp_id}"):
                        st.session_state[f'show_photos_{emp_id}'] = not st.session_state.get(f'show_photos_{emp_id}', False)
                with col4:
                    if st.button("Delete", key=f"del_{emp_id}"):
                        delete_employee(emp_id)
                        st.success(f"Employee {name} deleted!")
                        st.rerun()
                    
                    # Show all photos if expanded
                if st.session_state.get(f'show_photos_{emp_id}', False):
                    st.write("All photos:")
                    if len(all_photos) > 0:
                        photo_cols = st.columns(min(len(all_photos), 4))
                        for i, photo_path in enumerate(all_photos):
                            if photo_path and os.path.exists(photo_path):
                                with photo_cols[i % 4]:
                                    st.image(photo_path, width=80)
                    else:
                        st.write("No photos found")
                st.markdown("---")
        else:
            st.info("No employees found. Please add employees first.")
