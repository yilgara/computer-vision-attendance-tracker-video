# 🏢 Employee Attendance System 

A complete **computer vision-based employee attendance tracking system** using **DeepFace** for face recognition and **Streamlit** for an interactive web interface. This system allows adding employees, processing video recordings from entry/exit cameras, and generating daily attendance reports automatically in Excel.

---

## Features

- **Employee Management**
  - Add new employees with multiple photos.
  - Automatically compute face embeddings using DeepFace.
  - View, delete, and manage all employee photos.

- **Attendance Processing**
  - Process videos from entry and exit cameras.
  - Detect and recognize employees in real-time using pre-computed embeddings.
  - Skip frames for faster processing (configurable).
  - Generate timestamped attendance logs.

- **Attendance Records**
  - Store daily attendance in Excel files (`attendance_YYYY-MM-DD.xlsx`).
  - Automatically calculate total working hours.
  - Detect employees who are still in office (entry detected but no exit yet).
  - Track late arrivals (after 09:00 AM).
  - Download daily attendance reports.

- **Performance Optimizations**
  - Pre-compute embeddings for faster recognition.
  - Frame skipping during video processing.
  - Supports multiple employees and photos per employee.

---


## Tech Stack

- **Language:** Python
- **Core Libraries:**
  - OpenCV (`cv2`) for computer vision tasks
  - Deepface for face detection & encoding
  - NumPy for data manipulation
  - Pandas for attendance log management
  - Streamlit for web interface


---


## Folder Structure

```
project_root/
│
├─ app.py                    # Main Streamlit application
├─ employee.py               # Employee management functions
├─ embeddings.py             # Embedding computation and storage
├─ attendance.py             # Attendance processing and Excel logs
├─ views/                    # Streamlit page modules
│   ├─ employee_management.py
│   ├─ process_attendance.py
│   └─ view_records.py
├─ employee_photos/          # Folder to store uploaded employee images
├─ employee_data.pkl         # Pickle file storing employee info
├─ employee_embeddings.pkl   # Pickle file storing embeddings for fast recognition
└─ requirements.txt          # Python dependencies
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/employee-attendance-system.git
   cd employee-attendance-system
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   _Dependencies include: `streamlit`, `opencv-python`, `numpy`, `pandas`, `deepface`._

4. **Create directories for employee photos**
   ```bash
   mkdir employee_photos
   ```
   
## Usage

### 1. Run the Streamlit App

Launch the interactive web interface with:
```bash
streamlit run app.py
```

### 2. Navigate the App

- **Employee Management:**  
  Add new employees with multiple photos and manage existing employee profiles.

- **Process Attendance:**  
  Upload entry and exit camera videos and process attendance automatically. The system detects, recognizes, and logs employee entries and exits.

- **View Records:**  
  Browse daily attendance, download Excel reports, and view lists of late arrivals or employees currently in the office.

---

### 3. Attendance Logs

- **Storage:**  
  Attendance logs are automatically saved as `attendance_YYYY-MM-DD.xlsx` files.

- **Columns Included:**  
  - Employee ID  
  - Employee Name  
  - Date  
  - Entry Time  
  - Exit Time  
  - Total Hours

---



## Deployment

The application is deployed and accessible at:  
**[https://computer-vision-attendance-tracker-video-enpsabwula333ktsiph55.streamlit.app](https://computer-vision-attendance-tracker-video-enpsabwula333ktsiph55.streamlit.app)**

---
