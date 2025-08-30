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

## Folder Structure

