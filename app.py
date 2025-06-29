# Streamlit Attendance App with Face Recognition and Location Check
# Admin login: admin / password

import streamlit as st
import pandas as pd
import cv2
from geopy.distance import geodesic
import os
from datetime import datetime
import numpy as np
from streamlit_js_eval import streamlit_js_eval
import json
from PIL import Image
from io import BytesIO
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load face models using official pretrained weights (no .pth files needed)
mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device='cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Company location
COMPANY_LOCATION = (30.022353800687085, 31.46491236103251)
MAX_DISTANCE_METERS = 200

# Paths
EMPLOYEE_CSV = 'employees.csv'
ATTENDANCE_CSV = 'attendance.csv'

# Load admin credentials from file
CREDENTIALS_PATH = 'admin_credentials.json'
def load_admin_credentials():
    with open(CREDENTIALS_PATH, 'r') as f:
        creds = json.load(f)
    return creds.get('username', ''), creds.get('password', '')

# Helper functions
def load_employees():
    if os.path.exists(EMPLOYEE_CSV):
        df = pd.read_csv(EMPLOYEE_CSV)
        return df
    return pd.DataFrame(columns=['name'])

def save_employees(df):
    df.to_csv(EMPLOYEE_CSV, index=False)

def load_attendance():
    if os.path.exists(ATTENDANCE_CSV):
        return pd.read_csv(ATTENDANCE_CSV)
    return pd.DataFrame(columns=['name', 'datetime', 'latitude', 'longitude'])

def save_attendance(df):
    df.to_csv(ATTENDANCE_CSV, index=False)

def check_location(user_loc):
    dist = geodesic(COMPANY_LOCATION, user_loc).meters
    return dist <= MAX_DISTANCE_METERS, dist

def to_rgb_uint8(img_file_or_bytes):
    # Accepts file-like or bytes, returns RGB uint8 numpy array or None
    try:
        if hasattr(img_file_or_bytes, 'getvalue'):
            img_bytes = img_file_or_bytes.getvalue()
        elif isinstance(img_file_or_bytes, bytes):
            img_bytes = img_file_or_bytes
        else:
            return None
        pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
        arr = np.array(pil_img)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    except Exception:
        return None

# Custom embedding extraction

def extract_embedding_from_array(img_array):
    img = Image.fromarray(img_array).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding.squeeze(0).cpu().numpy()

def identify_person_custom(embedding, csv=EMPLOYEE_CSV, threshold=0.8):
    df = pd.read_csv(csv, converters={'face_encoding': eval})
    best_name, best_dist = "Unknown", float('inf')
    for _, row in df.iterrows():
        encoding = row.get('face_encoding', None)
        if not encoding or encoding in ['nan', 'NaN', '']:
            continue
        stored = np.array(encoding)
        dist = np.linalg.norm(embedding - stored)
        if dist < best_dist:
            best_dist, best_name = dist, row['name']
    return best_name if best_dist < threshold else "Unknown"

# Streamlit app
def main():
    # Rerun workaround: check for rerun flag
    if 'do_rerun' in st.session_state and st.session_state.do_rerun:
        st.session_state.do_rerun = False
        st.rerun()

    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    st.title('Employee Attendance System')

    # Sidebar: show logout if admin, else menu
    if st.session_state.admin_logged_in:
        if st.sidebar.button('Logout'):
            st.session_state.admin_logged_in = False
            st.session_state.do_rerun = True
        admin_dashboard()
    else:
        menu = ['Login', 'Attendance']
        choice = st.sidebar.selectbox('Menu', menu)
        if choice == 'Login':
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            if st.button('Login'):
                ADMIN_USER, ADMIN_PASS = load_admin_credentials()
                if username == ADMIN_USER and password == ADMIN_PASS:
                    st.session_state.admin_logged_in = True
                    st.session_state.do_rerun = True
                else:
                    st.error('Invalid credentials')
        else:
            attendance_page()

def admin_dashboard():
    st.title('Admin Dashboard')
    # Download attendance
    att_df = load_attendance()
    st.download_button("Download Attendance CSV", att_df.to_csv(index=False), file_name="attendance.csv", mime="text/csv")
    # Remove employee option
    employees = load_employees()
    if not employees.empty:
        st.subheader("Remove Employee")
        employee_names = employees['name'].tolist()
        selected_remove = st.selectbox("Select employee to remove", employee_names)
        if st.button("Remove Employee"):
            employees = employees[employees['name'] != selected_remove]
            save_employees(employees)
            st.success(f"Employee '{selected_remove}' removed successfully!")
    # Add new employee form
    with st.form("add_employee_form"):
        new_name = st.text_input("Employee Name")
        method = st.radio("Add face by:", ["Upload Image(s)", "Use Camera"])
        uploaded_images = None
        camera_image = None
        if method == "Upload Image(s)":
            uploaded_images = st.file_uploader('Upload Employee Face Images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        else:
            camera_image = st.camera_input('Capture Employee Face')
        submitted = st.form_submit_button("Add New Employee")
        if submitted:
            if not new_name.strip():
                st.warning("Please enter a valid name.")
            elif (method == "Upload Image(s)" and not uploaded_images) or (method == "Use Camera" and not camera_image):
                st.warning("Please provide at least one image.")
            else:
                face_found = False
                embeddings = []
                images_to_process = []
                if method == "Upload Image(s)":
                    for img_file in uploaded_images:
                        rgb_img = to_rgb_uint8(img_file)
                        if rgb_img is not None:
                            images_to_process.append(rgb_img)
                else:
                    rgb_img = to_rgb_uint8(camera_image)
                    if rgb_img is not None:
                        images_to_process.append(rgb_img)
                for rgb_img in images_to_process:
                    embedding = extract_embedding_from_array(rgb_img)
                    if embedding is not None:
                        face_found = True
                        embeddings.append(embedding)
                if not face_found:
                    st.warning("No face detected in any image. Please try again.")
                else:
                    employees = load_employees()
                    if new_name in employees["name"].values:
                        st.warning("Employee already exists.")
                    else:
                        avg_embedding = np.mean(embeddings, axis=0)
                        encoding_str = json.dumps(avg_embedding.tolist())
                        new_row = pd.DataFrame([{"name": new_name, "face_encoding": encoding_str}])
                        employees = pd.concat([employees, new_row], ignore_index=True)
                        save_employees(employees)
                        st.success(f"Employee '{new_name}' added successfully!")

def attendance_page():
    st.header('Employee Attendance')
    captured_image = st.camera_input('Take a photo')
    recognized_name = None
    recognition_error = None
    allow_submit = False
    if captured_image:
        rgb_img = to_rgb_uint8(captured_image)
        if rgb_img is None:
            st.error("Captured image is not a valid color image. Please use a standard JPG or PNG.")
            recognition_error = 'Invalid image format.'
        else:
            embedding = extract_embedding_from_array(rgb_img)
            if embedding is not None:
                employees = load_employees()
                try:
                    recognized_name = identify_person_custom(embedding)
                except Exception as e:
                    recognition_error = f'Face recognition error: {e}'
                if recognized_name == "Unknown":
                    recognition_error = 'Face not recognized. Please try again with a clearer photo.'
                    st.error(recognition_error)
                    allow_submit = False
                else:
                    allow_submit = True
            else:
                recognition_error = 'No face detected in image.'
                st.error(recognition_error)
                allow_submit = False
    js_code = """
    new Promise((resolve, reject) => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude
                    });
                },
                (error) => {
                    resolve({latitude: null, longitude: null, error: error.message});
                }
            );
        } else {
            resolve({latitude: null, longitude: null, error: 'Geolocation not supported'});
        }
    });
    """
    loc = streamlit_js_eval(js_expressions=js_code, key="get_location")
    latitude = loc.get('latitude') if loc else None
    longitude = loc.get('longitude') if loc else None
    if latitude and longitude:
        st.success(f"Detected location: {latitude}, {longitude}")
    else:
        st.warning('Waiting for location permission...')
    if captured_image and recognized_name and recognized_name != "Unknown":
        st.success(f"Recognized: {recognized_name}")
    submit_disabled = not (captured_image and allow_submit and latitude and longitude)
    if st.button('Submit Attendance', disabled=submit_disabled):
        user_loc = (latitude, longitude)
        is_near, dist = check_location(user_loc)
        if is_near:
            att_df = load_attendance()
            now = datetime.now()
            check_type = "Check In" if now.hour < 16 else "Check Out"
            name = recognized_name if recognized_name else 'Employee'
            new_row = pd.DataFrame([{'name': name, 'datetime': now, 'latitude': latitude, 'longitude': longitude, 'type': check_type}])
            att_df = pd.concat([att_df, new_row], ignore_index=True)
            save_attendance(att_df)
            st.success(f'{check_type} marked for {name}!')
        else:
            st.error(f'You are too far from the company location ({dist:.2f} meters).')

if __name__ == '__main__':
    main()
