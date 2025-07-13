import streamlit as st
import pandas as pd
import base64, random, time, datetime, io, pickle
from PIL import Image
from streamlit_tags import st_tags
import pymysql
from pyresparser import ResumeParser
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import nltk
import plotly.express as px

# Download NLTK models
nltk.download('stopwords')
nltk.download('punkt')

# Load ML Model and Tools
with open("resume_classifier.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def predict_role(skills):
    text = " ".join(skills)
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    return le.inverse_transform([pred])[0]

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def pdf_reader(file):
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    with open(file, 'rb') as fp:
        for page in PDFPage.get_pages(fp, caching=True, check_extractable=True):
            interpreter.process_page(page)
    device.close()
    content = retstr.getvalue()
    retstr.close()
    return content

def show_pdf(path):
    with open(path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>', unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for i, (c_name, c_link) in enumerate(course_list[:no_of_reco], 1):
        st.markdown(f"({i}) [{c_name}]({c_link})")
        rec_course.append(c_name)
    return rec_course

# Connect DB
conn = pymysql.connect(host='localhost', user='root', password='Ug@12345', db='resume_parser')
cursor = conn.cursor()

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    insert_sql = """
        INSERT INTO user_data
        VALUES (0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills, courses)
    cursor.execute(insert_sql, values)
    conn.commit()

def run():
    st.set_page_config(page_title="AI Resume Analyzer", page_icon='./Logo/logo2.png')
    st.image(Image.open('./Logo/logo2.png'))
    st.title("AI Resume Analyzer")

    choice = st.sidebar.selectbox("Choose among the given options:", ["User", "Admin"])
    st.sidebar.markdown('[© Developed by Ujjwal Gola](https://www.linkedin.com/feed/)', unsafe_allow_html=True)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            ID INT NOT NULL AUTO_INCREMENT,
            Name VARCHAR(500),
            Email_ID VARCHAR(500),
            resume_score VARCHAR(8),
            Timestamp VARCHAR(50),
            Page_no VARCHAR(5),
            Predicted_Field TEXT,
            User_level TEXT,
            Actual_skills TEXT,
            Recommended_skills TEXT,
            Recommended_courses TEXT,
            PRIMARY KEY (ID)
        );
    """)

    if choice == 'User':
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload your resume, and get smart recommendations</h5>''', unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])

        if pdf_file is not None:
            with st.spinner('Uploading your Resume...'):
                time.sleep(3)
            save_path = './Uploaded_Resumes/' + pdf_file.name
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_path)

            data = ResumeParser(save_path).get_extracted_data()
            if data:
                resume_text = pdf_reader(save_path)
                st.success("Hello " + str(data.get('name', 'Candidate')))
                st.text('Name: ' + str(data.get('name', 'N/A')))
                st.text('Email: ' + str(data.get('email', 'N/A')))
                st.text('Contact: ' + str(data.get('mobile_number', 'N/A')))
                st.text('Resume pages: ' + str(data.get('no_of_pages', 'N/A')))

                level = "Fresher" if data.get('no_of_pages', 0) == 1 else "Intermediate" if data.get('no_of_pages', 0) == 2 else "Experienced"
                st.markdown(f'<h4 style="color: #1ed760;">You are at {level} level!</h4>', unsafe_allow_html=True)

                skills = data.get('skills', [])
                st_tags(label='### Your Current Skills', text='See our skills recommendation below', value=skills, key='skills1')

                reco_field = predict_role(skills) if skills else 'General IT'
                st.success(f"✅ Our ML Model Predicts: `{reco_field}`")

                if reco_field == 'Data Science':
                    recommended_skills = ['ML', 'Pandas', 'Scikit-Learn', 'TensorFlow', 'Streamlit']
                    rec_course = course_recommender(ds_course)
                elif reco_field == 'Web Development':
                    recommended_skills = ['React', 'Express', 'MongoDB', 'Next.js', 'Node.js']
                    rec_course = course_recommender(web_course)
                elif reco_field == 'Android Development':
                    recommended_skills = ['Kotlin', 'Flutter', 'Android SDK', 'SQLite']
                    rec_course = course_recommender(android_course)
                elif reco_field == 'IOS Development':
                    recommended_skills = ['Swift', 'UIKit', 'Xcode']
                    rec_course = course_recommender(ios_course)
                elif reco_field == 'UI-UX Development':
                    recommended_skills = ['Figma', 'Adobe XD', 'Prototyping', 'User Research']
                    rec_course = course_recommender(uiux_course)
                else:
                    recommended_skills = []
                    rec_course = []

                st_tags(label='### Recommended Skills', text='From ML model', value=recommended_skills, key='skills2')

                ts = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                res_score = 0
                sections = ['Objective', 'Declaration', 'Hobbies', 'Achievements', 'Projects']
                for sec in sections:
                    if sec.lower() in resume_text.lower():
                        res_score += 20
                        st.markdown(f"<h5 style='color:green;'>[+] Awesome! {sec} Found</h5>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h5 style='color:black;'>[-] Consider Adding {sec}</h5>", unsafe_allow_html=True)

                st.subheader("**Resume Score**")
                my_bar = st.progress(0)
                for i in range(res_score):
                    my_bar.progress(i + 1)
                    time.sleep(0.02)
                st.success(f"Your Resume Score: {res_score}")
                st.balloons()

                insert_data(data.get('name'), data.get('email'), str(res_score), ts, str(data.get('no_of_pages')), reco_field, level, str(skills), str(recommended_skills), str(rec_course))

                st.header("Bonus: Resume Writing Tips 🎯")
                rv_link, rv_title = random.choice(resume_videos)
                st.subheader("✅ " + rv_title)
                st.video(rv_link)

                st.header("Bonus: Interview Preparation Tips 🎯")
                iv_link, iv_title = random.choice(interview_videos)
                st.subheader("✅ " + iv_title)
                st.video(iv_link)

            else:
                st.error("Error reading resume content.")

    else:
        st.success("Welcome Admin")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type='password')
        if st.button("Login"):
            if user == "golaji" and pwd == "gola12345":
                st.success("Welcome Ujjwal Gola")
                df = pd.read_sql("SELECT * FROM user_data", conn)
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download CSV'), unsafe_allow_html=True)

                st.subheader("Prediction Distribution")
                fig1 = px.pie(df, names='Predicted_Field', title='Predicted Career Field')
                st.plotly_chart(fig1)

                st.subheader("Experience Level Distribution")
                fig2 = px.pie(df, names='User_level', title='Experience Level of Users')
                st.plotly_chart(fig2)
            else:
                st.error("Wrong credentials")

run()


