import pandas as pd
import streamlit as st
import numpy as np
from sklearn.utils.validation import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import altair as alt

df = pd.read_csv("https://raw.githubusercontent.com/ZuniAmandaDewi/dataset/main/milknew.csv")

#ambil data
X = df.drop(columns="Grade")  #data testing
y = df.Grade #data class

#sidebar dengan radio
load_data, preprocessing, modelling, implementasi = st.tabs(["Load Data","Prepocessing", "Modeling", "Implementasi"])
#menu = st.sidebar.radio("Menu", ["Home","Load Data", "Preprocessing", "Modelling", "Implementasi"])

# create content
##Halaman home
st.sidebar.title("Klasifikasi Kualitas Susu")
st.sidebar.caption('link datasets : https://raw.githubusercontent.com/ZuniAmandaDewi/dataset/main/milknew.csv')
st.sidebar.text("""
        Klasifikasi ini menggunakan data:
        1. pH               : Min = 3,0
                            Max = 9,5
        2. Temprature       : Min = 33
                            Max = 90
        3. Rasa             : Baik = 1
                            Buruk = 0
        4. Bau              : Baik = 1
                            Buruk = 0
        5. Lemak            : Tinggi = 1
                            Rendah = 0
        6. Kekeruhan        : Tinggi = 1
                            Rendah = 0
        7. Warna            : Min = 240
                            Max = 255
        
        Hasil Klasifikasi:
        * High
        * Medium
        * Low
        """)

##halaman load data
with load_data :
    st.title("Data Asli")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)
    #menampilkan data
    #st.dataframe(df)
    #menampilkan link data asal
    #st.caption('link datasets : https://www.kaggle.com/datasets/cpluzshrijayan/milkquality')

##halaman Preprocessing
with preprocessing :
    st.title("Preprocessing")
    normalisasi = st.multiselect ("Pilih apa yang ingin Anda lakukan :", ["Normalisasi"])
    submit = st.button("Submit")
    if submit :
        if normalisasi :
            #ambil data
            X = data.drop(columns="Grade")  #data testing
            y = data.Grade #data class
                    
            #mengambil nama kolom
            judul = X.columns.copy() 

            #menghitung hasil normalisasi + menampilkan
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            hasil = pd.DataFrame(X,columns=judul)
            st.dataframe(hasil)

            
with modelling :
    st.title("Modelling")

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    metode1 = KNeighborsClassifier(n_neighbors=3)
    metode1.fit(X_train, y_train)

    metode2 = GaussianNB()
    metode2.fit(X_train, y_train)

    metode3 = tree.DecisionTreeClassifier(criterion="gini")
    metode3.fit(X_train, y_train)

    st.write ("Pilih metode yang ingin anda gunakan :")
    met1 = st.checkbox("KNN")
    if met1 :
        st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
        st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * (metode1.score(X_test, y_test))))
    met2 = st.checkbox("Naive Bayes")
    if met2 :
        st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
        st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
    met3 = st.checkbox("Decesion Tree")
    if met3 :
        st.write("Hasil Akurasi Data Training Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
        st.write("Hasil Akurasi Data Testing Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
    submit2 = st.button("Pilih")

    if submit2:      
        if met1 :
            st.write("Metode yang Anda gunakan Adalah KNN")

        elif met2 :
            st.write("Metode yang Anda gunakan Adalah Naive Bayes")

        elif met3 :
            st.write("Metode yang Anda gunakan Adalah Decesion Tree")

        else :
            st.write("Anda Belum Memilih Metode")

with implementasi :
        # section output
    def submit3():
        joblib.load("scaler.save")
        X = scaler.transform([[ph, suhu, rasa, bau, lemak, kekeruhan, warna]])
        # input
        inputs = np.array(X)
        st.subheader("Data yang Anda Inputkan :")
        st.write(inputs)

        # import label encoder
        le = joblib.load("le.save")

        # create output
        if met1:
            metode1 = joblib.load("knn.joblib")
            y_pred1 = metode1.predict(inputs)
            st.title("k-nearest neighbors")
            st.write(f"Data yang Anda masukkan tergolong dalam kelas : {le.inverse_transform(y_pred1)[0]}")
            
        elif met2:
            metode2 = joblib.load("nb.joblib")
            y_pred2 = metode2.predict(inputs)
            st.title("Gaussian Naive Bayes")
            st.write(f"Data yang Anda masukkan tergolong dalam kelas : {le.inverse_transform(y_pred2)[0]}")

        elif met3:
            metode3 = joblib.load("tree.joblib")
            y_pred3 = metode3.predict(inputs)
            st.title("Decision Tree")
            st.write(f"Data yang Anda masukkan tergolong dalam kelas : {le.inverse_transform(y_pred3)[0]}")

        else :
            st.write("Metode yang Anda Pilih Belum Ada, Silahkan Kembali ke Tabs Modelling Untuk memilih Metode")

    st.title("Form Cek Kualitas Susu")
    ph = st.number_input("pH", 3.0, 9.5, step=0.1)
    suhu = st.number_input("Temprature", 33, 90, step=1)
    rasa = st.number_input("Rasa", 0, 1, step=1)
    bau =  st.number_input("Bau", 0, 1, step=1)
    lemak = st.number_input("lemak", 0, 1, step=1)
    kekeruhan = st.number_input("Kekeruhan", 0, 1, step=1)
    warna = st.number_input("Warna", 240, 255, step=1)

    # create button submit
    submitted = st.button("Cek")
    if submitted:
        submit3()

