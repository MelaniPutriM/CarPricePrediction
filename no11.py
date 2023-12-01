import pickle
import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))
df1 = pd.read_csv('CarPrice.csv')

# Sidebar
selected_page = st.sidebar.selectbox('Pilih Halaman:', ['Home', 'Jelajahi Data', 'Prediksi Harga Mobil'])

# Home Page
if selected_page == 'Home':
    st.title('Prediksi Harga Mobil')
    st.header("Selamat datang di prediksi harga mobil!")
    st.image('image3.jpeg', use_column_width=True)


    df1 = pd.read_csv('CarPrice.csv')

    st.subheader("Informasi Dataset")
    st.dataframe(df1)

    st.write("Grafik Highway-mpg")
    chart_highwaympg = pd.DataFrame(df1, columns=["highwaympg"])
    st.line_chart(chart_highwaympg)

    st.write("Grafik curbweight")
    chart_curbweight = pd.DataFrame(df1, columns=["curbweight"])
    st.line_chart(chart_curbweight)

    st.write("Grafik horsepower")
    chart_horsepower = pd.DataFrame(df1, columns=["horsepower"])
    st.line_chart(chart_horsepower)

# Jelajahi Data Page
elif selected_page == 'Jelajahi Data':
    st.title('Jelajahi Data Mobil')
    st.subheader("Jelajahi Data")

    # Sidebar
    selected_feature = st.sidebar.selectbox('Pilih Fitur untuk Visualisasi:', df1.columns)

    chart_data = pd.DataFrame(df1, columns=[selected_feature])
    st.line_chart(chart_data)

    selected_car_index = st.slider("Pilih Indeks Mobil dari Dataset", 0, len(df1)-1, step=1)

    st.subheader("Informasi Mobil Terpilih")
    st.write(f"Detail mobil dengan indeks {selected_car_index}:")
    st.table(df1.loc[selected_car_index])

    st.subheader("Detail Tambahan")
    st.write(f"Detail tambahan untuk mobil dengan indeks {selected_car_index}:")
    st.write(f"Jarak tempuh (highwaympg): {df1['highwaympg'].iloc[selected_car_index]}")
    st.write(f"Berat kendaraan (curbweight): {df1['curbweight'].iloc[selected_car_index]}")
    st.write(f"Tenaga kuda (horsepower): {df1['horsepower'].iloc[selected_car_index]}")

    st.subheader("Visualisasi Fitur Mobil Terpilih")
    selected_car_features = df1.iloc[selected_car_index][["highwaympg", "curbweight", "horsepower"]]
    st.bar_chart(selected_car_features)

# Prediksi Harga Mobil Page
elif selected_page == 'Prediksi Harga Mobil':
    st.title('Prediksi Harga Mobil')
    st.subheader("Prediksi Harga Mobil")

    st.markdown(f"""
        Silakan masukkan detail mobil untuk mendapatkan perkiraan harga:
    """)

    highwaympg = st.slider("Highway", 0, 10000, step=1)
    curbweight = st.slider("Curb Weight", 0, 10000, step=1)
    horsepower = st.slider("Horsepower", 0, 10000, step=1)

    if st.button('Prediksi Harga'):
        car_prediction = model.predict([[highwaympg, curbweight, horsepower]])

        harga_mobil_str = np.array(car_prediction)
        harga_mobil_float = float(harga_mobil_str[0])
        harga_mobil_formatted = "{:,.2f}".format(harga_mobil_float)

        st.markdown(f"""
            Berdasarkan data yang Anda masukkan, perkiraan harga mobil adalah: **Rp{harga_mobil_formatted}**

            Selamat! Anda telah berhasil memprediksi harga mobil.
        """)
        
        st.subheader("Visualisasi Hasil Prediksi")

        df_prediction = pd.DataFrame({
            'Features': ['Highway MPG', 'Curb Weight', 'Horsepower', 'Predicted Price'],
            'Values': [highwaympg, curbweight, horsepower, harga_mobil_float]
        })

        df_prediction['Values'] = df_prediction['Values'].apply(lambda x: '{:.2f}'.format(x).rstrip('0').rstrip('.'))

        st.bar_chart(df_prediction.set_index('Features'))

        st.subheader("Detail Prediksi")
        st.table(df_prediction)
