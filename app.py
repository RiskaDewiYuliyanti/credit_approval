import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sistem Pinjaman", layout="wide")
st.title("Sistem Pengajuan Pinjaman Koperasi Semarak Dana")

# Load model
try:
    model = joblib.load("loan_model.pkl")
    st.success("Model berhasil dimuat!")
except Exception as e:
    model = None
    st.error(f"Gagal memuat model: {str(e)}")

menu = st.sidebar.selectbox("Navigasi", [
    "Form Menu", 
    "Data Training", 
    "Data Testing", 
    "Klasifikasi", 
    "Form Pengujian"
])

# =========================
# Form Menu
if menu == "Form Menu":
    st.header("Formulir Pengajuan Pinjaman")

    no_of_dependents = st.number_input("Number of Dependents", value=0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Income Annually", value=0)
    loan_amount = st.number_input("Loan Amount", value=0)
    loan_term = st.number_input("Loan Term (in years)", value=1)
    cibil_score = st.number_input("Cibil Score", value=0)
    residential_assets_value = st.number_input("Residential Assets Value", value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", value=0)
    bank_asset_value = st.number_input("Bank Asset Value", value=0)

    if st.button("Klasifikasikan Pinjaman"):
        education_encoded = 1 if education == "Graduate" else 0
        self_employed_encoded = 1 if self_employed == "Yes" else 0

        input_data = pd.DataFrame([{
            "no_of_dependents": no_of_dependents,
            "education": education_encoded,
            "self_employed": self_employed_encoded,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }])

        if model is not None:
            try:
                result = model.predict(input_data)
                st.success(f"Hasil klasifikasi: **{'Lolos' if result[0] == 1 else 'Tidak Lolos'}**")
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {str(e)}")
        else:
            st.warning("Model belum dimuat.")

# =========================
# Data Training
elif menu == "Data Training":
    st.header("Data Training")

    try:
        X_train = pd.read_csv("X_train.csv")
        y_train = pd.read_csv("y_train.csv")

        st.subheader("Tambah Data Training")
        with st.form("form_train"):
            no_of_dependents = st.number_input("Number of Dependents", value=0)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            income_annum = st.number_input("Income Annually", value=0)
            loan_amount = st.number_input("Loan Amount", value=0)
            loan_term = st.number_input("Loan Term (in years)", value=1)
            cibil_score = st.number_input("Cibil Score", value=0)
            residential_assets_value = st.number_input("Residential Assets Value", value=0)
            commercial_assets_value = st.number_input("Commercial Assets Value", value=0)
            luxury_assets_value = st.number_input("Luxury Assets Value", value=0)
            bank_asset_value = st.number_input("Bank Asset Value", value=0)
            label = st.selectbox("Label (Lolos/Tidak)", ["Lolos", "Tidak Lolos"])

            submitted = st.form_submit_button("Simpan")
            if submitted:
                education_encoded = 1 if education == "Graduate" else 0
                self_employed_encoded = 1 if self_employed == "Yes" else 0

                new_row = pd.DataFrame([{
                    "no_of_dependents": no_of_dependents,
                    "education": education_encoded,
                    "self_employed": self_employed_encoded,
                    "income_annum": income_annum,
                    "loan_amount": loan_amount,
                    "loan_term": loan_term,
                    "cibil_score": cibil_score,
                    "residential_assets_value": residential_assets_value,
                    "commercial_assets_value": commercial_assets_value,
                    "luxury_assets_value": luxury_assets_value,
                    "bank_asset_value": bank_asset_value
                }])

                X_train = pd.concat([X_train, new_row], ignore_index=True)
                y_train = pd.concat([y_train, pd.DataFrame([[1 if label == "Lolos" else 0]])], ignore_index=True)

                X_train.to_csv("X_train.csv", index=False)
                y_train.to_csv("y_train.csv", index=False)
                st.success("Data training berhasil disimpan.")

        st.subheader("Hapus Data Training")
        row_to_delete = st.number_input("Index Baris yang Akan Dihapus", min_value=0, max_value=len(X_train)-1, step=1)
        if st.button("Hapus", key="hapus_train"):
            X_train = X_train.drop(index=row_to_delete).reset_index(drop=True)
            y_train = y_train.drop(index=row_to_delete).reset_index(drop=True)
            X_train.to_csv("X_train.csv", index=False)
            y_train.to_csv("y_train.csv", index=False)
            st.success("Baris berhasil dihapus.")

        st.subheader("Data X_train")
        st.dataframe(X_train)

        st.subheader("Label y_train")
        st.dataframe(y_train)

    except Exception as e:
        st.error(f"Gagal membaca data training: {str(e)}")

# =========================
# Data Testing
elif menu == "Data Testing":
    st.header("Data Testing")

    try:
        X_test = pd.read_csv("X_test.csv")
        y_test = pd.read_csv("y_test.csv")

        st.subheader("Tambah Data Testing")
        with st.form("form_test"):
            income_annum = st.number_input("Income Annually", value=0, key="test_income")
            loan_amount = st.number_input("Loan Amount", value=0, key="test_loan")
            label = st.selectbox("Label (Lolos/Tidak)", ["Lolos", "Tidak Lolos"], key="test_label")

            submitted = st.form_submit_button("Simpan")
            if submitted:
                new_row = pd.DataFrame([{
                    "income_annum": income_annum,
                    "loan_amount": loan_amount
                }])
                X_test = pd.concat([X_test, new_row], ignore_index=True)
                y_test = pd.concat([y_test, pd.DataFrame([[1 if label == "Lolos" else 0]])], ignore_index=True)

                X_test.to_csv("X_test.csv", index=False)
                y_test.to_csv("y_test.csv", index=False)
                st.success("Data berhasil disimpan.")

        st.subheader("Hapus Data Testing")
        row_to_delete = st.number_input("Index Baris yang Akan Dihapus", min_value=0, max_value=len(X_test)-1, step=1)
        if st.button("Hapus", key="hapus_test"):
            X_test = X_test.drop(index=row_to_delete).reset_index(drop=True)
            y_test = y_test.drop(index=row_to_delete).reset_index(drop=True)
            X_test.to_csv("X_test.csv", index=False)
            y_test.to_csv("y_test.csv", index=False)
            st.success("Baris berhasil dihapus.")

        st.subheader("Data X_test")
        st.dataframe(X_test)

        st.subheader("Label y_test")
        st.dataframe(y_test)

    except Exception as e:
        st.error(f"Gagal membaca data testing: {str(e)}")

# =========================
# Klasifikasi Cepat
elif menu == "Klasifikasi":
    st.header("Uji Coba Klasifikasi Cepat")

    no_of_dependents = st.number_input("Number of Dependents", value=0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Income Annually", value=0)
    loan_amount = st.number_input("Loan Amount", value=0)
    loan_term = st.number_input("Loan Term (in years)", value=1)

    if st.button("Prediksi Cepat"):
        education_encoded = 1 if education == "Graduate" else 0
        self_employed_encoded = 1 if self_employed == "Yes" else 0

        input_data = pd.DataFrame([{
            "no_of_dependents": 0,
            "education": education_encoded,
            "self_employed": self_employed_encoded,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": 1,
            "cibil_score": 0,
            "residential_assets_value": 0,
            "commercial_assets_value": 0,
            "luxury_assets_value": 0,
            "bank_asset_value": 0
        }])

        if model is not None:
            try:
                result = model.predict(input_data)
                st.info(f"Hasil uji coba: **{'Lolos' if result[0] == 1 else 'Tidak Lolos'}**")
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {str(e)}")
        else:
            st.warning("Model belum dimuat.")

# =========================
elif menu == "Form Pengujian":
    st.header("Form Pengujian")
    st.write("Form ini digunakan untuk membandingkan hasil klasifikasi algoritma Naïve Bayes dengan data asli.")

    # Upload file X_test dan y_test
    x_test_file = st.file_uploader("Upload File X_test (CSV)", type="csv")
    y_test_file = st.file_uploader("Upload File y_test (CSV)", type="csv")

    if x_test_file is not None and y_test_file is not None:
        try:
            # Load data uji
            X_test = pd.read_csv(x_test_file)
            y_test = pd.read_csv(y_test_file).squeeze()  # squeeze jadi Series

            if model is not None:
                y_pred = model.predict(X_test)

                # Hitung klasifikasi tepat dan tidak tepat
                benar = (y_pred == y_test).sum()
                salah = (y_pred != y_test).sum()
                total = len(y_test)
                akurasi = (benar / total) * 100

                st.success("Pengujian berhasil dilakukan.")
                st.write(f"Jumlah Data Testing: **{total}**")
                st.write(f"Jumlah Klasifikasi Tepat (Approved): **{benar}**")
                st.write(f"Jumlah Klasifikasi Tidak Tepat: **{salah}**")
                st.write(f"Persentase Akurasi: **{akurasi:.2f}%**")

            else:
                st.warning("Model belum dimuat.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca data uji: {str(e)}")
    else:
        st.info("Silakan unggah kedua file untuk melakukan pengujian.")
