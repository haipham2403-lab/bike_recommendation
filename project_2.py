import streamlit as st
import pandas as pd
import numpy as np
import pickle
import regex
from gensim import corpora, models

# ===========================
# Cấu hình trang
# ===========================
st.set_page_config(page_title="Project Report", layout="wide")

# ===========================
# Sidebar menu
# ===========================
st.sidebar.title("Menu")
menu_option = st.sidebar.radio("Chọn mục:", 
                               ["Trang chủ", "Giới thiệu", "Gợi ý xe máy cũ", 'Phân loại xe'])

# Thông tin nhóm
st.sidebar.markdown("---")
st.sidebar.markdown("### Nhóm thực hiện")
st.sidebar.markdown("- HV1: Phạm Văn Hải, email: haipham2403@gmail.com")
st.sidebar.markdown("- HV2: Nguyễn Trần Xuân Linh, email: xuanlinh86@gmail.com")

# ===========================
# ===========================

if menu_option == "Trang chủ":
    st.title("Đồ án tốt nghiệp: Đề xuất xe máy tương tự & phân khúc thị trường")
    
    # Banner lớn
    st.image("xe_may_cu.jpg", use_container_width=True)

    st.markdown("""
    <div style="padding:20px; background-color:#f0f8ff; border-radius:10px; margin-top:20px;">
    <h3 style="color:#0288d1;">Học viên thực hiện:</h3>
    <ul style="font-size:16px; line-height:1.6;">
        <li><b>Phạm Văn Hải</b> – email: <a href="mailto:haipham2403@gmail.com">haipham2403@gmail.com</a></li>
        <li><b>Nguyễn Trần Xuân Linh</b> – email: <a href="mailto:xuanlinh86@gmail.com">xuanlinh86@gmail.com</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="margin-top:20px; font-size:16px;">
    Chúng tôi thực hiện đồ án với mục tiêu xây dựng hệ thống:
    <ul>
        <li>Gợi ý các mẫu xe máy tương tự dựa trên nội dung tin đăng.</li>
        <li>Phân khúc thị trường xe máy dựa trên giá, đời xe, km sử dụng và thương hiệu.</li>
    </ul>
    </p>
    """, unsafe_allow_html=True)




elif menu_option == "Giới thiệu":
    st.title("Giới thiệu dự án")
    
    st.image("xe_may_cu.jpg", use_container_width=True)
    
    st.markdown("""
    <div style="padding:20px; background-color:#fff3e0; border-radius:10px; margin-top:20px;">
    <h3 style="color:#f57c00;">Mô tả dự án:</h3>
    <p style="font-size:16px; line-height:1.6;">
    Thị trường xe máy cũ trên các nền tảng trực tuyến như Chợ Tốt có số lượng tin đăng rất lớn, đa dạng theo giá, thương hiệu, đời xe và tình trạng sử dụng. Điều này tạo ra hai nhu cầu quan trọng:
    </p>
    <ol style="font-size:16px; line-height:1.6;">
        <li><b>Gợi ý xe tương tự:</b> Giúp người dùng nhanh chóng tìm các xe có đặc điểm tương đồng.</li>
        <li><b>Phân khúc thị trường xe máy:</b> Nhóm các mẫu xe thành từng phân khúc dựa trên giá, đời xe, km sử dụng và thương hiệu, hỗ trợ phân tích thị trường.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

elif menu_option == "Gợi ý xe máy cũ":
    st.title("Gợi ý xe máy thông minh")
    st.image("Baner_2.jpg", use_container_width=True)

    # ------------------------------
    # Load dữ liệu & mô hình
    # ------------------------------
    dictionary = corpora.Dictionary.load("dictionary.gensim")
    tfidf = models.TfidfModel.load("tfidf_model.pkl")
    with open("xe_gen_sim.pkl", "rb") as f:
        gen_sim = pickle.load(f)

    df_bikes = pd.read_excel('motorbike_cleaned.xlsx', engine='openpyxl')

    # ------------------------------
    # Tiền xử lý
    # ------------------------------
    def simple_word_processing(text):
        text = text.lower()
        text = regex.sub(r"[^0-9a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]", " ", text)
        text = regex.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(text):
        return text.lower().split()

    @st.cache_data
    def build_corpus(df, _dictionary):
        processed_texts = df["Content"].apply(preprocess).tolist()
        return [_dictionary.doc2bow(text) for text in processed_texts]

    corpus = build_corpus(df_bikes, dictionary)

    # ------------------------------
    # Hàm recommend
    # ------------------------------
    def recommend_gensim(df, bike_id, top_n=5):
        matching_indices = df.index[df['id'] == bike_id].tolist()
        if not matching_indices:
            return pd.DataFrame()
        idx = matching_indices[0]
        query_vec = tfidf[corpus[idx]]
        sims = gen_sim[query_vec]
        ranked_idx = np.argsort(sims)[::-1]
        ranked_idx = [i for i in ranked_idx if i != idx]
        top_idx = ranked_idx[:top_n]
        result = df.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result

    def recommend_by_text(df, query, top_n=5):
        processed = simple_word_processing(query)
        tokens = processed.split()
        if not tokens:
            return pd.DataFrame()
        bow = dictionary.doc2bow(tokens)
        sims = gen_sim[tfidf[bow]]
        ranked_idx = np.argsort(sims)[::-1]
        top_idx = ranked_idx[:top_n]
        result = df.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result

    # ------------------------------
    # Hàm hiển thị xe
    # ------------------------------
    def display_recommended_bikes(recommended_bikes, cols=5):
        for i in range(0, len(recommended_bikes), cols):
            col_objects = st.columns(cols)
            for j, col in enumerate(col_objects):
                if i + j < len(recommended_bikes):
                    bike = recommended_bikes.iloc[i + j]
                    with col:
                        st.markdown(f"### {bike['title']}")
                        st.markdown(f"**ID xe:** {bike['id']}")

                        price_val = bike.get('price', None)
                        price = f"{int(price_val):,} VND" if pd.notnull(price_val) else "Chưa cập nhật"
                        st.markdown(f"**Giá:** {price}")

                        if 'brand' in bike.index:
                            st.markdown(f"**Hãng:** {bike['brand']}")
                        if 'model' in bike.index:
                            st.markdown(f"**Dòng xe:** {bike['model']}")
                        if 'mileage' in bike.index:
                            st.markdown(f"**Số km đã đi:** {bike['mileage']:,.0f} km")
                        if 'year' in bike.index:
                            st.markdown(f"**Năm xe đăng ký:** {bike['year']}")
                        if 'href' in bike.index and pd.notnull(bike['href']):
                            st.markdown(f"[Xem chi tiết]({bike['href']})")

                        desc_col = 'description' if 'description' in bike.index else 'Content'
                        truncated = " ".join(str(bike[desc_col]).split()[:70]) + "..."
                        expander = st.expander("Mô tả")
                        expander.write(truncated)

    # =====================================================
    # 🔥 TẠO 2 TAB
    # =====================================================
    tab1, tab2 = st.tabs(["🔎 Tìm kiếm theo từ khóa", "🛵 Gợi ý theo xe đang xem"])

    # =====================================================
    # TAB 1 – Tìm kiếm
    # =====================================================
    with tab1:
        st.subheader("Tìm kiếm xe máy theo từ khóa")
        search_query = st.text_input("Nhập từ khóa (ví dụ: vision 2019, sirius fi, xe tay ga...)")

        if search_query:
            st.write("### Kết quả tìm kiếm:")
            search_results = recommend_by_text(df_bikes, search_query, top_n=5)
            if not search_results.empty:
                display_recommended_bikes(search_results)
            else:
                st.warning("Không tìm thấy xe nào phù hợp.")

    # =====================================================
    # TAB 2 – Chọn xe ngẫu nhiên
    # =====================================================
    with tab2:
        st.subheader("Chọn một xe bạn muốn xem")

        random_bikes = df_bikes.sample(20, random_state=42)
        bike_options = [(row["title"], row["id"]) for _, row in random_bikes.iterrows()]

        selected = st.selectbox("Chọn xe:", options=bike_options, format_func=lambda x: x[0])
        selected_bike_id = selected[1]
        selected_row = df_bikes[df_bikes["id"] == selected_bike_id]

        if not selected_row.empty:
            st.write("### Bạn vừa chọn:")
            st.write("## ", selected_row["title"].values[0])

            # ---- Giá ----
            price_val = selected_row['price'].values[0]
            price = f"{price_val:,.0f} VND" if pd.notnull(price_val) else "Chưa cập nhật"

            # ---- Hãng, dòng xe, mileage ----
            brand = selected_row["brand"].values[0] if "brand" in selected_row.columns else "Không có dữ liệu"
            model = selected_row["model"].values[0] if "model" in selected_row.columns else "Không có dữ liệu"

            mileage_val = selected_row["mileage"].values[0] if "mileage" in selected_row.columns else None
            mileage = f"{mileage_val:,.0f} km" if pd.notnull(mileage_val) else "Chưa cập nhật"

            # ---- Năm đăng ký ----
            year_val = selected_row["year"].values[0] if "year" in selected_row.columns else None
            year_used = str(int(year_val)) if pd.notnull(year_val) else "Chưa cập nhật"

            # 👉 Hiển thị thông tin xe
            st.markdown(f"""
            **Giá xe:** {price}  
            **Hãng:** {brand}  
            **Dòng xe:** {model}  
            **Số km đã đi:** {mileage}  
            **Năm đăng ký:** {year_used}
            """)

            # ---- Mô tả ----
            desc_col = "description" if "description" in selected_row.columns else "Content"
            truncated_description = " ".join(selected_row[desc_col].values[0].split()[:100]) + "..."
            st.write("##### Thông tin:")
            st.write(truncated_description)

            # ---- Link chi tiết ----
            href_val = selected_row['href'].values[0] if 'href' in selected_row.columns else None
            if pd.notnull(href_val):
                st.markdown(f"[Xem chi tiết]({href_val})", unsafe_allow_html=True)

            st.write("##### Các xe máy tương tự bạn có thể quan tâm:")
            recs = recommend_gensim(df_bikes, selected_bike_id, top_n=5)
            display_recommended_bikes(recs)

elif menu_option == "Phân loại xe":
    st.title("Phân loại xe")
    st.image("baner_1.jpg", use_container_width =True)
    # -------------------------------
    # 1. Load mô hình đã lưu
    # -------------------------------
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)


    st.write("""
    Chức năng này giúp phân loại xe dựa trên **Giá xe (VND)**, **Năm đăng ký xe** và **Số km đã đi**.
    Chỉ cần nhập thông tin bên dưới để biết xe thuộc cụm nào.
    """)

    # -------------------------------
    # 2. Nhập dữ liệu từ người dùng
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        price = st.number_input("Giá xe (VND)", min_value=0, step=1_000_000)

    with col2:
        year = st.number_input("Năm đăng ký xe", min_value=0, step=1)

    with col3:
        mileage = st.number_input("Số km đã đi", min_value=0, step=1_000)
    

    # -------------------------------
    # 3. Dự đoán phân cụm
    # -------------------------------
    if st.button("Phân cụm xe"):
        X_new = np.array([[price, year, mileage]])

        # chuẩn hóa theo scaler đã train
        X_scaled = scaler.transform(X_new)

        # dự đoán cụm
        cluster_id = kmeans.predict(X_scaled)[0]

        st.success(f"**➡️ Xe thuộc cụm: {cluster_id}**")

        # mô tả cụm xem cho dễ hiểu (tùy bạn điều chỉnh)
        cluster_desc = {
            0: "Cụm xe giá rẻ, đời cũ – phù hợp người dùng phổ thông.",
            1: "Cụm xe tầm trung – giá hợp lý, chất lượng tốt.",
            2: "Cụm xe cao cấp, đời mới – phân khúc nâng cao.",
            3: "Cụm đặc biệt (giá cao hoặc rất mới)."
        }

        st.info(cluster_desc.get(cluster_id, "Không có mô tả cụm phù hợp."))

        # --------------------------------
        # 4. Gợi ý thêm: hiển thị lại dữ liệu vừa nhập
        # --------------------------------
        st.subheader("📌 Thông tin xe đã nhập:")
        st.write(pd.DataFrame({
            "Giá xe (VND)": [price],
            "Năm đăng ký xe": [year],
            "Số km đã đi": [mileage],
            "Cụm dự đoán": [cluster_id]
        }))

