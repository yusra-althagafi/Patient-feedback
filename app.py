import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="تحليل ملاحظات المرضى", layout="wide")

st.title("نظام تحليل وتصنيف شكاوى وملاحظات المرضى الآلي")
st.caption("قسم المعلوماتية الصحية")

@st.cache_resource
def train_model():
    data = [
        ("الطبيب لم يشرح لي العلاج بشكل كاف واستعجل في الفحص", "قسم الأطباء"),
        ("الدكتور كان ممتازاً واستمع لكل استفساراتي", "قسم الأطباء"),
        ("الممرضة تأخرت جداً في استبدال المغذي وسلوكها حاد", "قسم التمريض"),
        ("تعامل التمريض في العناية المركزة كان رائعاً", "قسم التمريض"),
        ("تأخير كبير في موعد العيادة وانتظرت أكثر من ساعتين", "المواعيد والاستقبال"),
        ("موظف الاستقبال غير متعاون ولم يجد ملفي", "المواعيد والاستقبال"),
        ("الغرفة غير نظيفة ودورة المياه تحتاج إلى صيانة فورية", "النظافة والمرافق"),
        ("المستشفى نظيف جداً والخدمات ممتازة", "النظافة والمرافق")
    ]
    
    df_train = pd.DataFrame(data, columns=["text", "category"])
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df_train["text"], df_train["category"])
    return model

model = train_model()

# دالة تحديد مستوى الأولوية موزعة حسب الأقسام
def get_priority(text):
    # كلمات تدل على أولوية عالية موزعة على الأقسام المختلفة
    high_keywords = [
        "خطأ طبي", "تشخيص خاطئ", "إهمال الطبيب", "نزيف", 
        "سقوط من السرير", "خطأ في الجرعة", "عدوى بالمستشفى"
    ]
    
    # كلمات تدل على أولوية متوسطة موزعة على الأقسام المختلفة
    medium_keywords = [
        "تأخير موعد", "انتظار طويل", "تعامل الممرض", "نظافة الغرفة", 
        "تأخر الفني", "ازدحام الاستقبال", "تعطل الجهاز"
    ]
    
    if any(word in text for word in high_keywords):
        return "عالية جداً (تدخل عاجل)", "high"
    elif any(word in text for word in medium_keywords):
        return "متوسطة (تستوجب المتابعة)", "medium"
    else:
        return "عادية (ملاحظات عامة)", "low"

tab1, tab2 = st.tabs(["تحليل ملاحظة واحدة", "تحليل ملف شكاوى متعدد"])

with tab1:
    st.subheader("تحليل ملاحظة أو شكوى جديدة")
    user_input = st.text_area("أدخل نص الشكوى أو الملاحظة هنا:")
    
    if st.button("تحليل النص الآن"):
        if user_input.strip() != "":
            prediction = model.predict([user_input])[0]
            priority_label, priority_level = get_priority(user_input)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("القسم المعني بالتصنيف", prediction)
            with col_b:
                if priority_level == "high":
                    st.error(f"مستوى الأولوية: {priority_label}")
                elif priority_level == "medium":
                    st.warning(f"مستوى الأولوية: {priority_label}")
                else:
                    st.success(f"مستوى الأولوية: {priority_label}")
        else:
            st.warning("يرجى كتابة نص الملاحظة أولا")

with tab2:
    st.subheader("رفع واستعراض تحليل شامل لبيانات الاستبيانات")
    st.write("يرجى رفع ملف يحتوي على عمود باسم الشكوى")
    
    uploaded_file = st.file_uploader("اختر الملف", type=["csv"])
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            if "الشكوى" in df_uploaded.columns:
                df_uploaded["القسم المقترح"] = model.predict(df_uploaded["الشكوى"])
                df_uploaded["مستوى الأولوية"] = df_uploaded["الشكوى"].apply(lambda x: get_priority(str(x))[0])
                
                st.write("النتائج بعد التصنيف الآلي واحتساب الأولوية:")
                st.dataframe(df_uploaded, use_container_width=True)
                
                st.write("توزيع الشكاوى والملاحظات على الأقسام:")
                st.bar_chart(df_uploaded["القسم المقترح"].value_counts())
            else:
                st.error("الملف المرفوع لا يحتوي على عمود باسم الشكوى")
        except Exception as e:
            st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
