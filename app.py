import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(
    page_title="Cat Health Predictor",
    page_icon="🐱",
    layout="wide"
)

st.title("🐱 Cat Health Predictor")
st.markdown("### Предсказание здоровья кошек на основе машинного обучения")
st.markdown("---")

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_data():
    df = pd.read_csv('cat_2.csv', sep=';')
    # Очищаем названия колонок
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ---
def is_healthy(row):
    """Определяет, здорова ли кошка на основе параметров"""
    problems = 0
    if row['Age'] > 15:  # Старая кошка
        problems += 1
    if row['Weight'] < 2 or row['Weight'] > 8:  # Проблемы с весом
        problems += 1
    if row['Playing (min.)'] < 20:  # Малоактивная
        problems += 1
    if row['Sleeps (hours)'] < 10 or row['Sleeps (hours)'] > 20:  # Нарушение сна
        problems += 1
    return 'Healthy' if problems < 2 else 'Needs Care'

df['Health_Status'] = df.apply(is_healthy, axis=1)

# --- ОБУЧЕНИЕ МОДЕЛИ ---
@st.cache_resource
def train_model():
    features = ['Age', 'Weight', 'Playing (min.)', 'Sleeps (hours)']
    X = df[features].fillna(df[features].mean())
    y = df['Health_Status']
    
    # Кодирование целевой переменной
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    # Оценка точности
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    accuracy = model.score(X_test, y_test)
    
    return model, le, features, accuracy

model, label_encoder, features, accuracy = train_model()

# --- БОКОВАЯ ПАНЕЛЬ - КОНТРОЛЫ (4 штуки) ---
st.sidebar.header("🎮 Управление приложением")

# КОНТРОЛ 1: Выбор породы (мультивыбор)
st.sidebar.subheader("1. Фильтр по породе")
selected_breeds = st.sidebar.multiselect(
    "Выберите породы:",
    options=sorted(df['Breed'].unique()),
    default=[]
)

# КОНТРОЛ 2: Диапазон веса (слайдер)
st.sidebar.subheader("2. Диапазон веса")
min_weight = float(df['Weight'].min())
max_weight = float(df['Weight'].max())
weight_range = st.sidebar.slider(
    "Вес (кг):",
    min_value=min_weight,
    max_value=max_weight,
    value=(min_weight, max_weight),
    step=0.5
)

# КОНТРОЛ 3: Уровень активности (радио-кнопка)
st.sidebar.subheader("3. Уровень активности")
activity_level = st.sidebar.radio(
    "Активность:",
    options=["Все", "Активные (>30 мин)", "Малоподвижные (<20 мин)"]
)

# КОНТРОЛ 4: Показать только здоровых (чекбокс)
st.sidebar.subheader("4. Фильтр здоровья")
show_healthy_only = st.sidebar.checkbox("Показать только здоровых кошек")

# Применяем фильтры
filtered_df = df.copy()

if selected_breeds:
    filtered_df = filtered_df[filtered_df['Breed'].isin(selected_breeds)]

filtered_df = filtered_df[
    (filtered_df['Weight'] >= weight_range[0]) & 
    (filtered_df['Weight'] <= weight_range[1])
]

if activity_level == "Активные (>30 мин)":
    filtered_df = filtered_df[filtered_df['Playing (min.)'] > 30]
elif activity_level == "Малоподвижные (<20 мин)":
    filtered_df = filtered_df[filtered_df['Playing (min.)'] < 20]

if show_healthy_only:
    filtered_df = filtered_df[filtered_df['Health_Status'] == 'Healthy']

# --- ОСНОВНАЯ ОБЛАСТЬ ---
st.header("📊 Обзор данных")
st.info(f"📋 Показано записей: {len(filtered_df)} из {len(df)}")
st.dataframe(filtered_df, use_container_width=True)

# --- ВИЗУАЛИЗАЦИЯ ---
st.header("📈 Визуализация")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Распределение пород")
    breed_counts = filtered_df['Breed'].value_counts()
    if len(breed_counts) > 0:
        st.bar_chart(breed_counts)
    else:
        st.info("Нет данных")

with col2:
    st.subheader("Статус здоровья")
    health_counts = filtered_df['Health_Status'].value_counts()
    if len(health_counts) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_health = ['#2ECC71', '#E74C3C']
        ax.pie(health_counts.values, labels=health_counts.index, 
               autopct='%1.1f%%', colors=colors_health)
        ax.set_title('Соотношение здоровых кошек')
        st.pyplot(fig)
    else:
        st.info("Нет данных")

# --- ПРОГНОЗИРОВАНИЕ НА ОСНОВЕ ML ---
st.header("🤖 Прогнозирование здоровья кошки")
st.markdown(f"*Модель машинного обучения (Random Forest) | Точность: {accuracy*100:.1f}%*")

st.markdown("### Введите данные о кошке:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "📅 Возраст (лет)", 
        min_value=0.0, 
        max_value=30.0, 
        value=3.0,
        step=0.5,
        help="Молодые кошки (до 10 лет) обычно здоровее"
    )
    
    weight = st.number_input(
        "⚖️ Вес (кг)", 
        min_value=1.0, 
        max_value=15.0, 
        value=4.0,
        step=0.5,
        help="Нормальный вес для большинства пород: 3-5 кг"
    )

with col2:
    play_minutes = st.number_input(
        "🎾 Время игр (минут в день)", 
        min_value=0, 
        max_value=120, 
        value=30,
        step=5,
        help="Рекомендуется 20-40 минут активных игр"
    )
    
    sleep_hours = st.number_input(
        "😴 Часы сна в день", 
        min_value=5.0, 
        max_value=24.0, 
        value=14.0,
        step=0.5,
        help="Кошки спят 12-18 часов в сутки"
    )

# Кнопка предсказания
if st.button("🔮 Предсказать здоровье", type="primary", use_container_width=True):
    # Подготовка данных
    input_data = pd.DataFrame([[age, weight, play_minutes, sleep_hours]], columns=features)
    
    # Предсказание
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Результат
    result = label_encoder.inverse_transform([prediction])[0]
    
    st.markdown("---")
    st.subheader("📋 Результат прогноза:")
    
    # Отображение результата
    col_r, col_p = st.columns(2)
    
    with col_r:
        if result == "Healthy":
            st.success("✅ **Кошка здорова!**")
            st.balloons()
        else:
            st.error("⚠️ **Кошка требует внимания!**")
            st.snow()
    
    with col_p:
        if result == "Healthy":
            health_prob = probabilities[1] * 100
            st.metric("Вероятность здоровья", f"{health_prob:.1f}%")
        else:
            risk_prob = probabilities[0] * 100
            st.metric("Вероятность проблем", f"{risk_prob:.1f}%")
    
    # Рекомендации
    st.markdown("---")
    st.markdown("### 💡 Рекомендации:")
    
    recommendations = []
    if age > 12:
        recommendations.append("📌 Пожилая кошка - рекомендуются регулярные осмотры у ветеринара (2 раза в год)")
    if weight < 2:
        recommendations.append("📌 Низкий вес - проконсультируйтесь с ветеринаром о питании")
    if weight > 8:
        recommendations.append("📌 Избыточный вес - рекомендуется диета и увеличение физической активности")
    if play_minutes < 20:
        recommendations.append("📌 Низкая активность - попробуйте увеличить время игр с кошкой")
    if play_minutes > 60:
        recommendations.append("📌 Очень активная кошка - это нормально, но следите чтобы она не переутомлялась")
    if sleep_hours < 10:
        recommendations.append("📌 Мало спит - возможно, кошке что-то мешает")
    if sleep_hours > 20:
        recommendations.append("📌 Спит слишком много - обратите внимание на активность кошки")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✨ Все параметры в норме! Продолжайте заботиться о кошке.")

# --- СТАТИСТИКА МОДЕЛИ ---
with st.expander("📊 Информация о модели машинного обучения"):
    st.markdown(f"""
    **Характеристики модели:**
    
    - **Алгоритм:** Random Forest Classifier
    - **Количество деревьев:** 100
    - **Признаки:** Возраст, Вес, Время игр, Часы сна
    - **Целевая переменная:** Health_Status (Healthy / Needs Care)
    - **Точность модели:** {accuracy*100:.1f}%
    - **Размер датасета:** {len(df)} записей
    
    **Важность признаков:**
    """)
    
    # Важность признаков
    importance_df = pd.DataFrame({
        'Признак': features,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False)
    
    st.dataframe(importance_df, use_container_width=True)

# --- ФУТЕР ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🐱 Cat Health Predictor | ML модель Random Forest | Данные о кошках</p>", 
    unsafe_allow_html=True
)
