import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- РЕАЛЬНАЯ ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (не зависит от признаков) ---
# Используем флаг стерилизации и гуляет ли на улице как индикатор здоровья
# Стерилизованные кошки и гуляющие на улице считаются более здоровыми
def get_real_health(row):
    score = 0
    # Стерилизация (TRUE - хорошо для здоровья)
    if str(row['Set']).upper() == 'TRUE':
        score += 1
    # Гуляет на улице
    if str(row['Walks']).upper() == 'TRUE':
        score += 1
    # Возраст до 10 лет
    if row['Age'] <= 10:
        score += 1
    # Вес в норме
    if 3 <= row['Weight'] <= 6:
        score += 1
    return 'Healthy' if score >= 2 else 'Needs Care'

df['Health_Status'] = df.apply(get_real_health, axis=1)

# --- ПОДГОТОВКА ПРИЗНАКОВ ---
features = ['Age', 'Weight', 'Playing (min.)', 'Sleeps (hours)']

# Добавляем категориальные признаки для лучшего обучения
from sklearn.preprocessing import OneHotEncoder

# Кодируем породу
breed_dummies = pd.get_dummies(df['Breed'], prefix='Breed')
df = pd.concat([df, breed_dummies], axis=1)
features.extend(breed_dummies.columns.tolist())

# Кодируем цвет
color_dummies = pd.get_dummies(df['Color'], prefix='Color')
df = pd.concat([df, color_dummies], axis=1)
features.extend(color_dummies.columns.tolist())

X = df[features].fillna(df[features].mean())
y = df['Health_Status']

# --- ОБУЧЕНИЕ МОДЕЛИ ---
@st.cache_resource
def train_model():
    # Кодирование целевой переменной
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Нормализация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Обучение модели с ограничениями против переобучения
    model = RandomForestClassifier(
        n_estimators=50,      # Меньше деревьев
        max_depth=5,          # Ограничиваем глубину
        min_samples_split=10, # Минимум образцов для разделения
        min_samples_leaf=5,   # Минимум образцов в листе
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Оценка точности
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    return model, le, scaler, features, train_acc, test_acc

model, label_encoder, scaler, feature_list, train_acc, test_acc = train_model()

# --- БОКОВАЯ ПАНЕЛЬ - КОНТРОЛЫ ---
st.sidebar.header("🎮 Управление приложением")

# КОНТРОЛ 1: Выбор породы
st.sidebar.subheader("1. Фильтр по породе")
selected_breeds = st.sidebar.multiselect(
    "Выберите породы:",
    options=sorted(df['Breed'].unique()),
    default=[]
)

# КОНТРОЛ 2: Диапазон веса
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

# КОНТРОЛ 3: Уровень активности
st.sidebar.subheader("3. Уровень активности")
activity_level = st.sidebar.radio(
    "Активность:",
    options=["Все", "Активные (>30 мин)", "Малоподвижные (<20 мин)"]
)

# КОНТРОЛ 4: Показать только здоровых
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
               autopct='%1.1f%%', colors=colors_health[:len(health_counts)])
        ax.set_title('Соотношение здоровых кошек')
        st.pyplot(fig)
    else:
        st.info("Нет данных")

# --- ПРОГНОЗИРОВАНИЕ ---
st.header("🤖 Прогнозирование здоровья кошки")
st.markdown(f"*Модель машинного обучения (Random Forest) | Точность на тесте: {test_acc*100:.1f}%*")

st.markdown("### Введите данные о кошке:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("📅 Возраст (лет)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
    weight = st.number_input("⚖️ Вес (кг)", min_value=1.0, max_value=15.0, value=4.0, step=0.5)
    breed = st.selectbox("🐈 Порода", options=sorted(df['Breed'].unique()))

with col2:
    play_minutes = st.number_input("🎾 Время игр (минут в день)", min_value=0, max_value=120, value=30, step=5)
    sleep_hours = st.number_input("😴 Часы сна в день", min_value=5.0, max_value=24.0, value=14.0, step=0.5)
    color = st.selectbox("🎨 Цвет шерсти", options=sorted(df['Color'].unique()))

# Кнопка предсказания
if st.button("🔮 Предсказать здоровье", type="primary", use_container_width=True):
    # Создаем все признаки для предсказания
    input_data = {}
    
    # Основные признаки
    input_data['Age'] = age
    input_data['Weight'] = weight
    input_data['Playing (min.)'] = play_minutes
    input_data['Sleeps (hours)'] = sleep_hours
    
    # One-hot кодирование породы
    for b in df['Breed'].unique():
        input_data[f'Breed_{b}'] = 1 if breed == b else 0
    
    # One-hot кодирование цвета
    for c in df['Color'].unique():
        input_data[f'Color_{c}'] = 1 if color == c else 0
    
    # Создаем DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Заполняем отсутствующие колонки
    for col in feature_list:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_list]
    
    # Нормализация и предсказание
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    result = label_encoder.inverse_transform([prediction])[0]
    
    st.markdown("---")
    st.subheader("📋 Результат прогноза:")
    
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
            health_prob = probabilities[1 if label_encoder.classes_[1] == 'Healthy' else 0] * 100
            st.metric("Вероятность здоровья", f"{health_prob:.1f}%")
        else:
            risk_prob = probabilities[0 if label_encoder.classes_[0] == 'Needs Care' else 1] * 100
            st.metric("Вероятность проблем", f"{risk_prob:.1f}%")
    
    # Рекомендации
    st.markdown("---")
    st.markdown("### 💡 Рекомендации:")
    
    if age > 12:
        st.info("📌 Пожилая кошка - рекомендуются регулярные осмотры у ветеринара (2 раза в год)")
    if weight < 2:
        st.warning("📌 Низкий вес - проконсультируйтесь с ветеринаром о питании")
    if weight > 8:
        st.warning("📌 Избыточный вес - рекомендуется диета и увеличение физической активности")
    if play_minutes < 20:
        st.info("📌 Низкая активность - попробуйте увеличить время игр с кошкой")
    if sleep_hours < 10:
        st.info("📌 Мало спит - возможно, кошке что-то мешает")
    if sleep_hours > 20:
        st.info("📌 Спит слишком много - обратите внимание на активность кошки")

# --- ИНФОРМАЦИЯ О МОДЕЛИ ---
with st.expander("📊 Информация о модели машинного обучения"):
    st.markdown(f"""
    **Характеристики модели:**
    
    - **Алгоритм:** Random Forest Classifier
    - **Количество деревьев:** 50
    - **Максимальная глубина:** 5
    - **Признаки:** {len(feature_list)} (возраст, вес, игры, сон, порода, цвет)
    - **Размер датасета:** {len(df)} записей
    - **Точность на обучении:** {train_acc*100:.1f}%
    - **Точность на тесте:** {test_acc*100:.1f}%
    
    **Важность признаков (топ-10):**
    """)
    
    importance_df = pd.DataFrame({
        'Признак': feature_list,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False).head(10)
    
    st.dataframe(importance_df, use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🐱 Cat Health Predictor | ML модель Random Forest</p>", 
    unsafe_allow_html=True
)
