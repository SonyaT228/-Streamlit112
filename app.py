import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Cat Health Predictor", layout="wide")
st.title("🐱 Cat Health Predictor")
st.markdown("### Анализ и прогнозирование здоровья кошек")
st.markdown("---")

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv('cat_2.csv', sep=';')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Создаем модель
@st.cache_resource
def train_model():
    def is_healthy(row):
        problems = 0
        if row['Age'] > 15:
            problems += 1
        if row['Weight'] < 2 or row['Weight'] > 8:
            problems += 1
        if row['Playing (min.)'] < 20:
            problems += 1
        if row['Sleeps (hours)'] < 10 or row['Sleeps (hours)'] > 20:
            problems += 1
        return 'Yes' if problems < 2 else 'No'
    
    df['Healthy'] = df.apply(is_healthy, axis=1)
    
    features = ['Age', 'Weight', 'Playing (min.)', 'Sleeps (hours)']
    X = df[features].fillna(df[features].mean())
    y = df['Healthy']
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_enc)
    
    return model, le, features

model, le, features = train_model()

# --- БОКОВАЯ ПАНЕЛЬ С ФИЛЬТРАМИ ---
st.sidebar.header("🔧 Фильтры")

# Фильтр по породе
breeds = st.sidebar.multiselect(
    "Порода:",
    options=sorted(df['Breed'].unique()),
    default=[]
)

# Фильтр по весу
min_weight = float(df['Weight'].min())
max_weight = float(df['Weight'].max())
weight_range = st.sidebar.slider(
    "Вес (кг):",
    min_value=min_weight,
    max_value=max_weight,
    value=(min_weight, max_weight)
)

# Фильтр по цвету
colors = st.sidebar.multiselect(
    "Цвет шерсти:",
    options=sorted(df['Color'].unique()),
    default=[]
)

# Фильтр по здоровью
only_healthy = st.sidebar.checkbox("Показать только здоровых кошек")

# Применяем фильтры
filtered_df = df.copy()
if breeds:
    filtered_df = filtered_df[filtered_df['Breed'].isin(breeds)]
filtered_df = filtered_df[(filtered_df['Weight'] >= weight_range[0]) & 
                          (filtered_df['Weight'] <= weight_range[1])]
if colors:
    filtered_df = filtered_df[filtered_df['Color'].isin(colors)]
if only_healthy:
    filtered_df = filtered_df[filtered_df['Healthy'] == 'Yes']

# --- ОСНОВНАЯ ОБЛАСТЬ ---
st.header("📊 Данные о кошках")
st.info(f"Показано записей: {len(filtered_df)} из {len(df)}")
st.dataframe(filtered_df, use_container_width=True)

# --- ВИЗУАЛИЗАЦИЯ ---
st.header("📈 Визуализация данных")

# Выбор типа графика
chart_type = st.selectbox(
    "Выберите тип графика:",
    ["Гистограмма пород", "Средний вес по породам", 
     "Распределение по возрасту", "Распределение по весу",
     "Активность по породам", "Сон по породам",
     "Соотношение здоровых/нездоровых", "Сравнение по странам"]
)

col1, col2 = st.columns(2)

with col1:
    if chart_type == "Гистограмма пород":
        st.subheader("Количество кошек по породам")
        breed_counts = filtered_df['Breed'].value_counts()
        st.bar_chart(breed_counts)
        
    elif chart_type == "Средний вес по породам":
        st.subheader("Средний вес по породам (кг)")
        avg_weight = filtered_df.groupby('Breed')['Weight'].mean().sort_values(ascending=False)
        st.bar_chart(avg_weight)
        
    elif chart_type == "Распределение по возрасту":
        st.subheader("Распределение возраста кошек")
        fig, ax = plt.subplots(figsize=(10, 5))
        filtered_df['Age'].hist(bins=20, ax=ax, color='#FF6B6B', edgecolor='black')
        ax.set_xlabel('Возраст (лет)')
        ax.set_ylabel('Количество')
        ax.set_title('Распределение возраста')
        st.pyplot(fig)
        
    elif chart_type == "Распределение по весу":
        st.subheader("Распределение веса кошек")
        fig, ax = plt.subplots(figsize=(10, 5))
        filtered_df['Weight'].hist(bins=20, ax=ax, color='#4ECDC4', edgecolor='black')
        ax.set_xlabel('Вес (кг)')
        ax.set_ylabel('Количество')
        ax.set_title('Распределение веса')
        st.pyplot(fig)
        
    elif chart_type == "Активность по породам":
        st.subheader("Среднее время игр по породам (мин/день)")
        avg_play = filtered_df.groupby('Breed')['Playing (min.)'].mean().sort_values(ascending=False)
        st.bar_chart(avg_play)
        
    elif chart_type == "Сон по породам":
        st.subheader("Средняя продолжительность сна по породам (часы)")
        avg_sleep = filtered_df.groupby('Breed')['Sleeps (hours)'].mean().sort_values(ascending=False)
        st.bar_chart(avg_sleep)
        
    elif chart_type == "Соотношение здоровых/нездоровых":
        st.subheader("Здоровье кошек")
        health_counts = filtered_df['Healthy'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors_health = ['#2ECC71', '#E74C3C']
        ax.pie(health_counts.values, labels=['Здоровые', 'Требуют внимания'], 
               autopct='%1.1f%%', colors=colors_health, startangle=90)
        ax.set_title('Соотношение здоровых и нездоровых кошек')
        st.pyplot(fig)
        
    else:  # Сравнение по странам
        st.subheader("Количество кошек по странам")
        country_counts = filtered_df['Country'].value_counts().head(10)
        st.bar_chart(country_counts)

with col2:
    # Дополнительная статистика
    st.subheader("📊 Статистика по выбранным данным")
    
    avg_age = filtered_df['Age'].mean()
    avg_weight_val = filtered_df['Weight'].mean()
    avg_play_val = filtered_df['Playing (min.)'].mean()
    avg_sleep_val = filtered_df['Sleeps (hours)'].mean()
    healthy_percent = (filtered_df['Healthy'] == 'Yes').mean() * 100
    
    st.metric("Средний возраст", f"{avg_age:.1f} лет")
    st.metric("Средний вес", f"{avg_weight_val:.1f} кг")
    st.metric("Среднее время игр", f"{avg_play_val:.0f} мин/день")
    st.metric("Средний сон", f"{avg_sleep_val:.1f} час/день")
    st.metric("Здоровых кошек", f"{healthy_percent:.1f}%")

# --- ПРОГНОЗИРОВАНИЕ ---
st.markdown("---")
st.header("🤖 Прогнозирование здоровья кошки")
st.markdown("Введите данные о вашей кошке:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Возраст (лет)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
    weight = st.number_input("Вес (кг)", min_value=1.0, max_value=15.0, value=4.0, step=0.5)

with col2:
    play = st.number_input("Время игр (минут в день)", min_value=0, max_value=120, value=30, step=5)
    sleep = st.number_input("Часы сна в день", min_value=5.0, max_value=24.0, value=14.0, step=0.5)

if st.button("🔮 Предсказать здоровье", type="primary", use_container_width=True):
    input_data = pd.DataFrame([[age, weight, play, sleep]], columns=features)
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    result = le.inverse_transform([pred])[0]
    
    st.subheader("Результат прогноза:")
    
    col_r, col_p = st.columns(2)
    with col_r:
        if result == "Yes":
            st.success("✅ Кошка ЗДОРОВА!")
            st.balloons()
        else:
            st.error("⚠️ Кошка требует ВНИМАНИЯ!")
    
    with col_p:
        if result == "Yes":
            st.metric("Вероятность здоровья", f"{prob[1]*100:.1f}%")
        else:
            st.metric("Вероятность проблем", f"{prob[0]*100:.1f}%")

# --- ДОПОЛНИТЕЛЬНЫЕ СОВЕТЫ ---
with st.expander("💡 Рекомендации по здоровью кошек"):
    st.markdown("""
    **Как сохранить кошку здоровой:**
    
    - 🏥 **Регулярные осмотры у ветеринара** (раз в год, для пожилых кошек - 2 раза)
    - 🍗 **Правильное питание** (сбалансированный корм, чистая вода)
    - 🎾 **Активные игры** (минимум 20-30 минут в день)
    - 💤 **Комфортный сон** (тихое место, мягкая лежанка)
    - 🪥 **Гигиена** (чистка зубов, расчесывание)
    - 😺 **Внимание и ласка** (стресс вреден для здоровья)
    """)

# --- ФУТЕР ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🐱 Cat Health Predictor | Данные о кошках | Предсказание на основе ML</p>", 
    unsafe_allow_html=True
)
