# Проект: Продвинутое прогнозирование стоимости страховых выплат с MLOps

**Выпускная квалификационная работа**  
**Выполнил:** Фахрутдинов Марат Альбертович  
**Программа:** Цифровая кафедра - ИИ в экономике, управлении и финансах предприятия

## Описание проекта

Цель проекта - разработать продвинутую систему прогнозирования итоговой стоимости страхового возмещения (`UltimateIncurredClaimCost`) с использованием современных методов машинного обучения и инструментов MLOps.

Датасет: **Workers Compensation** из OpenML (`data_id=42876`). Тип задачи - регрессия.

В рамках продвинутой версии реализованы:

- Углубленный анализ данных с временными трендами
- Продвинутый feature engineering
- Сегментация страховых случаев и анализ по группам
- Сравнение базовых, бустинговых и ансамблевых моделей
- Оптимизация моделей с помощью Optuna
- Интеграция с ClearML для управления экспериментами
- Интерпретируемость предсказаний с помощью SHAP, LIME и PDP
- Анализ ошибок модели по ценовым сегментам

## Выполненные задачи

### **1. Сегментация данных и анализ по группам**

- **Описание:** Данные разделены на группы, после чего сравнивается качество общей модели и моделей для отдельных сегментов.
- **Реализовано:**
  - K-Means кластеризация с выбором числа кластеров и Silhouette Score
  - DBSCAN для поиска плотностных кластеров и выбросов
  - Доменная сегментация по начальной оценке дела
  - Сравнение общей модели и специализированных моделей по сегментам
  - Визуализация различий между сегментами

### **2. Детальный анализ датасета**

- **Описание:** Создана отдельная страница в Streamlit с углубленной аналитикой.
- **Реализовано:**
  - Временной анализ трендов стоимости возмещений по годам и месяцам
  - Анализ распределения целевой переменной и проверка нормальности
  - IQR-анализ выбросов
  - Корреляционный анализ с heatmap
  - Анализ категориальных признаков
  - Сравнение начальных оценок с итоговой стоимостью

### **3. Продвинутый feature engineering**

- **Описание:** Созданы дополнительные признаки для улучшения качества модели и анализа данных.
- **Реализовано:**
  - Извлечение временных признаков из дат несчастного случая и подачи заявки
  - Создание признаков взаимодействия, включая `Age_x_WeeklyPay`
  - Создание отношения `InitialCaseEstimate / WeeklyPay`
  - Бинарные признаки `HasDependents` и `IsFullTime`
  - Логарифмическое преобразование начальной оценки
  - PCA для анализа структуры признаков
  - t-SNE для визуализации многомерных данных
- **Примечание:** TF-IDF/Word2Vec для `ClaimDescription` в проекте не используется, потому что этот пункт в методичке приведен как один из возможных вариантов.

### **4. Выбор и обучение более мощных моделей**

- **Описание:** Базовые модели сравнены с бустинговыми и ансамблевыми алгоритмами.
- **Реализовано:**
  - Linear Regression как baseline
  - Random Forest и Gradient Boosting
  - XGBoost, LightGBM и CatBoost
  - Voting Regressor
  - Stacking Regressor с мета-моделью Ridge
  - Сохранение обученных моделей в `models/`

### **5. Оптимизация модели с Optuna**

- **Описание:** Добавлен автоматический подбор гиперпараметров и оценка устойчивости модели.
- **Реализовано:**
  - K-Fold Cross-Validation
  - Optuna для XGBoost и LightGBM
  - Поиск параметров `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
  - L1/L2-регуляризация через `reg_alpha` и `reg_lambda`
  - Early Stopping для LightGBM
  - Сравнение качества до и после оптимизации

### **6. Внедрение ClearML**

- **Описание:** Интеграция MLOps-инструментов для экспериментов, датасетов, моделей и деплоя.
- **Реализовано:**
  - Отслеживание экспериментов через ClearML Task и Logger
  - Логирование метрик, параметров и графиков
  - Версионирование датасета через ClearML Dataset
  - Регистрация модели через ClearML OutputModel
  - Загрузка модели по Model ID в Streamlit
  - Развертывание модели через ClearML Serving
  - REST API для предсказаний через Docker-контейнер

### **7. Интерпретируемость модели**

- **Описание:** Добавлены объяснения глобальных и локальных предсказаний модели.
- **Реализовано:**
  - SHAP summary plots
  - SHAP beeswarm plots
  - Waterfall plots для индивидуальных предсказаний
  - Таблица SHAP-вкладов по выбранному наблюдению
  - LIME для локального объяснения предсказаний
  - Partial Dependence Plots 1D и 2D

### **8. Анализ ошибок модели**

- **Описание:** Выполнен анализ случаев, где модель ошибается сильнее всего.
- **Реализовано:**
  - Расчет абсолютных и относительных ошибок
  - Топ-20 случаев с наибольшей ошибкой
  - Анализ MAE и MAPE по ценовым сегментам
  - Сравнение признаков у топ-500 ошибочных случаев и остальных наблюдений
  - Интерактивный фильтр ошибок в Streamlit

## Структура проекта

```text
workers-compensation-ml/
├── app.py
├── README.md
├── requirements.txt
├── pages/
│   ├── 1_Segmentation.py
│   ├── 2_Data_Analysis.py
│   ├── 3_Feature_Engineering.py
│   ├── 4_Models.py
│   ├── 5_Optimization.py
│   ├── 6_ClearML.py
│   ├── 7_SHAP.py
│   └── 8_Error_Analysis.py
├── src/
│   ├── clearml_integration.py
│   ├── config.py
│   ├── data_loader.py
│   ├── logger.py
│   ├── models.py
│   └── preprocessing.py
├── clearml_scripts/
│   ├── dataset_creation.py
│   ├── experiment1.py
│   └── experiment2.py
└── clearml_serving/
    ├── deploy_serving.sh
    └── preprocess.py
```

Папки `data/`, `models/`, `catboost_info/`, `.venv/` и локальные конфиги не должны попадать в git.

## Развертывание проекта с нуля

Требования:

- Python 3.10-3.12
- Git
- Docker Desktop, если нужно демонстрировать ClearML Serving
- Доступ в интернет для первой загрузки датасета из OpenML

Команды для Windows PowerShell:

```powershell
cd $env:USERPROFILE\Downloads
git clone https://github.com/bnm76543210/workers-compensation-advanced-mlops.git
cd workers-compensation-advanced-mlops

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

Приложение откроется по адресу:

```text
http://localhost:8501
```

При первом запуске датасет загружается из OpenML и сохраняется в `data/workers_comp_raw.csv`.

## Проверка функций приложения

1. Открыть главную страницу Streamlit.
2. Проверить страницу `1_Segmentation`: K-Means, DBSCAN, доменная сегментация.
3. Проверить страницу `2_Data_Analysis`: распределения, временные графики, корреляции, InitialEstimate vs Cost.
4. Проверить страницу `3_Feature_Engineering`: новые признаки, IQR, PCA, t-SNE.
5. Проверить страницу `4_Models`: базовые модели и ансамбли.
6. Проверить страницу `5_Optimization`: K-Fold, Optuna, Early Stopping.
7. Проверить страницу `6_ClearML`: эксперименты, загрузка модели, inference.
8. Проверить страницу `7_SHAP`: SHAP, LIME, PDP.
9. Проверить страницу `8_Error_Analysis`: таблица ошибок и интерактивный фильтр.

Тяжелые шаги вроде CatBoost, ансамблей, Optuna, SHAP и t-SNE могут занимать несколько минут.

## Настройка ClearML

Секреты ClearML нельзя записывать в README, `.env`, код или коммиты. Настройте их локально:

```powershell
clearml-init
```

После настройки должен появиться файл:

```text
%USERPROFILE%\clearml.conf
```

Запуск ClearML-скриптов:

```powershell
python clearml_scripts/dataset_creation.py
python clearml_scripts/experiment1.py
python clearml_scripts/experiment2.py
```

## ClearML Serving и Docker

Проверить Docker:

```powershell
docker ps
```

Скачать образ, если его еще нет:

```powershell
docker pull allegroai/clearml-serving-inference:1.3.2
```

Пример запуска serving-контейнера:

```powershell
$serviceId = "f4898d5556a942c19f571692a97737a6"
$image = "allegroai/clearml-serving-inference:1.3.2"
$conf = "$env:USERPROFILE\clearml.conf"

docker rm -f clearml_pm_serving
docker run -d `
  --name clearml_pm_serving `
  -v "${conf}:/root/clearml.conf:ro" `
  -p 8080:8080 `
  -e CLEARML_SERVING_TASK_ID=$serviceId `
  -e CLEARML_SERVING_POLL_FREQ=5 `
  -e CLEARML_EXTRA_PYTHON_PACKAGES="scikit-learn==1.6.1 xgboost==3.2.0 pandas==2.2.3" `
  $image
```

Проверка логов:

```powershell
docker logs --tail 100 clearml_pm_serving
```

Пример запроса из Windows PowerShell:

```powershell
$body = @{
  Age = 35
  WeeklyPay = 500
  InitialCaseEstimate = 5000
  HoursWorkedPerWeek = 40
  DaysWorkedPerWeek = 5
  Gender = "M"
  MaritalStatus = "S"
  DependentChildren = 0
  DependentsOther = 0
  PartTimeFullTime = "F"
  Accident_Year = 2010
  Accident_Month = 6
  Accident_DayOfWeek = 2
  Reported_Year = 2010
  Reported_Month = 7
  Reported_DayOfWeek = 1
  ReportDelay_Days = 30
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8080/serve/workers_compensation" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Технический стек

| Инструмент | Назначение |
|---|---|
| Streamlit | Многостраничное приложение |
| pandas, numpy | Обработка данных |
| scikit-learn | Модели, метрики, CV, PCA, t-SNE |
| XGBoost, LightGBM, CatBoost | Градиентный бустинг |
| Optuna | Подбор гиперпараметров |
| SHAP, LIME | Интерпретируемость |
| ClearML | MLOps, эксперименты, датасеты, модели |
| Docker | ClearML Serving |
| Plotly, Matplotlib, Seaborn | Визуализация |

## Отладка

Подробные логи управляются флагом `DEBUG`:

```python
# src/config.py
DEBUG = True
```

Быстрая проверка синтаксиса:

```powershell
python -m compileall app.py pages src clearml_scripts clearml_serving
```
