# ML-in-bussines
Курсовой проект курса GeekBrains "Машинное обучение в бизнесе"

Dataset - https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

Задача: Предсказание удовлетворенности пассажиров перелетом. 

ML: sklearn, pandas, numpy. API: flask
Модель - CatBoostClassifier



Файлы: 1. passenger_satisfaction_train.ipynb - анализ данных, подбор моделей с использование GridSearchCV и обучение итоговой модели.
2. passenger_satisfaction_test.ipynb - проверка работы модели на тестовых данных
3. requirements.txt - необходимые библиотеки
4. app/models/satisfaction2.dill - сохраненная модель
5. app/AS_run_server.py - запуск Flask сервера.
6. app/AS_request.py - запрос к серверу

Коментарии: Так же добавлены X_test.csv и y_test.csv, т.к. запрос файла AS_requests.py составляется с помощью X_test. 2 варианта - запрос 1 строки и запрос 500 строк.



