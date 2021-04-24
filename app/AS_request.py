import pandas as pd
import json
import urllib.request
import requests
from sklearn.metrics import roc_auc_score

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

data = X_test[X_test.index == 8].to_dict('records')[0]


def send_json(x):
    # В json 0, превращается в ''. Исправляю
    for i in x.keys():
        if x[i] == 0:
            x[i] = 0.000000000001

    gender, customer_type, age, type_of_travel, clas, \
    flight_distance, inflight_wifi_service, \
    departure_arrival_time_convenient, ease_of_online_booking, \
    gate_location, food_and_drink, online_boarding, seat_comfort, \
    inflight_entertainment, on_board_service, leg_room_service, \
    baggage_handling, checkin_service, inflight_service, \
    cleanliness, departure_delay_in_minutes, \
    arrival_delay_in_minutes = x.values()


    body = {'Gender': gender,
            'Customer Type': customer_type,
            'Age': age,
            'Type of Travel': type_of_travel,
            'Class': clas,
            'Flight Distance': flight_distance,
            'Inflight wifi service': inflight_wifi_service,
            'Departure/Arrival time convenient': departure_arrival_time_convenient,
            'Ease of Online booking': ease_of_online_booking,
            'Gate location': gate_location,
            'Food and drink': food_and_drink,
            'Online boarding': online_boarding,
            'Seat comfort': seat_comfort,
            'Inflight entertainment': inflight_entertainment,
            'On-board service': on_board_service,
            'Leg room service': leg_room_service,
            'Baggage handling': baggage_handling,
            'Checkin service': checkin_service,
            'Inflight service': inflight_service,
            'Cleanliness': cleanliness,
            'Departure Delay in Minutes': departure_delay_in_minutes,
            'Arrival Delay in Minutes': arrival_delay_in_minutes
            }

    myurl = 'http://127.0.0.1:5000/predict'
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(myurl, json=body, headers=headers)
    return response.json()['predictions']


if __name__ == '__main__':
    response = send_json(data)
    print('Одно предсказание', response)
    n = 500
    predictions = []
    for item in range(n):
        data = X_test.iloc[item:].to_dict('records')[0]
        response = send_json(data)
        predictions.append(response)
    # print(predictions)
    print(f'ROC_AUC_SCORE для {n} объектов = ', roc_auc_score(y_score=predictions, y_true=y_test.iloc[:n]))


