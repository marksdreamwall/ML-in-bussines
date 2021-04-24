import flask
import pandas as pd
import dill
import json

# Project path
PATH = '/home/dreamwall/PycharmProjects/Air_sat_project/'

# Load model
with open(PATH + 'satisfaction2.dill', 'rb') as m:
    model = dill.load(m)

# Load df
X_test = pd.read_csv(PATH + 'X_test.csv')
y_test = pd.read_csv(PATH + 'y_test.csv')

test_columns = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
                'Flight Distance', 'Inflight wifi service',
                'Departure/Arrival time convenient', 'Ease of Online booking',
                'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                'Inflight entertainment', 'On-board service', 'Leg room service',
                'Baggage handling', 'Checkin service', 'Inflight service',
                'Cleanliness', 'Departure Delay in Minutes',
                'Arrival Delay in Minutes']

# def req_check(req, columns):
#     if request_json[req] in columns:
#         req.lower()


app = flask.Flask(__name__)

app.debug = True


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = {'success': False}

    if flask.request.method == "POST":
        gender, customer_type, age, type_of_travel, clas, \
        flight_distance, inflight_wifi_service, \
        departure_arrival_time_convenient, ease_of_online_booking, \
        gate_location, food_and_drink, online_boarding, seat_comfort, \
        inflight_entertainment, on_board_service, leg_room_service, \
        baggage_handling, checkin_service, inflight_service, \
        cleanliness, departure_delay_in_minutes, \
        arrival_delay_in_minutes = "", "", "", "", "", "", "", "", \
                                   "", "", "", "", "", "", "", "", "", \
                                   "", "", "", "", ""

        request_json = flask.request.get_json()
        if request_json['Gender']:
            gender = request_json['Gender']

        if request_json['Customer Type']:
            customer_type = request_json['Customer Type']

        if request_json['Age']:
            age = request_json['Age']

        if request_json['Type of Travel']:
            type_of_travel = request_json['Type of Travel']

        if request_json['Class']:
            clas = request_json['Class']

        if request_json['Flight Distance']:
            flight_distance = request_json['Flight Distance']

        if request_json['Inflight wifi service']:
            inflight_wifi_service = request_json['Inflight wifi service']

        if request_json['Departure/Arrival time convenient']:
            departure_arrival_time_convenient = request_json['Departure/Arrival time convenient']

        if request_json['Ease of Online booking']:
            ease_of_online_booking = request_json['Ease of Online booking']

        if request_json['Gate location']:
            gate_location = request_json['Gate location']

        if request_json['Food and drink']:
            food_and_drink = request_json['Food and drink']

        if request_json['Online boarding']:
            online_boarding = request_json['Online boarding']

        if request_json['Seat comfort']:
            seat_comfort = request_json['Seat comfort']

        if request_json['Inflight entertainment']:
            inflight_entertainment = request_json['Inflight entertainment']

        if request_json['On-board service']:
            on_board_service = request_json['On-board service']

        if request_json['Leg room service']:
            leg_room_service = request_json['Leg room service']

        if request_json['Baggage handling']:
            baggage_handling = request_json['Baggage handling']

        if request_json['Checkin service']:
            checkin_service = request_json['Checkin service']

        if request_json['Inflight service']:
            inflight_service = request_json['Inflight service']

        if request_json['Cleanliness']:
            cleanliness = request_json['Cleanliness']

        if request_json['Departure Delay in Minutes']:
            departure_delay_in_minutes = request_json['Departure Delay in Minutes']

        if request_json['Arrival Delay in Minutes']:
            arrival_delay_in_minutes = request_json['Arrival Delay in Minutes']


        preds = model.predict_proba(pd.DataFrame({'Gender': [gender],
                                                  'Customer Type': [customer_type],
                                                  'Age': [age],
                                                  'Type of Travel': [type_of_travel],
                                                  'Class': [clas],
                                                  'Flight Distance': [flight_distance],
                                                  'Inflight wifi service': [inflight_wifi_service],
                                                  'Departure/Arrival time convenient': [departure_arrival_time_convenient],
                                                  'Ease of Online booking': [ease_of_online_booking],
                                                  'Gate location': [gate_location],
                                                  'Food and drink': [food_and_drink],
                                                  'Online boarding': [online_boarding],
                                                  'Seat comfort': [seat_comfort],
                                                  'Inflight entertainment': [inflight_entertainment],
                                                  'On-board service': [on_board_service],
                                                  'Leg room service': [leg_room_service],
                                                  'Baggage handling': [baggage_handling],
                                                  'Checkin service': [checkin_service],
                                                  'Inflight service': [inflight_service],
                                                  'Cleanliness': [cleanliness],
                                                  'Departure Delay in Minutes': [departure_delay_in_minutes],
                                                  'Arrival Delay in Minutes': [arrival_delay_in_minutes]}))

        data["predictions"] = preds[:, 1][0]
        data["success"] = True
        # print('Все хорошо :)')

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
