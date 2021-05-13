from requests.structures import CaseInsensitiveDict
import requests


def return_detected_plate_details(img_name, plate_num, vehicle_side, ip_address, gate_id, img_locations):
    url = "http://localhost:7004/vtms/api/saveLicensePlateInfo"

    #  get 360 locations
    if vehicle_side == "FRONT":
        back_left = img_locations['BACK_LEFT']
        back_right = img_locations['BACK_RIGHT']
        front_left = img_locations['FRONT_LEFT']
        front_right = img_locations['FRONT_RIGHT']

        data = {
            "licensePlatePath": img_name,
            "licensePlateNum": plate_num,
            "vehicleSide": vehicle_side,
            "gateId": gate_id,
            "ipAddress": ip_address,
            "backLeftImgPath": back_left,
            "backRightImgPath": back_right,
            "frontLeftImgPath": front_left,
            "frontRightImgPath": front_right
        }
    else:
        data = {
            "licensePlatePath": img_name,
            "licensePlateNum": plate_num,
            "vehicleSide": vehicle_side,
            "gateId": gate_id,
            "ipAddress": ip_address
        }

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    # create JSON

    requests.post(url, headers=headers, json=data)


def return_detected_axle_details(axle_count, ip_address, gate_id):
    # create JSON
    url = "http://localhost:7004/vtms/api/saveAxleInfo"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    # create JSON

    data = {"axleCount": axle_count,
            "gateId": gate_id,
            "ipAddress": ip_address}

    requests.post(url, headers=headers, json=data)
