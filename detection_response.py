from requests.structures import CaseInsensitiveDict
import requests


def return_detected_plate_details(img_name, plate_num, vehicle_side, ip_address, gate_id):
    url = "http://localhost:7004/vtms/api/saveLicensePlateInfo"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    # create JSON

    data = {"licensePlatePath": img_name,
            "licensePlateNum": plate_num,
            "vehicleSide": vehicle_side,
            "gateId": gate_id,
            "ipAddress": ip_address}

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
