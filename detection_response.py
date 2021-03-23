from requests.structures import CaseInsensitiveDict
import requests


def return_detected_plate_details(path, plate_num):
    url = "http://localhost:7002/vtms/api/saveLicensePlateInfo"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    # create JSON

    data = {"licensePlatePath": path, "licensePlateNum": plate_num}

    requests.post(url, headers=headers, json=data)



def return_detected_axle_details(axle_count):
    # create JSON
    url = "http://localhost:7002/vtms/api/saveLicensePlateInfo"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    # create JSON

    data = '{"axle_count" :' + axle_count + '}'

    requests.post(url, headers=headers, data=data)