#!/usr/bin/env python

import json

def get_plan_string(mission_items,home_point):
    return json.dumps({
        "fileType": "Plan",
        "geoFence": {
            "circles": [],
            "polygons": [],
            "version": 2
            },
        "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": 20,
            "firmwareType": 3,
            "items": mission_items,
            "plannedHomePosition": [
                home_point[0],
                home_point[1],
                home_point[2]
                ],
            "vehicleType": 2,
            "version": 2
            },
        "rallyPoints": {
            "points": [],
            "version": 2
            },
        "version": 1
        }, indent=4)

def get_takeoff_item(jump_id):
    return {
        "autoContinue": True,
        "command": 22,
        "doJumpId": jump_id,
        "frame": 3,
        "params": [
            15,
            0,
            0,
            None,
            0,
            0,
            50
            ],
        "type": "SimpleItem"
        }

def get_waypoint_item(lat,lon,alt,jump_id):
    return {
        "AMSLAltAboveTerrain": alt,
        "Altitude": alt,
        "AltitudeMode": 1,
        "autoContinue": True,
        "command": 16,
        "doJumpId": jump_id,
        "frame": 3,
        "params": [
            0,
            0,
            0,
            None,
            lat,
            lon,
            alt
            ],
        "type": "SimpleItem"
        }

def get_mlagent_item(jump_id):
    return {
        "autoContinue": True,
        "command": 87,
        "doJumpId": jump_id,
        "frame": 2,
        "params": [
            0,
            0,
            0,
            0,
            0,
            0,
            0
            ],
        "type": "SimpleItem"
        }

def get_dojump_item(jump_id,target,repeat):
    return {
        "autoContinue": True,
        "command": 177,
        "doJumpId": jump_id,
        "frame": 2,
        "params": [
            target,
            repeat,
            0,
            0,
            0,
            0,
            0
            ],
        "type": "SimpleItem"
        }