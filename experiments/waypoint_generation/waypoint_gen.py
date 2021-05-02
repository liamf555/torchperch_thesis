#!/usr/bin/env python3

import math
import pygeodesy.ecef as geo
import gmplot
import webbrowser

apikey = 'AIzaSyDuLUX3eqIJAqVt97KaNPg8CPeR0ZHwAwA'

FIX_LAT = 51.4232   #increase to shift up
FIX_LONG = -2.6712 #decrease to shift left
BEARING = 260
RIGHT_TURN=False


def deg2rad(deg):
    return deg / 180.0 * math.pi

def generate_turn(centre, radius, initial_bearing, facets, right_turn=True):
    if right_turn:
        initial = deg2rad(initial_bearing)
        increment = math.pi / facets
    else:
        initial = math.pi + deg2rad(initial_bearing)
        increment = -math.pi / facets

    points = []
    for i in range(1,facets+1):
        x = -radius * math.cos(initial + i*increment)
        y = radius * math.sin(initial + i*increment)
        points.append( [x + centre[0], y + centre[1]] )
    
    return points

def generate_racetrack(
        fix_position,
        inbound_bearing,
        turn_radius=30,
        leg_length=150,
        ml_fraction=0.2,
        right_turn=True,
        turn_facets=4):
    
    cartesian_points = []
    
    ib_bearing_rad = deg2rad(inbound_bearing)

    v_ib = [ math.sin(ib_bearing_rad), math.cos(ib_bearing_rad) ]
    v_ob = [ -v_ib[0], -v_ib[1] ]

    v_r  = [ math.cos(ib_bearing_rad), -math.sin(ib_bearing_rad) ]
    if not right_turn:
        v_r = [ -v_r[0], -v_r[1] ]

    # Initial fix
    cartesian_points.append( [0,0] )

    # Outbound turn
    outbound_turn_centre = [
        turn_radius * v_r[0],
        turn_radius * v_r[1]
        ]

    cartesian_points += generate_turn(
        outbound_turn_centre, turn_radius, inbound_bearing, turn_facets, right_turn
        )

    # Outbound leg
    outbound_leg_end = [
        2 * turn_radius * v_r[0] + leg_length * v_ob[0],
        2 * turn_radius * v_r[1] + leg_length * v_ob[1]
        ]
    cartesian_points.append(outbound_leg_end)

     # Inbound turn
    inbound_turn_centre = [
        turn_radius * v_r[0] + leg_length * v_ob[0],
        turn_radius * v_r[1] + leg_length * v_ob[1]
        ]

    cartesian_points += generate_turn(
        inbound_turn_centre, turn_radius, (inbound_bearing + 180) % 360, turn_facets, right_turn
        )

    # Inbound ML point
    inbound_ml_point = [
        (1 - ml_fraction) * leg_length * v_ob[0],
        (1 - ml_fraction) * leg_length * v_ob[1]
        ]
    cartesian_points.append(inbound_ml_point)

    # Convert to lat/lon
    ecef_frame = geo.EcefCartesian(fix_position[0], fix_position[1])
    
    latlon_points = []
    for point in cartesian_points:
        latlon_tuple = ecef_frame.reverse( point[0], point[1], 0 )
        latlon_points.append( [ latlon_tuple[3], latlon_tuple[4] ] )
    
    return latlon_points

def generate_waypoints(fix_position, inbound_bearing, *,
        altitude=50,
        turn_radius=30,
        leg_length=150,
        ml_fraction=0.4,
        right_turn=True,
        turn_facets=4):
 
    racetrack_points = generate_racetrack(
        fix_position,
        inbound_bearing,
        turn_radius=turn_radius,
        leg_length=leg_length,
        ml_fraction=ml_fraction,
        right_turn=right_turn,
        turn_facets=turn_facets
        )
    
    return racetrack_points
    # for point in racetrack_points:
        # print("{},{},{}".format(point[0],point[1],50))

if __name__ == "__main__":
    import plan_gen


    racetrack_points = generate_waypoints((FIX_LAT,FIX_LONG),BEARING,turn_radius=50,right_turn=RIGHT_TURN,turn_facets=3)
    home_point = racetrack_points[-2] + [ 35.0 ]
       
   
    gmap = gmplot.GoogleMapPlotter(51.423098, -2.670601, 18, map_type='satellite', apikey=apikey)
    
    gmap.polygon(*zip(*racetrack_points), color='cornflowerblue', edge_width=10)

    # Draw the map to an HTML file:
    gmap.draw('map.html')
    
    webbrowser.open('map.html')
    
    jump_id = 1
    mission_items = [ plan_gen.get_takeoff_item(jump_id) ]
    jump_id+=1

    for point in racetrack_points:
        mission_items.append( plan_gen.get_waypoint_item(point[0],point[1],50,jump_id) )
        jump_id+=1
    
    mission_items.append( plan_gen.get_mlagent_item(jump_id) )
    jump_id+=1

    mission_items.append( plan_gen.get_dojump_item(jump_id,2,-1) )

    print(plan_gen.get_plan_string(mission_items,home_point))
