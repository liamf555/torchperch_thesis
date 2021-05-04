import bixler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

params = {"turbulence": "none", "noise": 0.0, "wind_mode": "steady", "wind_params": [0, 0, 0]}

bixler = bixler.Bixler(params)
bixler.wind_sim.update()
desired = 13
initial_state = np.array([[0,0,0, 0,0, 0, (13+params["wind_params"][0]),0,0, 0,0,0, 0,0,0]], dtype="float64")


bixler.throttle=0

bixler.set_state(initial_state)

airspeeds = []
throttles=[]

kp = 10
max_throttle = (bixler.mass*9.81)/2

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
bixler.airspeed=0

print("x,y,z,Roll,Pitch,Yaw,u,v,w,Roll rate,Theta dot,Yaw rate,Sweep,Elevator,Tip")
time = 0
for i in np.arange(1000):
    print(f"time: {time}, x: {bixler.position_e[0,0]}, z: {bixler.position_e[2,0]}, pitch: {np.rad2deg(bixler.orientation_e[1,0])}, u: {bixler.velocity_b[0,0]}, w: {bixler.velocity_b[2,0]}, throttle: {bixler.throttle}")
    print(f"airspeed: {bixler.airspeed}")
    print(bixler.D) # stability frame
    print(bixler.drag)# body frame
    print(bixler.wind)
    for i in range(1,10):


        # bixler.throttle = np.clip(-bixler.drag, 0, max_throttle)
        bixler.throttle=2.33
        # print(bixler.throttle)

        bixler.step(0.01)
        time += 0.01
        # plt.scatter(x=time, y=bixler.airspeed)
        # plt.pause(0.001)
        # print(bixler.airspeed)
        airspeeds.append(bixler.airspeed)
        throttles.append(bixler.throttle)
        
        # print(bixler.position_e[2,0])
    # if bixler.position_e[2,0] > 0:
    #     break
    # print( "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(*[elem[0] for elem in bixler.get_state()]) )
    # print(f"time: {time}, x: {bixler.position_e[0,0]}, z: {bixler.position_e[2,0]}, pitch: {np.rad2deg(bixler.orientation_e[1,0])}, u: {bixler.velocity_b[0,0]}, w: {bixler.velocity_b[2,0]}, throttle: {bixler.throttle}")
    # print(f"airspeed: {bixler.airspeed}")

plt.plot(airspeeds)
plt.plot(throttles)
plt.show()



