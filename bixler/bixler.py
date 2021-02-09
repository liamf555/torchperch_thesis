import numpy as np
from wind.wind_sim import Wind
from wind.dryden import DrydenGustModel
import wandb

# Force numpy to raise FloatingPointError for overflow
np.seterr(all='raise')

accel_gravity = 9.81 # m.s^-2

class Bixler(object):
    
    def __init__(self, parameters):
        # Model parameters
        self.noiselevel = parameters.get("noise")

        # Physical parameters
        self.mass = 1.385 # kg (+100g for Pi, BEC etc)
        self.rho = 1.225  # kg.m^-3
        self.S = 0.26     # m^2
        self.c = 0.2      # m
        self.inertia = np.array([
            [ 0.042,     0,     0 ],
            [     0, 0.029,     0 ],
            [     0,     0, 0.066 ]
            ])

        # Compute weight vector
        self.weight = np.array([[0],[0],[accel_gravity]]) * self.mass

        # Setup rotation matricies
        self.dcm_earth2body = np.eye(3)
        self.dcm_wind2body = np.eye(3)
        self.jacobian = np.eye(3)

    
        # State
        self.position_e = np.zeros((3,1))
        self.velocity_b = np.array([[0],[0],[0]])
        self.Vr = np.zeros((3,1))
        self.acceleration_b = np.zeros((3,1))
        
        self.orientation_e = np.zeros((3,1)) # (rad)
        self.omega_b = np.zeros((3,1))       # (rad/s)
        self.omega_dot_b = np.zeros((3,1))   # (rad/s^2)
        
        # Control surfaces
        self.sweep    = 0.0 # (deg)
        self.elev     = 0.0 # (deg)
        self.rudder   = 0.0 # (deg)
        self.tip_port = 0.0 # (deg)
        self.tip_stbd = 0.0 # (deg)
        self.washout  = 0.0 # (deg)
        self.throttle = 0.0 # (N)
        
        # Air data
        self.alpha    = 0.0 # (deg)
        self.beta     = 0.0 # (deg)
        self.airspeed = 0.0 # (m/s)
        self.wind = np.zeros((3,1))
        self.wind_sim = Wind(wind_mode=parameters.get("wind_mode"), wind_params=parameters.get("wind_params"))
        
        # Control surface limits
        self.sweep_limits = np.rad2deg([-0.1745, 0.5236])
        # self.elev_limits = np.rad2deg([-0.872665, 0.872665]) # changed from +- 10 to 20
        self.elev_limits = np.rad2deg([-0.436332, 0.436332]) # changed from +- 10 to 25

        self.update_air_data(np.zeros((3,1)))
        self.velocity_e = np.array([[0],[0],[0]])
        # self.velocity_e[0] += parameters.get("wind_params")[0]
        self.wind_b = np.array([[0],[0],[0]])

        self.dryden = DrydenGustModel(Va = 13, intensity=parameters.get("turbulence"))

        self.np_random = None

        self.seed()
        
    def seed(self, seed=None):

        self.np_random = np.random.RandomState(seed)
        self.wind_sim.seed(seed)
        self.dryden.seed(seed)

        
    def _interpolate(self, x_target, x_data, y_data):
        if x_target < x_data[0]:
            m = (y_data[0] - y_data[1])/(x_data[0] - x_data[1])
            return m * x_target + (y_data[0] - m * x_data[0])
        if x_target > x_data[-1]:
            m = (y_data[-2] - y_data[-1])/(x_data[-2] - x_data[-1])
            return m * x_target + (y_data[-1] - m * x_data[-1])
        #highidx = np.argmax( x_data < np.array(x_target) )
        highidx = 1
        for i in range(1,len(x_data)):
            if x_data[i] > x_target:
                highidx = i
                break
        lowidx = highidx - 1
        m = (y_data[lowidx] - y_data[highidx])/(x_data[lowidx] - x_data[highidx])
        return m * x_target + (y_data[lowidx] - m * x_data[lowidx])
        #f = interpolate.interp1d(x_data,y_data,fill_value="extrapolate")
        #return f(x_target)
    
    def get_state(self):
        return np.concatenate((
            self.position_e,
            self.orientation_e,
            self.velocity_b,
            self.omega_b,
            np.array([
                [self.sweep],
                [self.elev],
                [self.tip_port]
                ])
            ), axis=0)
    
    def set_state(self,state):
        """Takes a row vector representing the state and stores it in local variables"""
        self.position_e    = np.array( [state[0,0:3]], dtype='float64' ).T
        self.orientation_e = np.array( [state[0,3:6]], dtype='float64' ).T
        self.velocity_b    = np.array( [state[0,6:9]], dtype='float64' ).T
        self.omega_b       = np.array( [state[0,9:12]], dtype='float64' ).T
        self.sweep         = np.float64(state[0,12])
        self.elev          = np.float64(state[0,13])
        self.tip_port      = np.float64(state[0,14])
        
        # Ensure air data reflects new state
        self.update_air_data(np.zeros((3,1)))
        self.dryden.reset()
        
        self.update_dcms()
        self.dryden.reset()

    def step(self, steptime):
        # Update the cosine matricies

        gusts = self.dryden.update(self.airspeed, steptime)
        # gusts_2 = self.dryden_2.get_turbulence_linear(self.counter)[0]

        # wandb.log({'gust_1':gusts[0]})
        # wandb.log({'gust_2':gusts_2})

        self.update_dcms()
        
        # Update derivatives (Accel and AngAccel)
        self.update_derivatives()
        
        self.velocity_b = self.velocity_b + self.acceleration_b * steptime

        # print(self.acceleration_b)
        self.omega_b = self.omega_b + self.omega_dot_b * steptime
    
        euler_rates = np.matmul(self.jacobian, self.omega_b)

        self.orientation_e = self.orientation_e + euler_rates * steptime
        
        self.wrap_orientation()
        
        self.velocity_e = np.matmul(self.dcm_earth2body.T, self.velocity_b)
        
        self.position_e = self.position_e + self.velocity_e * steptime

        # Update alpha and beta for next step
        self.update_air_data(gusts)


    def _update_dcm_earth2body(self):
        roll  = self.orientation_e[0,0]
        theta = self.orientation_e[1,0]
        yaw   = self.orientation_e[2,0]


        cr = np.cos(roll)
        sr = np.sin(roll)

        ct = np.cos(theta)
        st = np.sin(theta)

        cy = np.cos(yaw)
        sy = np.sin(yaw)

        # Build the rotation matrix
        rot_x = np.array([
            [ 1,   0,   0 ],
            [ 0,  cr, -sr ],
            [ 0,  sr,  cr ]
            ])
        rot_y = np.array([
            [  ct,  0, st ],
            [   0,  1,  0 ],
            [ -st,  0, ct ]
            ])
        rot_z = np.array([
            [ cy, -sy,  0 ],
            [ sy,  cy,  0 ],
            [  0,   0,  1 ]
            ])

        # Multiply the matrices together to get the combined matrix
        rot = np.matmul(np.matmul(rot_z,rot_y),rot_x)

        self.dcm_earth2body = np.linalg.inv(rot)
    
    def _update_dcm_wind2body(self):
        ca = np.cos(np.deg2rad(self.alpha))
        sa = np.sin(np.deg2rad(self.alpha))

        cb = np.cos(np.deg2rad(self.beta))
        sb = np.sin(np.deg2rad(self.beta))

        body2wind = np.array([
            [  ca * cb,   sb,  sa * cb ],
            [ -ca * sb,   cb, -sa * sb ],
            [    -sa  ,    0,    ca    ]
            ])
        self.dcm_wind2body = np.linalg.inv(body2wind)

    def _update_jacobian(self):
        phi = self.orientation_e[0,0]
        theta = self.orientation_e[1,0]

        sp = np.sin(phi)
        cp = np.cos(phi)

        st = np.sin(theta)
        ct = np.cos(theta)
        tt = np.tan(theta)
        # eqn 3.3
        self.jacobian = np.array([
            [ 1, sp*tt, cp*tt ],
            [ 0,   cp ,  -sp  ],
            [ 0, sp/ct, cp/ct ]
            ])

    def update_dcms(self):
        self._update_dcm_earth2body()
        self._update_dcm_wind2body()
        self._update_jacobian()

    def _cross(self,a,b):
        return np.array([ ((a[1] * b[2]) - (a[2] * b[1])),
                          ((a[2] * b[0]) - (a[0] * b[2])),
                          ((a[0] * b[1]) - (a[1] * b[0])) ])

    def update_derivatives(self):
        aeroforces_w, moments_w = self.get_forces_and_moments()
        
        # Forces
        # Rotate forces into body frame
        aeroforces_b = np.matmul(self.dcm_wind2body, aeroforces_w)
        # Rotate weight into the body frame
        weight_b = np.matmul(self.dcm_earth2body, self.weight)
        # Generate a thrust force (Assumes directly on x_body)
        thrust_b = np.array([[self.throttle],[0],[0]])
        # Get sum of forces on in body frame
        force_b = aeroforces_b + weight_b + thrust_b


        # Get acceleration of body
        self.acceleration_b = force_b * (1/self.mass)
        # Remove effects of rotating reference frame
        # np.cross slow for small sets so replace with own (function call overhead?)
        #self.acceleration_b = self.acceleration_b - np.cross(self.omega_b, self.velocity_b, axis=0)
        self.acceleration_b = self.acceleration_b - self._cross(self.omega_b, self.velocity_b)
        # Generate noise
        noise = self.np_random.rand(3,1) * self.noiselevel

        # noise = self.np_random.rand(3,1) * self.noiselevel

        # Add noise to acceleration
        self.acceleration_b = self.acceleration_b + noise

        # Moments
        # Rotate moments into body frame
        moments_b = np.matmul(self.dcm_wind2body, moments_w)

        # Define angular acceleration
        iw = np.matmul(self.inertia, self.omega_b)
        # np.cross slow for small sets so replace with own (function call overhead?)
        #cp = np.cross(self.omega_b, iw, axis=0)
        cp = self._cross(self.omega_b, iw)
        idw = moments_b - cp

        self.omega_dot_b = np.matmul(np.linalg.inv(self.inertia), idw)


    def update_air_data(self, gusts):

        self.wind = self.wind_sim.get_wind()

        # print(f"Wind: {self.wind}")

        # print(f"Gusts: {gusts}")

        # print(np.matmul(self.dcm_earth2body, self.wind.T))

        self.wind_b = np.matmul(self.dcm_earth2body, self.wind.T) + gusts.T

        # print(wind_b)

        # print(f"Wind_b: {wind_b}")

        # wandb.log({"wind": wind_b})

        self.Vr = self.velocity_b - self.wind_b

        # self.Vr = self.velocity_b

        # print(f"velocity_b: {self.velocity_b}")

        # print(f"Vr: {self.Vr}")
# 
        uSqd = self.Vr[0][0]**2
        vSqd = self.Vr[1][0]**2
        wSqd = self.Vr[2][0]**2

        # print(uSqd, vSqd, wSqd)

        self.airspeed = np.sqrt( uSqd + vSqd + wSqd )

        # print(f"airspeed {self.airspeed}")
        
        self.alpha = np.rad2deg(np.arctan2(self.Vr[2][0],self.Vr[0][0]))

        # print(self.alpha)
        
        if self.airspeed == 0:
            self.beta = 0
        else:
            cosBeta = (uSqd + wSqd) / (self.airspeed * np.sqrt( uSqd + wSqd ))
            # numpy slowness...
            #cosBeta = np.clip(cosBeta,-1.0,1.0)
            if cosBeta < -1.0: cosBeta = -1.0
            elif cosBeta > 1.0: cosBeta = 1.0
            # Apply sign convention
            self.beta = np.rad2deg(np.copysign(np.arccos(cosBeta), self.Vr[1][0]))
    
    def wrap_orientation(self):
        self.orientation_e = (self.orientation_e + np.pi*2) % (np.pi*2)
        self.orientation_e = self.orientation_e - (self.orientation_e > np.pi) * np.pi*2
    
    def get_forces_and_moments(self):
        C_L0, C_Lalpha, C_Lq, C_D0, C_Dalpha, C_Yv, C_l0, C_ldTipStbd, C_ldTipPort, C_lp, C_m0, C_malpha, C_melev, C_msweep, C_mwashout, C_mq, C_n0, C_nrudder, C_nr = self.get_coefficients()
        
        # Dynamic pressure
        Q = 0.5 * self.rho * (self.airspeed**2)


        # Roll rate
        p = np.rad2deg(self.omega_b[0,0])

        # Pitch rate
        q = np.rad2deg(self.omega_b[1,0])
        
        # Yaw Rate
        r = np.rad2deg(self.omega_b[2,0])

        # Sideslip
        # TODO: Fix definition of v, should be relative to airmass i.e. calculate using airdata
        # v = self.velocity_b[1,0]

        # Drag
        C_D = C_D0 + C_Dalpha * self.alpha

        D = Q * self.S * C_D
        X = -D
        
        # Sideforce
        #Y = 0.5 * self.rho * (v**2) * self.S * C_Yv * v
        Y = 0 # (removed to check against JAVA_MODEL)

        # Lift
        C_L = C_L0 + (C_Lalpha * self.alpha) + (C_Lq * q)
        L = Q * self.S * C_L
        Z = -L

        # Rolling moment
        C_l = C_l0 + (C_ldTipPort * self.tip_port) + (C_ldTipStbd * self.tip_stbd) + (C_lp * np.deg2rad(p))
        l = Q * self.S * self.c * C_l
        # Pitching moment
        C_m = C_m0 + (C_malpha * self.alpha) + (C_melev * self.elev) + (C_msweep * self.sweep) + (C_mwashout * self.washout) + (C_mq * q)
        # print(f'Q: {Q}, Cm: {C_m}')
        m = Q * self.S * self.c * C_m

        # print(Q, C_m)
        #print("0: {}, a: {}, e: {}, s: {}, w: {}, q: {}".format(C_m0,(C_malpha * self.alpha),(C_melev * self.elev),(C_msweep * self.sweep),(C_mwashout * self.washout),(C_mq * q)))
        #print("{},{},{},{},{},{}".format(C_m0,(C_malpha * self.alpha),(C_melev * self.elev),(C_msweep * self.sweep),(C_mwashout * self.washout),(C_mq * q)))
        #print("a: {}, e: {}, s: {}, w: {}, q: {}".format(self.alpha,self.elev,self.sweep,self.washout,q))
        #print("alpha: {}, u: {}, w: {}".format(self.alpha,self.velocity_b[0,0],self.velocity_b[2,0]))
        #print("Moment: {}".format(m))
        
        # Yawing moment
        C_n = C_n0 + (C_nrudder * self.rudder) + (C_nr * r)
        #n = Q * self.S * self.c * C_n
        n = Q * self.S * C_n # What!? No chord! To match JAVA_MODEL...

        # print(self.alpha)
        
        return (
            np.array([[X],[Y],[Z]]), # Forces
            np.array([[l],[m],[n]])  # Moments
            )

    def _get_coefficients_C_L(self):
        # Definition of characteristic Aerodynamic Coefficient Arrays obtained from wind tunnel testing 
        # The tested airspeeds being [6 8 10 12 14] m/s
        airspeeds = [6, 8, 10, 12, 14]
        
        # Lift coefficient data
        CL0_m5to5 = [ 0.3417,  0.3771, 0.39, 0.41, 0.43 ]
        CL0_5to10 = [ 0.634, 0.6106, 0.604, 0.586, 0.593 ]
        CL0_over10 = [ 1.103, 1.077, 1.0296, 1.026, 1.0313 ]

        CLalpha_m5to5 = [ 0.085, 0.08117, 0.0791, 0.0778, 0.0753 ]
        CLalpha_5to10 = [ 0.0299, 0.0354, 0.037, 0.039, 0.039 ]
        CLalpha_over10 = [ -0.0170, -0.0113, -0.0056, -0.005, -0.0048 ] # Post-stall

        
        # Dynamic Pitching Lift coefficient
        speed_sample_dynamic = [6, 8, 10]
        sweep_sample_dynamic = [0, 10, 20, 30]

        CLq_6ms  = [0.01012, 0.01217, 0.01005, 0.010104]
        CLq_8ms  = [0.008201, 0.0081213, 0.008229, 0.008633]
        CLq_10ms = [0.00555, 0.0073, 0.007708, 0.006868]

        C_L0_samples = []
        C_Lalpha_samples = []
        
        if self.alpha < 5:
            C_L0_samples = CL0_m5to5
            C_Lalpha_samples = CLalpha_m5to5
        elif (self.alpha >= 5) and (self.alpha < 10):
            C_L0_samples = CL0_5to10
            C_Lalpha_samples = CLalpha_5to10
        elif (self.alpha >= 10):
            C_L0_samples = CL0_over10
            C_Lalpha_samples = CLalpha_over10

        C_L0     = self._interpolate(self.airspeed, airspeeds, C_L0_samples)
        C_Lalpha = self._interpolate(self.airspeed, airspeeds, C_Lalpha_samples)

        C_Lq = 0.0

        if self.omega_b[1,0] < 0:
            C_Lq = 0
        else:
            # scipy.interpolate.RectBivariateSpline?
            CLq6 = self._interpolate(self.sweep, sweep_sample_dynamic, CLq_6ms)
            CLq8 = self._interpolate(self.sweep, sweep_sample_dynamic, CLq_8ms)
            CLq10 = self._interpolate(self.sweep, sweep_sample_dynamic, CLq_10ms)
            CLq_speeds = [CLq6, CLq8, CLq10];		
            C_Lq = self._interpolate( self.airspeed, speed_sample_dynamic, CLq_speeds)

        return (C_L0, C_Lalpha, C_Lq)

    def _get_coefficients_C_D(self):
        # The tested airspeeds being [6 8 10 12 14] m/s
        airspeeds = [6, 8, 10, 12, 14]

        CD0_m5to5 =  [  0.0682 ,  0.075 ,  0.0765, 0.087 ,  0.084  ]
        CD0_5to10 =  [ -0.02125, -0.0054,  0.0259, 0.0394,  0.0456 ]
        CD0_over10 = [  0.0027 , -0.0167, -0.0715, -0.06 , -0.039  ]
        
        
        CDalpha_m5to0 =  [ 0.0006 , 0      , 0.0003 , 0.0003 , 0.0005  ] 
        CDalpha_0to5 =   [ 0.0067 , 0.00463, 0.00584, 0.00414, 0.00436 ]
        CDalpha_5to10 =  [ 0.02457, 0.02071, 0.01596, 0.01366, 0.01204 ]
        CDalpha_over10 = [ 0.02217, 0.02185, 0.0257 , 0.0236 , 0.0205  ] #Post-Stall

        C_D0_samples = []
        C_Dalpha_samples = []
        
        if self.alpha < 0:
            C_D0_samples = CD0_m5to5
            C_Dalpha_samples = CDalpha_m5to0
        elif (self.alpha >= 0) and (self.alpha < 5):
            C_D0_samples = CD0_m5to5
            C_Dalpha_samples = CDalpha_0to5
        elif (self.alpha >= 5) and (self.alpha < 10):
            C_D0_samples = CD0_5to10
            C_Dalpha_samples = CDalpha_5to10
        elif self.alpha >= 10:
            C_D0_samples =  CD0_over10
            C_Dalpha_samples = CDalpha_over10

        C_D0     = self._interpolate(self.airspeed, airspeeds, C_D0_samples)
        C_Dalpha = self._interpolate(self.airspeed, airspeeds, C_Dalpha_samples)

        return (C_D0, C_Dalpha)

    def _get_coefficients_C_Y(self):
        C_Yv = -0.05 # Coeff defined relative to sideways velocity!
        return C_Yv

    def _get_coefficients_C_l(self):
        # Wing Tip Port and Starboard effectiveness Rolling Moment Coefficient
        sweep_sample_wingtip = [-10, 0, 10, 20]
        alpha_sample_wingtip = [-5, 0, 5, 10, 15, 20, 25]

        CMXportwingtip_m10sweep = [0.0230, 0.02153, 0.022415, 0.016924, 0.01618, 0.011757, 0.010183]
        CMXportwingtip_0sweep = [0.026895, 0.02629, 0.02659, 0.02242, 0.018992, 0.01786, 0.01563]
        CMXportwingtip_10sweep = [0.02987, 0.03098, 0.0323, 0.03146, 0.0292, 0.02529, 0.01958]
        CMXportwingtip_20sweep = [0.0289, 0.0304, 0.03098, 0.0225, 0.03213, 0.02575, 0.0293]

        CMXstarboardwingtip_m10sweep = [-0.01822, -0.019, -0.020148, -0.02041, -0.02316, -0.02308, -0.018754]
        CMXstarboardwingtip_0sweep = [-0.02083, -0.02190, -0.02313, -0.02769, -0.0327, -0.026486, -0.02888]
        CMXstarboardwingtip_10sweep = [-0.02062, -0.0253, -0.03089, -0.03504, -0.03468, -0.0349, -0.03094]
        CMXstarboardwingtip_20sweep = [-0.02167, -0.02885, -0.02985, -0.02913, -0.03361, -0.027941, -0.027445]
        
        C_l0 = 0.0
        C_ldTipPort = 0.0
        C_ldTipStbd = 0.0
        C_lp = -0.4950 # Eye average from Stanford student projects. Units? Probably per radian...
    
        if (self.alpha + self.tip_port) >= 15:
            C_ldTipPort = 0.0
        else:
            CMXportwingtipM10 = self._interpolate(self.alpha,alpha_sample_wingtip,CMXportwingtip_m10sweep)
            CMXportwingtip0 = self._interpolate(self.alpha,alpha_sample_wingtip,CMXportwingtip_0sweep)
            CMXportwingtip10 = self._interpolate(self.alpha,alpha_sample_wingtip,CMXportwingtip_10sweep)
            CMXportwingtip20 = self._interpolate(self.alpha,alpha_sample_wingtip,CMXportwingtip_20sweep)
            
            CMXportwingtip_sweeps = [CMXportwingtipM10, CMXportwingtip0, CMXportwingtip10, CMXportwingtip20]

            C_ldTipPort = self._interpolate(self.sweep,sweep_sample_wingtip,CMXportwingtip_sweeps)
    
        if (self.alpha + self.tip_stbd) >= 15:
            C_ldTipStbd = 0
        else:
            CMXstarboardwingtipM10 = self._interpolate(self.alpha,alpha_sample_wingtip,CMXstarboardwingtip_m10sweep)
            CMXstarboardwingtip0 = self._interpolate(self.alpha,alpha_sample_wingtip,CMXstarboardwingtip_0sweep)
            CMXstarboardwingtip10 = self._interpolate(self.alpha,alpha_sample_wingtip,CMXstarboardwingtip_10sweep)
            CMXstarboardwingtip20 = self._interpolate(self.alpha,alpha_sample_wingtip,CMXstarboardwingtip_20sweep)
            CMXstarboardwingtip_sweeps = [CMXstarboardwingtipM10, CMXstarboardwingtip0, CMXstarboardwingtip10, CMXstarboardwingtip20]

            C_ldTipStbd = self._interpolate(self.sweep, sweep_sample_wingtip, CMXstarboardwingtip_sweeps)

        return (C_l0, C_ldTipStbd, C_ldTipPort, C_lp)

    def _get_coefficients_C_m(self):
        # The tested airspeeds being [6 8 10 12 14] m/s
        airspeeds = [6, 8, 10, 12, 14]
        # Dynamic sample points
        speed_sample_dynamic = [6, 8, 10]
        sweep_sample_dynamic = [0, 10, 20, 30]
        
        # Pitching Moment Coefficients
        CMY0_m5to14 = [  0.01045,  0.0215,  0.02 ,  0.01 , -0.006  ]
        CMY0_over14 = [ -0.29075, -0.304 , -0.323, -0.307, -0.3065 ]

        CMYalpha_m5to0 =  [ -0.01185, -0.0115 , -0.01152, -0.01054, -0.01224 ]
        CMYalpha_0to14 =  [ -0.02151, -0.02325, -0.0245 , -0.02264, -0.02154 ]
        CMYalpha_over14 = [ 0       ,        0,        0,        0,        0 ]


        # Dynamic Pitching Moment Coefficient
        CMYq_6ms = [-0.0024608, -0.0033244, -0.003728, -0.0046806]
        CMYq_8ms = [-0.002354, -0.00269025, -0.0030326, -0.0038794]
        CMYq_10ms = [-0.002311, -0.0023768, -0.002771, -0.0029218]

        # Elevator effectiveness Pitching Moment Coefficient
        alpha_sample_elev = [-5, -2.5, 0, 2.5, 5]

        CMYelev_8p5to4p4 = [-0.00480, -0.00523, -0.006512, -0.004707, -0.00573]
        CMYelev_4p4to1p8 = [-0.01088, -0.01069, -0.011315, -0.01111, -0.01004]
        CMYelev_1p8tom1p2 = [-0.00533, -0.0085, -0.008993, -0.0099, -0.0103]
        CMYelev_m1p2tom5 = [-0.002184, -0.004632, -0.006184, -0.005978, -0.006237]

        CMYelev_overall = [-0.005355, -0.0068404, -0.0073926, -0.00745304, -0.0077259]

        # Wing Sweep effectiveness Pitching Moment Coefficient
        speed_sample_sweep = [6, 8, 10, 12, 14]
        alpha_sample_sweep = [-5, 0, 5, 10, 13, 15, 20, 25]

        # CMYsweep_8ms_5aoa = 0.011973;
        CMYsweep_6ms = [-0.00302, 0.000745, 0.003357, 0.00472, 0.005048, 0.00518, 0.00528, 0.00573]
        CMYsweep_8ms = [-0.00221, 0.00363, 0.00837, 0.01157, 0.01191, 0.0107, 0.0124, 0.0131]
        CMYsweep_10ms = [-0.00083, 0.00889, 0.01672, 0.02257, 0.02328, 0.02455, 0.0236, 0.0278]
        CMYsweep_12ms = [-0.00125, 0.0140, 0.02631, 0.03646, 0.03951]
        CMYsweep_14ms = [-0.00339, 0.01116, 0.02494]

        # Wing Tip Washout effectiveness Pitching Moment Coefficient
        sweep_sample_washout = [10, 20, 30]
        alpha_sample_washout = [-5, 0, 5, 10, 15, 20, 25]

        CMYwashout_0sweep_m30tom10deflec = [-0.00331, -0.003047, -0.00362, -0.00428, -0.00511, -0.001305, -0.000811]
        CMYwashout_0sweep_m10to5deflec = [0.01109, 0.009013, 0.01248, 0.01463]

        CMYwashout_10sweep = [0.00358, 0.00492, 0.0072, 0.009088, 0.00869, 0.009086, 0.01204]
        CMYwashout_20sweep = [0.01247, 0.0134, 0.01282, 0.01429, 0.01214, 0.01406, 0.01528]
        CMYwashout_30sweep = [0.01679, 0.01582, 0.01279, 0.01254, 0.01307, 0.01519, 0.01659]

        C_m0_samples = []
        C_malpha_samples = []

        if self.alpha < 0:
            C_m0_samples = CMY0_m5to14
            C_malpha_samples = CMYalpha_m5to0
        elif (self.alpha >= 0) and (self.alpha < 14):
            C_m0_samples = CMY0_m5to14
            C_malpha_samples = CMYalpha_0to14
        elif self.alpha >= 14:
            C_m0_samples = CMY0_over14
            C_malpha_samples = CMYalpha_over14
        
        C_m0 = self._interpolate(self.airspeed, airspeeds, C_m0_samples)
        #print('ASI: {}, result: {}, speeds: {}, samples: {}'.format(self.airspeed,C_m0,airspeeds,C_m0_samples))
        C_malpha = self._interpolate(self.airspeed, airspeeds, C_malpha_samples)
        
        C_melev_samples = []
        
        # Elevator effectiveness coefficient estimation
        if self.elev > 4.4:
            C_melev_samples = CMYelev_8p5to4p4
        elif (self.elev <= 4.4) and (self.elev > 1.8):
            C_melev_samples = CMYelev_4p4to1p8
        elif (self.elev <= 1.8) and (self.elev > -1.2):
            C_melev_samples = CMYelev_1p8tom1p2
        elif self.elev <= -1.2:
            C_melev_samples = CMYelev_m1p2tom5

        # Override the above to match JAVA_MODEL
        C_melev_samples = CMYelev_overall

        C_melev = self._interpolate(self.alpha,alpha_sample_elev,C_melev_samples)

        # Wing Sweep effictiveness coefficient estimation
        CMYsweep6 = self._interpolate(self.alpha, alpha_sample_sweep,CMYsweep_6ms)
        CMYsweep8 = self._interpolate(self.alpha, alpha_sample_sweep,CMYsweep_8ms)
        CMYsweep10 = self._interpolate(self.alpha, alpha_sample_sweep,CMYsweep_10ms)
        CMYsweep12 = self._interpolate(self.alpha, alpha_sample_sweep[0:5],CMYsweep_12ms)
        CMYsweep14 = self._interpolate(self.alpha, alpha_sample_sweep[0:3],CMYsweep_14ms)

        CMYsweep_speeds = [CMYsweep6, CMYsweep8, CMYsweep10, CMYsweep12, CMYsweep14]
        C_msweep = self._interpolate(self.airspeed,speed_sample_sweep,CMYsweep_speeds)

        # Wing Tip Symetric Washout effictiveness coefficient estimation
        C_mwashout = 0.0
        
        if self.sweep < 5:
            if self.washout <= -10:
                C_mwashout = self._interpolate(self.alpha, alpha_sample_washout, CMYwashout_0sweep_m30tom10deflec)
            elif self.washout > -10:
                C_mwashout = self._interpolate(self.alpha, alpha_sample_washout[3:7], CMYwashout_0sweep_m10to5deflec)
        elif self.sweep >= 5:
            CMYwashout10 = self._interpolate(self.alpha, alpha_sample_washout, CMYwashout_10sweep)
            CMYwashout20 = self._interpolate(self.alpha, alpha_sample_washout, CMYwashout_20sweep)
            CMYwashout30 = self._interpolate(self.alpha, alpha_sample_washout, CMYwashout_30sweep)
            CMYwashout_sweeps = [CMYwashout10, CMYwashout20, CMYwashout30]

            C_mwashout = self._interpolate(self.sweep, sweep_sample_washout, CMYwashout_sweeps)

        # Dynamic Pitching Moment Coefficient estimation
        CMYq6 = self._interpolate(self.sweep,sweep_sample_dynamic,CMYq_6ms)
        CMYq8 = self._interpolate(self.sweep,sweep_sample_dynamic,CMYq_8ms)
        CMYq10 = self._interpolate(self.sweep,sweep_sample_dynamic,CMYq_10ms)
        CMYq_speeds = [CMYq6, CMYq8, CMYq10]
        C_mq = self._interpolate(self.airspeed,speed_sample_dynamic,CMYq_speeds)

        return (C_m0, C_malpha, C_melev, C_msweep, C_mwashout, C_mq)

    def _get_coefficients_C_n(self):
        # Rudder effectiveness coefficient estimation
        #C_n0 = 0.0020716
        C_n0 = 0.0 # From JAVA_MODEL

        # Rudder effectiveness Yawing Moment Coefficient
        alpha_sample_rudd = [-5, -2.5, 0, 2.5, 5]

        CMZrudd_20to10 = [0.0003325, 0.0002946, 0.0002567, 0.000308, 0.00036241]
        CMZrudd_10to0 = [0.0003961, 0.00038266, 0.00036924, 0.000399, 0.00042889]
        CMZrudd_0tom10 = [0.0003996, 0.0003719, 0.00034422, 0.0003965, 0.0004488]
        CMZrudd_m10tom20 = [0.00025022, 0.0002496, 0.00024895, 0.00031153, 0.0003741]

        C_nrudder_samples = []
        if self.rudder > 10:
            C_nrudder_samples = CMZrudd_20to10
        elif (self.rudder <= 10) and (self.rudder > 0):
            C_nrudder_samples = CMZrudd_10to0
        elif (self.rudder <= 0) and (self.rudder > -10):
            C_nrudder_samples = CMZrudd_0tom10
        elif self.rudder <= -10:
            C_nrudder_samples = CMZrudd_m10tom20

        C_nrudder = self._interpolate(self.alpha, alpha_sample_rudd, C_nrudder_samples)

        #C_nr = -0.05 # Eye average from Stanford student projects. Units? Probably per radian...
        C_nr = -0.002 # From JAVA_MODEL. Per degree

        return (C_n0, C_nrudder, C_nr)

    def get_coefficients(self):
        
        C_L0, C_Lalpha, C_Lq = self._get_coefficients_C_L()
        C_D0, C_Dalpha = self._get_coefficients_C_D()
        C_Yv = self._get_coefficients_C_Y()
        C_l0, C_ldTipStbd, C_ldTipPort, C_lp = self._get_coefficients_C_l()
        C_m0, C_malpha, C_melev, C_msweep, C_mwashout, C_mq = self._get_coefficients_C_m()
        C_n0, C_nrudder, C_nr = self._get_coefficients_C_n()
        
        return (
            C_L0, C_Lalpha, C_Lq,
            C_D0, C_Dalpha,
            C_Yv,
            C_l0, C_ldTipStbd, C_ldTipPort, C_lp,
            C_m0, C_malpha, C_melev, C_msweep, C_mwashout, C_mq,
            C_n0, C_nrudder, C_nr
            )

if __name__ == '__main__':
    import sys
    
    bixler = Bixler()
    
    if len(sys.argv) > 1:
    
        def setairstate(airstate):
            bixler.airspeed = airstate[0]
            bixler.alpha = airstate[1]
            bixler.beta = airstate[2]
            
            bixler.elev = airstate[4]
            bixler.sweep = airstate[5]
            bixler.washout = airstate[6]
            bixler.tip_stbd = airstate[7]
            bixler.tip_port = airstate[8]
            bixler.rudder = airstate[9]
            
            bixler.omega_b[0,0] = np.deg2rad(airstate[10])
            bixler.omega_b[1,0] = np.deg2rad(airstate[ 3])
            bixler.omega_b[2,0] = np.deg2rad(airstate[11])
            
        # Setup the model state
        setairstate([ float(x) for x in sys.argv[1:] ])
        
        print(bixler.get_forces_and_moments())
    
    else:
        print("x,y,z,Roll,Pitch,Yaw,u,v,w,Roll rate,Theta dot,Yaw rate,Sweep,Elevator,Tip")
        
        for i in np.arange(200):
            for i in range(1,10):
                bixler.step(0.01)
            print( "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(*[elem[0] for elem in bixler.get_state()]) )
