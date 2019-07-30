import unittest

import sys, csv, numpy as np

from bixler import Bixler

def testModel(referenceData):
    stateElemNames = ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r", "sweep", "elev", "tip"]
    
    epsilon = 0.01

    bixler = Bixler()

    # Re-orient data as lower levels expect
    referenceData = np.expand_dims(referenceData,axis=1)

    # Set initial state to first line of report
    bixler.set_state(referenceData[0])
    
    # Set the initial action
    bixler.sweep_rate = np.rad2deg(referenceData[0,0,15])
    bixler.elev_rate = np.rad2deg(referenceData[0,0,16])


    # Step forwards by 0.1s in 0.01s increments
    for i in range(10):
        bixler.step(0.01)

    count = 0

    # Loop over remaining lines
    for timestepData in referenceData[1:]:
        # Compare state output to reference
        modelState = bixler.get_state()
        # Convert control surface positions to radians
        modelState = np.concatenate( ( modelState[0:12], np.deg2rad(modelState[12:]) ) )
        referenceState = timestepData[0,0:15]
        count = count + 1
        print("Step: {}".format(count))
        #print("Mdl: {}".format(modelState.T))
        #print("Ref: {}".format(referenceState))
        
        stop = False
        
        for i in range(15):
            error = np.abs(modelState[i] - referenceState[i])
            if error > epsilon :#* np.abs(modelState[i]) and not (error < 1e-8):
                print("Error comparing {} ({})".format(stateElemNames[i],i))
                print("Expected: {}, Actual: {}".format(referenceState[i],modelState[i][0]))
                print("Airstate: {} {} {} {} {} {} {} {} {} {} {} {} ".format(
                    bixler.airspeed, bixler.alpha, bixler.beta,
                    bixler.omega_b[1,0],
                    bixler.elev, bixler.sweep, bixler.washout,
                    bixler.tip_stbd, bixler.tip_port, bixler.rudder,
                    bixler.omega_b[0,0], bixler.omega_b[2,0]))
                stop = True
                break
        
        if stop:
            break

        # Set the actions
        bixler.sweep_rate = np.rad2deg(timestepData[0,15])
        bixler.elev_rate = np.rad2deg(timestepData[0,16])

        #newModelState = np.concatenate( ( referenceState[0:12], np.rad2deg(referenceState[12:]) ) )
        #bixler.set_state( np.expand_dims(newModelState,axis=0) )
        
        print("Airstate: {} {} {} {} {} {} {} {} {} {} {} {} ".format(
            bixler.airspeed, bixler.alpha, bixler.beta,
            bixler.omega_b[1,0],
            bixler.elev, bixler.sweep, bixler.washout,
            bixler.tip_stbd, bixler.tip_port, bixler.rudder,
            bixler.omega_b[0,0], bixler.omega_b[2,0]))

        # Step forwards by 0.1s in 0.01s increments
        for i in range(10):
            bixler.step(0.01)


def loadData(filename):
    with open(filename,"rb") as datafile:
        rows = list(csv.reader(datafile, delimiter=' '))
        referenceData = np.empty([len(rows), 18])
        for idx in range(len(rows)):
            # First elem is time, last elem is windows newline mess
            rowFloats = [float(x) for x in rows[idx][1:-1]]
            # Include the tip data
            rowFloats = rowFloats[:14] + [0.0] + rowFloats[14:] + [0.0]
            referenceData[idx,:] = np.array(rowFloats)
        return referenceData

class TestBixler(unittest.TestCase):
    
    def setUp(self):
        self.bixler = Bixler()
        self.initial_state = np.array([[-40,0,-2, 0,0,0, 13,0,0, 0,0,0, 0,0,0]])
        self.bixler.set_state(self.initial_state)

    def test_set_action(self):
        # Reference values
        elev_rates = [-60, -10, -5, 0, 5, 10, 60]
        sweep_rates = [ -60, -10, -5, 0, 5, 10, 60 ]
        
        # For each action combination
        for elev_action in range(7):
            for sweep_action in range(7):
                with self.subTest(elev_action=elev_action, sweep_action=sweep_action):
                    # Call method
                    self.bixler.set_action(elev_action*7 + sweep_action)
                    # Check results
                    self.assertEqual( self.bixler.elev_rate, elev_rates[elev_action],
                                      'Incorrect elev rate' )
                    self.assertEqual( self.bixler.sweep_rate, sweep_rates[sweep_action],
                                      'Incorrect sweep rate' )

    def test_set_state(self):
        # Generate random state
        position_e = np.random.random((3,1))
        orientation_e = np.random.random((3,1))
        velocity_b = np.random.random((3,1))
        omega_b = np.random.random((3,1))
        sweep = np.random.random((1,1))
        elev = np.random.random((1,1))
        tip_port = np.random.random((1,1))
        
        # Setup state as row vector
        test_state = np.concatenate(
            (position_e, orientation_e, velocity_b, omega_b, sweep, elev, tip_port)
            ).T
        
        # Call method
        self.bixler.set_state(test_state)
        
        # Check types
        self.assertTrue(self.bixler.position_e.dtype == np.float64().dtype, 'Incorrect type for position_e' )
        self.assertTrue(self.bixler.orientation_e.dtype == np.float64().dtype, 'Incorrect type for orientation_e' )
        self.assertTrue(self.bixler.velocity_b.dtype == np.float64().dtype, 'Incorrect type for velocity_b' )
        self.assertTrue(self.bixler.omega_b.dtype == np.float64().dtype, 'Incorrect type for omega_b' )
        self.assertTrue(self.bixler.sweep.dtype == np.float64().dtype, 'Incorrect type for sweep' )
        self.assertTrue(self.bixler.elev.dtype == np.float64().dtype, 'Incorrect type for elev' )
        self.assertTrue(self.bixler.tip_port.dtype == np.float64().dtype, 'Incorrect type for tip_port' )
        
        # Check results
        self.assertTrue( np.all(self.bixler.position_e == position_e), 'Incorrect position_e' )
        self.assertTrue( np.all(self.bixler.orientation_e == orientation_e), 'Incorrect orientation_e' )
        self.assertTrue( np.all(self.bixler.velocity_b == velocity_b), 'Incorrect velocity_b' )
        self.assertTrue( np.all(self.bixler.omega_b == omega_b), 'Incorrect omega_b' )
        self.assertEqual( self.bixler.sweep, sweep, 'Incorrect sweep' )
        self.assertEqual( self.bixler.elev, elev, 'Incorrect elev' )
        self.assertEqual( self.bixler.tip_port, tip_port, 'Incorrect tip_port' )
    
    def test_terminal_conditions(self):
        # Requires test_set_state to have passed...
        
        with self.subTest( condition='Non-terminal' ):
            self.assertFalse( self.bixler.is_terminal(), 'Terminal condition falsely triggered' )
        
        with self.subTest( condition='Flip over' ):
            self.bixler.orientation_e[1,0] = np.pi/2 + 0.05
            self.assertTrue( self.bixler.is_terminal(), 'Flip over condition failed to trigger' )

        self.bixler.set_state(self.initial_state)
        
        with self.subTest( condition='Through ground' ):
            self.bixler.position_e[2,0] = 1
            self.assertTrue( self.bixler.is_terminal(), 'Through ground condition failed to trigger' )

        self.bixler.set_state(self.initial_state)

        with self.subTest( condition='Past endpoint' ):
            self.bixler.position_e[0,0] = 12
            self.assertTrue( self.bixler.is_terminal(), 'Past endpoint condition failed to trigger' )



if __name__ == '__main__':
    referenceData = loadData(sys.argv[1])
    testModel(referenceData)
