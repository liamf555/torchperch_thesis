import sys, csv

import numpy as np

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
        
        for i in range(15):
            error = np.abs(modelState[i] - referenceState[i])
            if error > epsilon :#* np.abs(modelState[i]) and not (error < 1e-8):
                print("Error comparing {} ({})".format(stateElemNames[i],i))
                print("Expected: {}, Actual: {}".format(referenceState[i],modelState[i][0]))

        # Set the actions
        bixler.sweep_rate = np.rad2deg(timestepData[0,15])
        bixler.elev_rate = np.rad2deg(timestepData[0,16])

        #newModelState = np.concatenate( ( referenceState[0:12], np.rad2deg(referenceState[12:]) ) )
        #bixler.set_state( np.expand_dims(newModelState,axis=0) )

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

if __name__ == '__main__':
    referenceData = loadData(sys.argv[1])
    testModel(referenceData)
