import json
import argparse 
import os
from shutil import copyfile


class JsonMod:

    def __init__(self, args):

        self.args = args

        os.mkdir(args.log_file)
        copyfile("./sim_params.json", (args.log_file + "/sim_params.json"))

        with open((args.log_file + "/sim_params.json"), "r") as jsonFile:
            self.data = json.load(jsonFile)


        if self.args.array:
            print('goat')
            with open("../torchperch/scripts/bluecrystal/arrays.json", "r") as arrayFile:
                self.arrays = json.load(arrayFile)

        self.name = None


    def json_array(self):
        print('goat')
        for key, value in vars(self.args).items():
            if value is not None: 
                if str(key) is not "log_file" and key is not "array":
                    name = key + '_' + str(self.arrays[key][value])
                    self.name = name.replace('.', '_')
                    if key is "steady_vector":
                        wind_north = self.arrays[key][value]
                        self.data[key] = [wind_north, 0.0, 0.0]
                    else:
                        self.data[key] = self.arrays[key][value]
                

    def json_single(self):
        for key, value in vars(self.args).items():
            if value is not None:
                if key is not "log_file" and key is not "array":
                    self.name = key + '_' + value
                    if key is "steady_vector":
                        wind_north = float(args.steady_vector)
                        self.data[key] = [wind_north, 0.0, 0.0]
                    else:
                        self.data[key] = value 
        


    def json_amend(self):

        if self.args.array:
            self.json_array()
        else:
            self.json_single()

        if self.args.algorithm == None:
                algorithm_name = self.data["algorithm"]
        else:
            algorithm_name = self.args.algorithm


        model_name = '/' + algorithm_name + '_' + self.name 

        self.data["model_file"] = self.args.log_file + model_name
        self.data["log_file"] = self.args.log_file 

        with open(self.args.log_file + "/sim_params.json", "w") as jsonFile:
            json.dump(self.data, jsonFile, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RL for Bixler UAV')
    parser.add_argument('--algorithm', '-a', type=str)
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--controller', type=str)
    parser.add_argument('--log_file', type = str)
    parser.add_argument('--latency', type = float)
    parser.add_argument('--turbulence', type = str)
    parser.add_argument('--noise', type = float)
    parser.add_argument('--steady_vector', type = str)
    parser.add_argument('--variable_start', type =str)
    parser.add_argument('--array_flag', action = 'store_true', dest = 'array')
    args = parser.parse_args()

    amend = JsonMod(args)

    amend.json_amend()

    



