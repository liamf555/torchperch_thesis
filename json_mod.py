import json
import argparse 
from pathlib import Path
from shutil import copyfile


class JsonMod:

    def __init__(self, args):

        self.args = args

        self.wind_mode = args.wind_mode
        self.wind_params = args.wind_params

        
        Path(args.log_file).mkdir(parents=True, exist_ok=True)

        eval_dir = Path(args.log_file) / 'eval'
        eval_dir.mkdir(parents=True, exist_ok=True)

        copyfile("./sim_params.json", (args.log_file + "/sim_params.json"))
        # copyfile("./model_params", (args.log_file + "/model_params"))

        with open((args.log_file + "/sim_params.json"), "r") as jsonFile:
            self.data = json.load(jsonFile)

        if self.args.array:
            with open("../torchperch/scripts/bluecrystal/arrays.json", "r") as arrayFile:
                self.arrays = json.load(arrayFile)

    def json_array(self):
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
                

    # def json_single(self):
    #     for key, value in vars(self.args).items():
    #         if value is not None:
    #             if key is not "log_file" and key is not "array" and key is not "steady_var":
    #                 self.name = key + '_' + value
    #                 if key is "wind_mode":
    #                     wind_north = float(args.steady_vector)
    #                     self.data[key] = [wind_north, 0.0, 0.0]
    #                 else:
    #                     self.data[key] = value 

    def json_single(self):
        for key, value in vars(self.args).items():
            if value is not None:
                if key is not "log_file" and key is not "array" and key is not "wind_params" and key is not "kwargs":
                    if key is "start_config":
                        start_config = [float(i) for i in value]
                        self.data["start_config"] = start_config
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


        model_name = algorithm_name + '_' + 'final_model' 

        self.data["model_file"] = self.args.log_file + model_name
        self.data["log_file"] = self.args.log_file
        # self.data["kwargs"] = self.args.kwargs
    

        self.wind_amend()

        
    
        with open(self.args.log_file + "/sim_params.json", "w") as jsonFile:
            json.dump(self.data, jsonFile, indent=2)

    def wind_amend(self):
        wind_parama_args = self.args.wind_params

        if self.wind_mode == 'normal' or "steady":
            # self.wind_params = (wind_parama_args.split(" "))
            self.wind_params = [float(i) for i in self.wind_params]
        
        self.data["wind_params"] = self.wind_params 

if __name__ == "__main__":

    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split('=')
                getattr(namespace, self.dest)[key] = value


    parser = argparse.ArgumentParser(description='RL for Bixler UAV')
    parser.add_argument('--algorithm', '-a', type=str)
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--controller', type=str)
    parser.add_argument('--env', type = str)
    parser.add_argument('--log_file', type = str)
    parser.add_argument('--latency', action='store_true')
    parser.add_argument('--wind_mode', type = str)
    parser.add_argument('--noise', type = float)
    parser.add_argument('--wind_params', type = str, nargs='*')
    parser.add_argument('--variable_start', action='store_true')
    parser.add_argument('--array_flag', action = 'store_true', dest = 'array')
    parser.add_argument('--turbulence', type = str)
    parser.add_argument('--timesteps', type = int)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--start_config', type = str, nargs='*')
    parser.add_argument('--net_arch', type = str)
    parser.add_argument('--obs_noise', action='store_true')
    parser.add_argument('--framestack', type = str)

    args = parser.parse_args()

    amend = JsonMod(args)

    amend.json_amend()
