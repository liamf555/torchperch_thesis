
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy

import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile


class Callbacks(object):

    def __init__(self, log_dir):

        self.log_dir = log_dir

    def get_callback_vars(self, model, **kwargs): 
        """
        Helps store variables for the callback functions
        :param model: (BaseRLModel)
        :param **kwargs: initial values of the callback variables
        """
        # save the called attribute in the model
        if not hasattr(model, "_callback_vars"): 
            model._callback_vars = dict(**kwargs)
        else: # check all the kwargs are in the callback variables
            for (name, val) in kwargs.items():
                if name not in model._callback_vars:
                    model._callback_vars[name] = val
        return model._callback_vars # return dict reference (mutable)

    def auto_save_callback(self, _locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        # get callback variables, with default values if unintialized
        callback_vars = self.get_callback_vars(_locals["self"], n_steps=0, best_mean_reward=-np.inf) 

        # skip every 20 steps
        if callback_vars["n_steps"] % 500 == 0:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                copyfile((self.log_dir+'/monitor.csv'), (self.log_dir + '/monitor_'+ str(x[-1]) +'.csv'))

                # New best model, you could save the agent here
                if mean_reward > callback_vars["best_mean_reward"]:
                    callback_vars["best_mean_reward"] = mean_reward
                    # Example for saving best model
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    _locals['self'].save(self.log_dir + '/' +  'best_model_' + str(x[-1]))
                    

        callback_vars["n_steps"] += 1
        return True

    def plotting_callback(self,_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        # get callback variables, with default values if unintialized
        callback_vars = self.get_callback_vars(_locals["self"], plot=None, n_steps = 0)

        if callback_vars["n_steps"] % 100 == 0:
        
            # get the monitor's data
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if callback_vars["plot"] is None: # make the plot
                plt.ion()
                fig = plt.figure(figsize=(6,3))
                ax = fig.add_subplot(111)
                line, = ax.plot(x, y)
                callback_vars["plot"] = (line, ax, fig)
                plt.show()
            else: # update and rescale the plot
                callback_vars["plot"][0].set_data(x, y)
                callback_vars["plot"][-2].relim()
                callback_vars["plot"][-2].set_xlim([_locals["total_timesteps"] * -0.02, 
                                                    _locals["total_timesteps"] * 1.02])
                callback_vars["plot"][-2].autoscale_view(True,True,True)
                callback_vars["plot"][-1].canvas.draw()
        callback_vars["n_steps"] += 1
            

