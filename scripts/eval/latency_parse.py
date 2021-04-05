# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
file_directory = Path("./experiments/logs/latency_logs/new")
dataframes = []
# %%                
for log in file_directory.glob('*.txt'):

    states = []
    lat_ep = False

    dataframe = pd.DataFrame()
    with open(log, "r") as file:
        for line in file:
            if 'Entering ML mode' in line:
                lat_ep = True
                first_flag = True
                time = 0
                states = []
            elif "Experimental mode disallowed" in line:
                lat_ep = False
                df = pd.DataFrame(states, columns=['Time', 'Latency'])
                dataframe = dataframe.append(df)
            if lat_ep:
                if "S: " in line:
                    line_list = line.split("]", 1)[1].split()
                    latencies = []
                    for item in line_list:
                        try:
                            latencies.append(int(item))
                        except:
                            pass
                    if first_flag: 
                        last_latency = latencies[0]
                        first_flag = False
                    latency = latencies[1] - latencies[0]
                    if abs(latency) < 1000:

                        time_delta = latencies[0] - last_latency
                        time += time_delta
                        last_latency = latencies[0]
    
                        states.append([time/1000, latency])
        dataframes.append(dataframe)


# # pd.set_option("display.max_rows", None, "display.max_columns", None)
final_frame = pd.DataFrame(columns=['Time', 'Latency'])
for i in dataframes:
    final_frame = final_frame.append(i)

print(final_frame)

final_frame["Latency"].hist(bins=100, normed=True)
# plt.show()




mu, sigma = 0, 1
s = 19 + np.random.lognormal(mu, sigma, 10000000)


count, bins, ignored = plt.hist(s, 1000, density=True, align='mid')

x = np.linspace(min(bins), max(bins), 10000)

pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))

       / (x * sigma * np.sqrt(2 * np.pi)))

plt.plot(x, pdf, linewidth=2, color='r')

axes = plt.gca()
# axes.set_xlim([0,100])

plt.show()

# %%
