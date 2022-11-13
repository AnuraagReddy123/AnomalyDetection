logfile = open('log.txt', 'r')

sensors = []
lines = logfile.read().splitlines()

for i, line in enumerate(lines):
    if 'Degenerate mixture covariance' in line:
        sensors.append(lines[i-2])

logfile.close()

# write in new file

with open('problem_sensors.txt', 'w') as f:
    for sensor in sensors:
        f.write(sensor+'\n')