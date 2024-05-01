import matplotlib.pyplot as plt

# Read data from the text file
data = []
with open('execution_times.txt', 'r') as file:
    for line in file:
        parts = line.split()
        size = int(parts[0])
        time = float(parts[1])
        data.append((size, time))

# Separate sizes and times for plotting
sizes = [item[0] for item in data]
times = [item[1] for item in data]

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b')
plt.xlabel('Matrix Size')
plt.ylabel('Average Execution Time (seconds)')
plt.title('Performance of Gaussian Elimination')
plt.grid(True)
plt.xticks(sizes)
plt.show()
