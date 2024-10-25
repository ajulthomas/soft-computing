import matplotlib.pyplot as plt

# Data for the scatter plot
average_wait_time = [
    40.45328,
    38.4595,
    38.6992,
    43.752,
    42.3325,
    38.4211,
    63.3025,
    69.14,
    75.14,
]
max_wait_time = [631, 746, 722, 526, 777, 636, 808, 826, 715]

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(average_wait_time, max_wait_time, color="blue", marker="o")
plt.title("Scatter Plot of Average Wait Time vs Max Wait Time")
plt.xlabel("Average Wait Time")
plt.ylabel("Max Wait Time")
plt.grid(True)
plt.show()
