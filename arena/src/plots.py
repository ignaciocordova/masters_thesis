import matplotlib.pyplot as plt
from matplotlib import rcParams

# Specify the font family and size
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12

# Data from the table
hours_ahead = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
persistence_forecast = [0.0416,
0.0609,
0.0837,
0.1006,
0.1176,
0.1233,
0.1317,
0.1384,
0.1440,
0.1500]
ViT = [0.0659,
0.0767,
0.0789,
0.0877,
0.0977,
0.1051,
0.1145,
0.1205,
0.1241,
0.1297]
ViViT = [0.0416,
0.0589,
0.0741,
0.0903,
0.0959,
0.1084,
0.1178,
0.1174,
0.1216,
0.1283]

# Customize plot attributes
point_size = 35
point_shape = 'o'  # Circle markers
x_label = "N hours ahead prediction"
y_label = "NMAE"
plot_title = "Comparison of NMAE Values"
legend_labels = ["Persistence forecast", "ViT", "ViViT"]

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(hours_ahead, persistence_forecast, marker=point_shape, label=legend_labels[0])
plt.plot(hours_ahead, ViT, marker=point_shape, label=legend_labels[1])
plt.plot(hours_ahead, ViViT, marker=point_shape, label=legend_labels[2])

# Customize plot labels and ticks
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.xticks(hours_ahead)  # Customize x-axis ticks
plt.yticks([0.05, 0.1, 0.15, 0.20])  # Customize y-axis ticks

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
