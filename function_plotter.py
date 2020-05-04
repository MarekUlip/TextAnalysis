import matplotlib.pyplot as plt
import numpy as np

# Create the vectors X and Y
x = np.array(range(-100,100))
y = 1 / (1+ np.e**(0.1*x*-1))

# Create the plot
plt.plot(x,y,label='y = x**2')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

# Add a title
plt.title('My first Plot with Python')

# Add X and y Label
plt.xlabel('x axis')
plt.ylabel('y axis')

# Add a grid
plt.grid(alpha=.4,linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.show()