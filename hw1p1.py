import matplotlib.pyplot as plt
import numpy as np

import person

list_of_names = ['Roger', 'Mary', 'Luisa', 'Elvis']
list_of_ages  = [23, 24, 19, 86]
list_of_heights_cm = [175, 162, 178, 182]

for name in list_of_names:
    print("The name {:} is {:} letters long".format(name, len(name)))
    
list_length_name = [len(x) for x in list_of_names]

people = {}
 
for i in range(len(list_of_names)):
    people[list_of_names[i]] = person.Person(name = list_of_names[i], age = list_of_ages[i],height = list_of_heights_cm[i])

array_ages = np.array(list_of_ages)
array_heights_cm = np.array(list_of_heights_cm)

mean_ages = np.mean(array_ages)
mean_heights_cm = np.mean(array_heights_cm)

print("The average age = {:}".format(mean_ages))
print("The average height in cm  = {:}".format(mean_heights_cm))

plt.title("Age and Height in cm")
plt.scatter(array_ages, array_heights_cm)
plt.xlabel("Age")
plt.ylabel("Height in cm")
plt.grid(True)
plt.savefig("./age_height_cm_plot.png")
plt.show()
