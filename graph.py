import matplotlib.pyplot as plt

students = []
accuracy = []

# Read CSV file
with open("accuracy_data.csv", "r") as f:
    for line in f:
        s, a = line.strip().split(",")
        students.append(int(s))
        accuracy.append(float(a))

# Create bar graph
plt.figure()
plt.bar(students, accuracy)

plt.xlabel("Number of Students")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Number of Students")
plt.ylim(0, 100)


plt.savefig("accuracy_graph.png", dpi=300)
plt.show()