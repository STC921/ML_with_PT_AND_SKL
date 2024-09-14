import csv
import matplotlib.pyplot as plt

threads = []
mat_vec_time = []
mflops = []

with open("results.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        threads.append(int(row['Threads']))
        mat_vec_time.append(float(row['Mat-Mat Time (sec)']))
        mflops.append(float(row['MFLOPS']))

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(threads, mat_vec_time, label='1-48 Threads', marker='o')
plt.xlabel('Threads')
plt.ylabel('Mat-Mat Time (sec)')
plt.title('Threads vs Mat-Mat Time (sec)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(threads, mflops, label='1-48 Threads', marker='o')
plt.xlabel('Threads')
plt.ylabel('MFLOPS')
plt.title('Threads vs MFLOPS')
plt.legend()

plt.tight_layout()
plt.show()
