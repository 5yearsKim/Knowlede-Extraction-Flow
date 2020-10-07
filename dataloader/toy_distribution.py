import numpy as np
import matplotlib.pyplot as plt



def moon1(num_samples):
    x1 = np.random.normal(0, 1, num_samples // 2)
    x2 = np.random.normal(0, 1, num_samples // 2)
    
    y1 = x1**2 + (x2**2)
    y2 = -x2**2 - (x1**2)

    x1 += 2
    x2 -= 2
    y1 += 4
    y2 -= 4

    label = np.array([0]*(num_samples//2) + [1]*(num_samples//2))
    merged = np.array([np.append(x1, x2), np.append(y1, y2)]).T
    return merged, label

def moon2(num_samples):
    x1 = np.random.normal(0, 3, num_samples // 2)
    x2 = np.zeros(num_samples // 2)
    for idx in range(num_samples // 2):
        x2[idx] = np.random.normal(.3 * x1[idx] ** 2, 1)
    x1 -= 10
    x2 -= 40

    y1 = np.random.normal(0, 3, num_samples // 2)
    y2 = np.zeros(num_samples // 2)

    for idx in range(num_samples // 2):
        y2[idx] = np.random.normal(-.3 * y1[idx] ** 2, 1)
    y1 += 10
    y2 += 40

    label = np.array([0]*(num_samples//2) + [1]*(num_samples//2))
    merged = np.array([np.append(x1, y1), np.append(x2, y2)]).T
    return merged, label

def mixed(num_samples):
    x1 = np.random.normal(0, 2, num_samples // 2)
    x2 = np.random.normal(0, 2, num_samples // 2)
    
    x1 -= 1
    x2 += 1

    y1 = (x1 +0.2*x2)**2
    y2 = (x2 + 0.2*x1)**2

    x1 -= 1
    x2 += 1

    label = np.array([0]*(num_samples//2) + [1]*(num_samples//2))
    merged = np.array([np.append(x1, x2), np.append(y1, y2)]).T
    return merged, label


def circle(num_samples):
    r = np.random.normal(0, 2, num_samples // 2)
    theta1 = np.random.normal(0, 100000, num_samples // 2) 
    theta2 = np.random.normal(0, 100000, num_samples // 2) 

    x1 = (5 - r**2) * np.cos(theta1)
    x2 = (6 + r**2) * np.cos(theta2)

    y1 = (5 - r**2) * np.sin(theta1)
    y2 = (6 + r**2) * np.sin(theta2)

    label = np.array([0]*(num_samples//2) + [1]*(num_samples//2))
    merged = np.array([np.append(x1, x2), np.append(y1, y2)]).T
    return merged, label



def pizza(num_samples):
    divide = 4
    r = np.random.normal(0, 1, num_samples // 2)
    theta1 = np.random.normal(0, 100000, num_samples // 2) 
    theta1 = (2*(theta1//30) + 1) * 2*np.pi/divide + theta1%(2*np.pi/divide)
    theta2 = np.random.normal(0, 100000, num_samples // 2) 
    theta2 = (2*(theta2//30)) * 2*np.pi/divide + theta2%(2*np.pi/divide)

    x1 = (3 - r**2) * np.cos(theta1)
    x2 = (3 - r**2) * np.cos(theta2)

    y1 = (3 - r**2) * np.sin(theta1)
    y2 = (3 - r**2) * np.sin(theta2)

    label = np.array([0]*(num_samples//2) + [1]*(num_samples//2))
    merged = np.array([np.append(x1, x2), np.append(y1, y2)]).T
    return merged, label



def stars(num_samples):
    num_groups = 6
    size = 4
    x = np.array([])
    y = np.array([])
    for i in range(num_groups):
        x = np.append(x, np.random.normal(size * np.cos(2*np.pi/num_groups * i), 0.3, num_samples // num_groups ))
        y = np.append(y, np.random.normal(size * np.sin(2*np.pi/num_groups * i), 0.3, num_samples // num_groups ))
    merged = np.array([x + 16.,y]).T
#	merged =  (merged - np.mean(merged, axis=0)) / np.std(merged, axis=0) * 2
    return merged

if __name__ == "__main__":
    dataset, label = pizza(num_samples=500)
    colormap = np.array(['b', 'r'])
    plt.scatter(dataset[:, 0], dataset[:, 1], c=colormap[label], s=1)
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    # plt.grid(True)
    # plt.savefig("moon2_ref")
    plt.grid()
    plt.show()
    print(dataset)
    # np.save('moon1.npy', merged)