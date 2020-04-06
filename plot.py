import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    with open("log.pickle", "rb") as f:
        data = [sum(points) / len(points) for points in pickle.load(f)]

    plt.plot(data, label="Put your ad here", marker='x', markersize=4)
    plt.title("Monte Carlo off-policy control\n {}".format("algorithm"), fontsize=23)
    plt.xlabel("Episode number", fontsize=15)
    plt.ylabel("Mean squared error, Q vs Q*", fontsize=15)
    plt.legend(fontsize=20, loc="upper right")
    plt.show()
