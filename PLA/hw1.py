#!/usr/bin/python3
# Homework 1 Code
import numpy as np
import matplotlib.pyplot as plt


def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
    
    # initialize the first weight vector / iteration variable t
    w = np.zeros((np.shape(data_in)[1]-1))
    t = 0

    # Loop through each row of data for PLA learning
        
    while True:
        t = t+1

        for x in (data_in):
            h_sign = np.sign(np.dot(w,x[:-1]))
            y = x[-1]

            if y != h_sign:
                w = w + x[:-1]*y
                break

        else:
            break


    return w, t


    # return w, iterations

def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW1
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    # print(data_set)

    # iterate num_exp times
    for i in range(num_exp):

        # print("hello")
        data_set = np.random.uniform((N, d + 1))
        # initialize np arrays and variables
        x = np.random.uniform(-1, 1, (N, d))
        w_opt = np.random.uniform(-1,1,(d+1,))

        x_first_d = np.ones((N, 1))
        # print(x_first_d)

        x_curr = np.concatenate((x_first_d, x), axis=1)
        # print(x_curr)
        
        y = np.sign(np.dot(x_curr,w_opt))
        
        data_set = np.concatenate((x_curr, y.reshape(N,1)),axis=1)
        
        w_learn, t_learn = perceptron_learn(data_set)
        # print("w_learn: ")
        # print(w_learn)
        # print("t_learn")
        # print(t_learn)

        # calculate values of R and rho -> theretical bound for t
        r = max(np.linalg.norm(x_curr,axis=1))
        rho = min(y*np.dot(x_curr,w_learn))
        
        # if rho < 0:
        #     print(rho)
        # print(r)
        # print(rho)
        theo_bound = r**2 * np.linalg.norm(w_learn)**2 / rho**2 - t_learn

        num_iters[i] = t_learn
        bounds_minus_ni[i] = theo_bound




    # Your code here, assign the values to num_iters and bounds_minus_ni:

    # print(num_iters)
    # print(bounds_minus_ni)
    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()
    

if __name__ == "__main__":
    main()
