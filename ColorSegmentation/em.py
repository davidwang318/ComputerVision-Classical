import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt


def gaussian(x_, mean_, cov_):
    dim = mean_.shape[0]
    x_hat = (x_-mean_).reshape((-1, 1, dim))
    N = np.sqrt((2*np.pi)**dim * det(cov_))
    fac = np.einsum('...k,kl,...l->...', x_hat, inv(cov_), x_hat)
    return np.exp(-fac / 2) / N


def generate_data(mean_, cov_, size_):
    return np.random.multivariate_normal(mean_, cov_, size_)


class GaussianMixture:
    '''
    input: data and number of clusters
    '''
    def __init__(self, data_, k_):
        self.data = data_
        self.dim = self.data.shape[1]  # dimension of the Gaussian model
        self.K = k_  # number of clusters

        # Initialize the mean
        num = self.data.shape[0]

        self.mean = []
        self.cov = []
        for k in range(self.K):
            self.mean.append(np.sum(self.data[k*num:(k+1)*num, ...], axis=0)/num)
            self.cov.append(np.identity(self.dim)*200) #*200
        self.weight = np.array([1.0 for _ in range(self.K)])
        self.likelihood = 0.

    def __expectation__(self):
        prob_pri = [gaussian(self.data, self.mean[k], self.cov[k])*self.weight[k] for k in range(self.K)]
        total = sum(prob_pri)  # sum up over clusters
        no_prob = np.ones(total.shape)/self.K
        self.prob = [np.where(total != 0, prob_pri[k]/total, no_prob) for k in range(self.K)]

    def __maximization__(self):
        for k in range(self.K):
            p_total = np.sum(self.prob[k])
            p_weighted = self.prob[k].reshape((-1, 1))*self.data

            #  update new mean
            p_weighted_sum = np.sum(p_weighted, axis=0)
            self.mean[k] = p_weighted_sum/p_total

            # update new covariance
            p_hat = self.data-self.mean[k]
            p_cov = self.prob[k].reshape((-1, 1, 1))*p_hat[:, :, None]*p_hat[:, None, :]  # batch cross dot
            p_cov = np.sum(p_cov, axis=0)
            self.cov[k] = p_cov/p_total
            self.weight[k] = p_total/self.data.shape[0]

    def train(self):
        self.__expectation__()
        self.__maximization__()

    def getLikelihood(self):
        likelihood = [gaussian(self.data, self.mean[k], self.cov[k]) * self.weight[k] for k in range(self.K)]
        likelihood = np.sum(np.log(sum(likelihood)))
        return likelihood

    def getModel(self):
        return self.mean, self.cov, self.weight

    def getPdf(self, x_):
        pdf = np.zeros(x_.shape[0])
        for k in range(self.K):
            pdf += self.weight[k]*gaussian(x_, self.mean[k], self.cov[k]).reshape((-1))
        return pdf


if __name__ == "__main__":
    #  Gaussian model: [mean, cov]
    G1 = [np.array([0, 1]), np.identity(2)*0.2]
    G2 = [np.array([4,5]), np.identity(2)*0.1]
    G_list = [G1, G2]
    K = len(G_list)  # number of clusters

    #  Generate training data
    data_list = [generate_data(G_list[k][0], G_list[k][1], 100) for k in range(K)] # K*dataNum*dim
    data = np.concatenate(data_list) # dataNum*dim

    #  Mixture Gaussian model
    n_iterations = 50
    mix = GaussianMixture(data, len(G_list))
    # [mix.train() for i in range(n_iterations)]
    for i in range(n_iterations):
        print("number of iterations: ", i)
        mix.train()
        mean, var, _ = mix.getModel()

        #  Plot results dynamically
        plt.figure(1)
        plt.plot(data[:, 0], data[:, 1], '.')
        circle = []
        for k in range(K):
            circle.append(plt.Circle(mean[k], np.sqrt(var[k][0, 0]), edgecolor='r', facecolor='none'))
            plt.gcf().gca().add_artist(circle[k])
        plt.pause(0.1)
        plt.clf()

    #  Plot final results
    plt.close()
    plt.figure(1)
    plt.plot(data[:, 0], data[:, 1], '.')
    circle = []
    for k in range(K):
        circle.append(plt.Circle(mean[k], np.sqrt(var[k][0, 0]), edgecolor='r', facecolor='none'))
        plt.gcf().gca().add_artist(circle[k])

    #  Generate ground truth
    # x_gt = np.linspace(-10, 15, 300)
    # y_gt_list = [gaussian(x_gt, G_list[k][0], G_list[k][1]) for k in range(K)]
    # y_gt = sum(y_gt_list)/K

    # plt.figure()
    # plt.plot(x_gt, y_gt_list)

    #  Plot results
    plt.figure(2)
    data = np.sort(data, axis=0)
    prob = mix.getPdf(data)
    print(data.shape)
    plt.plot(data, prob, '.')
    plt.show()
    
