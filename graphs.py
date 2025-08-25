import numpy as np
import matplotlib.pyplot as plt
import cv2
import serial


def predict(input, num_p, noice_p):

    particle = []
    for i in range(num_p):
        particle.append(input + np.random.rand() * noice_p)
    return particle


def pdf(x, mu, sigma):
    pdf_value = []
    for i in range(len(x)):
        PI = np.pi

        if sigma > 0:

            coefficient = 1.0 / (sigma * np.sqrt(2 * PI))

            exponent = -0.5 * ((x[i] - mu) / sigma) ** 2

            pdf_value.append(coefficient * np.exp(exponent))

    return pdf_value


def update_weights(particles, z, measurement_noise, weights_old):
    likelihood = pdf(particles, z, measurement_noise)
    weights = []

    for i in range(len(weights_old)):

        weights.append(weights_old[i] * likelihood[i])
    total_weight = np.sum(weights)
    weight2 = []
    if total_weight > 1e-15:
        for i in range(len(weights_old)):
            weight2.append(weights[i] / total_weight)

    else:
        # Handle case where all weights are zero
        weight2 = np.ones(len(weights)) / len(weights)
    return weight2


def resample(particles, weights):
    n_particles = len(particles)
    # Create a new set of particles by drawing from the old set,
    # where the probability of being chosen is proportional to its weight.

    new_particles = []
    for i in range(n_particles):
        indices = int(np.random.uniform(n_particles))
        new_particles.append(particles[indices])
    # After resampling, all particles are important again
    weights = np.ones(n_particles) / n_particles
    return new_particles, weights


def estimate(particles, weights):
    n_p = len(particles)
    n_w = len(weights)
    sum_p = 0
    sum_w = 0
    for i in range(n_p):
        sum_p += particles[i] * weights[i]
    for j in range(n_w):
        sum_w += weights[j]
    return (sum_p) / (sum_w)


def run_simulation(x, n_parti, weights):
    p_noice = 0.1
    m_noice = 2
    # x_true = x + np.random.randn() * p_noice
    measurement = x
    # x_true + np.random.randn() * m_noice
    paticles = predict(x, n_parti, p_noice)
    weight = update_weights(paticles, measurement, m_noice, weights)
    paticles2, weight2 = resample(paticles, weight)
    estimate_n = estimate(paticles2, weight2)
    return x, estimate_n, weight2


def plot_p(y_axis, x1, x2):
    plt.plot(y_axis, x1, y_axis, x2)
    plt.show()


def main():
    s = serial.Serial('COM1',9600)
    u = 1
    y = 0
    y_a = []
    x_true = []
    measurement = []
    estimate_n = []
    n_parti = 50
    weights = np.ones(n_parti) / n_parti
    steps = 100
    for step in range(steps):
        if s.in_waiting > 0:
            data = s.read(ser.in_waiting)
            x = np.double(data)  # serial in

            x_a, x_es, we = run_simulation(x, n_parti, weights)
            x_true.append(x)
            # measurement.append(measurement2)
            estimate_n.append(x_es)
            weights = []
            weights = we
            y_a.append(y)
            y += 0.1  # 100ms delay

    plot_p(y_a, x_true, estimate_n)


if __name__ == "__main__":
    main()
