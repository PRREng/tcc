# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from scipy import constants
import matplotlib.ticker as ticker
# from matplotlib.ticker import EngFormatter
# import scipy.io


def F_range(initial_frequency, last_frequency, number_of_points=100):
    frequency_Hz = np.logspace(np.log10(initial_frequency),
                               np.log10(last_frequency), number_of_points,
                               endpoint=True)
    angular_frequency = 2 * np.pi * frequency_Hz

    return angular_frequency, frequency_Hz


def Z_R(resistance):
    resistor_impedance = resistance
    return resistor_impedance


def Z_Q(non_ideal_capacitance, ideality_factor, angular_frequency):
    CPE_impedance = 1 / (non_ideal_capacitance *
                         (angular_frequency * 1j) ** ideality_factor)
    return CPE_impedance


def Z_W(sigma, angular_frequency):
    W_impedance = (sigma * np.sqrt(2)) / np.sqrt(1j * angular_frequency)
    return W_impedance


# print("F_range")
# print(F_range(0.01, 10000, number_of_points=10))
# print("Z_R")
# print(Z_R(1.5))
# print("Z_Q")
# print(Z_Q(1.8, 2, 10))
# print("Z_W")
# print(Z_W(0.9, 10))


def log_rand(initial_gen, last_gen, batch_size):
    initial_v = np.log(initial_gen)
    last_v = np.log(last_gen)
    log_array = np.exp((initial_v + (last_v - initial_v) *
                        np.random.rand(batch_size)))

    return log_array


def lin_rand(initial_gen, last_gen, batch_size):
    lin_arr = initial_gen + (last_gen - initial_gen) * \
              np.random.rand(batch_size)

    return lin_arr


# print("log_rand")
# print(log_rand(10, 1000, 32))
# print("lin_rand")
# print(lin_rand(10, 1000, 32))


# Array generator function
def genZR(batch_size, number_of_points, resistance):
    ZR = np.zeros((batch_size, number_of_points), dtype=complex)

    for idx in range(batch_size):
        for idx2 in range(number_of_points):
            ZR[idx][idx2] = Z_R(resistance[idx])

    return ZR


def genZQ(batch_size, number_of_points, non_ideal_capacitance,
          ideality_factor, angular_frequency):
    ZQ = np.zeros((batch_size, number_of_points), dtype=complex)

    for idx in range(batch_size):
        for idx2 in range(number_of_points):
            ZQ[idx][idx2] = Z_Q(non_ideal_capacitance[idx],
                                ideality_factor[idx],
                                angular_frequency[idx2])

    return ZQ


def genZW(batch_size, number_of_points, sigma, angular_frequency):
    ZW = np.zeros((batch_size, number_of_points), dtype=complex)

    for batch in range(batch_size):
        for point in range(number_of_points):
            ZW[batch][point] = Z_W(sigma[batch],
                                   angular_frequency[point])

    return ZW


# Plot formatter
plt.rcParams["axes.formatter.min_exponent"] = 4


class myformatter(ticker.LogFormatter):

    def _num_to_string(self, x, vmin, vmax):
        if x > 10000:
            s = "%1.0e" % x
        elif x < 1 and x >= 0.001:
            s = "%g" % x
        elif x < 0.001:
            s = "%1.0e" % x
        else:
            s = self._pprint_val(x, vmax - vmin)
        return s


xfmt = myformatter(labelOnlyBase=False, minor_thresholds=(4, 0.5))
yfmt = myformatter(labelOnlyBase=False, minor_thresholds=(3, 0.5))


# Plot the EIS spectrum
class Zplot:
    @staticmethod
    def full(ZZ, frequency, param, nrow=1, examp=1):
        px = 1 / plt.rcParams["figure.dpi"]  # pixels in inches

        _, axs = plt.subplots(figsize=(700*px, 200*px*nrow),
                              nrows=nrow, ncols=3)

        if nrow == 1:

            phase = np.degrees(np.arctan(ZZ[examp-1].imag/ZZ[examp-1].real))
            mag = np.absolute(ZZ[examp-1])

            paramtxt = ""
            if param != "":
                for idx in range(len(param[examp-1])):
                    if idx == 4:
                        paramtxt = paramtxt + " \n "
                    paramtxt = paramtxt + \
                        (format(param[examp-1][idx], ".3g"))+" | "

            axs[0].plot(ZZ[examp-1].real, -ZZ[examp-1].imag)
            axs[0].set_title(paramtxt)
            axs[0].set_xlabel("Z' (\u03A9)")
            axs[0].set_ylabel("-Z'' (\u03A9)")

            axs[1].plot(frequency, phase)
            axs[1].set_xscale('log')
            axs[1].set_title("Frequency vs. Phase")
            axs[1].set_xlabel("Frequency (Hz)")
            axs[1].set_ylabel("Phase (\u03B8)")
            axs[1].xaxis.set_minor_formatter(xfmt)

            axs[2].plot(frequency, mag)
            axs[2].set_xscale('log')
            axs[2].xaxis.set_minor_formatter(xfmt)
            axs[2].set_title("Frequency vs. |Z|")
            axs[2].set_xlabel("Frequency (Hz)")
            axs[2].set_ylabel("|Z| (\u03A9)")

            plt.tight_layout()

        elif nrow > 1:

            for row in range(nrow):

                phase = np.degrees(
                        np.arctan(ZZ[row+examp-1].imag/ZZ[row+examp-1].real))
                mag = np.absolute(ZZ[row+examp-1])

                paramtxt = ""
                if param != "":
                    for idx in range(len(param[row+examp-1])):
                        if idx == 4:
                            paramtxt = paramtxt + " \n "
                        paramtxt = paramtxt + \
                            (format(param[row+examp-1][idx], ".3g")) + \
                            " | "

                axs[row, 0].text(0.1, 0.9, row+examp,
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=axs[row, 0].transAxes)

                axs[row, 0].plot(ZZ[row + examp-1].real, -ZZ[row+examp-1].imag)
                axs[row, 0].set_title(paramtxt)
                axs[row, 0].set_xlabel("Z' (\u03A9)")
                axs[row, 0].set_ylabel("-Z'' (\u03A9)")

                axs[row, 1].plot(frequency, phase)
                axs[row, 1].set_xscale('log')
                axs[row, 1].set_title("Frequency vs. Phase")
                axs[row, 1].set_xlabel("Frequency (Hz)")
                axs[row, 1].set_ylabel("Phase (\u03B8)")
                axs[row, 1].xaxis.set_minor_formatter(xfmt)

                axs[row, 2].plot(frequency, mag)
                axs[row, 2].set_xscale('log')
                axs[row, 2].xaxis.set_minor_formatter(xfmt)
                axs[row, 2].set_title("Frequency vs. |Z|")
                axs[row, 2].set_xlabel("Frequency (Hz)")
                axs[row, 2].set_ylabel("|Z| (\u03A9)")

                plt.tight_layout()
        return

    @staticmethod
    def point(ZZ, nrow=1, examp=1):

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches

        _, axs = plt.subplots(figsize=(700*px, 200*px*nrow), nrows=nrow,
                              ncols=4)

        if nrow == 1:

            phase = np.degrees(np.arctan(ZZ[examp-1].imag/ZZ[examp-1].real))
            mag = np.absolute(ZZ[examp-1])

            axs[0].plot(ZZ[examp-1].real)
            axs[0].set_title("Z'")
            axs[0].set_xlabel("Point")
            axs[0].set_ylabel("Z' (\u03A9)")

            axs[1].plot(-ZZ[examp-1].imag)
            axs[1].set_title("Z''")
            axs[1].set_xlabel("Point")
            axs[1].set_ylabel("-Z'' (\u03A9)")

            axs[2].plot(phase)
            axs[2].set_title("Phase")
            axs[2].set_xlabel("Point")
            axs[2].set_ylabel("Phase (\u03B8)")
            axs[2].xaxis.set_minor_formatter(xfmt)

            axs[3].plot(mag)
            axs[3].xaxis.set_minor_formatter(xfmt)
            axs[3].set_title("|Z|")
            axs[3].set_xlabel("Point")
            axs[3].set_ylabel("|Z| (\u03A9)")

            plt.tight_layout()

        elif nrow > 1:

            for row in range(nrow):

                phase = np.degrees(np.arctan(
                    ZZ[row+examp-1].imag/ZZ[row+examp-1].real))
                mag = np.absolute(ZZ[row+examp-1])
                axs[row, 0].text(0.1, 0.9, row+examp,
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=axs[row, 0].transAxes)

                axs[row, 0].plot(ZZ[row+examp-1].real)
                axs[row, 0].set_title("Z'")
                axs[row, 0].set_xlabel("Point")
                axs[row, 0].set_ylabel("Z' (\u03A9)")

                axs[row, 1].plot(-ZZ[row+examp-1].imag)
                axs[row, 1].set_title("Z''")
                axs[row, 1].set_xlabel("Point")
                axs[row, 1].set_ylabel("-Z'' (\u03A9)")

                axs[row, 2].plot(phase)
                axs[row, 2].set_title("Phase")
                axs[row, 2].set_xlabel("Point")
                axs[row, 2].set_ylabel("Phase (\u03B8)")
                axs[row, 2].xaxis.set_minor_formatter(xfmt)

                axs[row, 3].plot(mag)
                axs[row, 3].xaxis.set_minor_formatter(xfmt)
                axs[row, 3].set_title("|Z|")
                axs[row, 3].set_xlabel("Point")
                axs[row, 3].set_ylabel("|Z| (\u03A9)")

                plt.tight_layout()

        return


def sim_cir1(batch_size, number_of_points, resistance_range, alpha_range,
             q_range, angular_frequency):

    R1 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr1 = genZR(batch_size, number_of_points, R1)

    R2 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr2 = genZR(batch_size, number_of_points, R2)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1],
                                         batch_size), 3)
    Q1 = log_rand(q_range[0], q_range[1], batch_size)
    Zq1 = genZQ(batch_size, number_of_points, Q1, ideality_factor1,
                angular_frequency)

    Zeq = Zr1 + 1 / (1 / Zr2 + 1 / Zq1)

    Zparam = []
    for idx in range(batch_size):
        Zparam.append([R1[idx], R2[idx], ideality_factor1[idx], Q1[idx]])

    return Zeq, np.array(Zparam)


def sim_cir2(batch_size, number_of_points, resistance_range, alpha_range,
             q_range, angular_frequency):

    R1 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr1 = genZR(batch_size, number_of_points, R1)

    R2 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr2 = genZR(batch_size, number_of_points, R2)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1],
                                         batch_size), 3)
    Q1 = log_rand(q_range[0], q_range[1], batch_size)
    Zq1 = genZQ(batch_size, number_of_points, Q1, ideality_factor1,
                angular_frequency)

    R3 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr3 = genZR(batch_size, number_of_points, R3)

    ideality_factor2 = np.round(lin_rand(alpha_range[0], alpha_range[1],
                                         batch_size), 3)
    Q2 = log_rand(q_range[0], q_range[1], batch_size)
    Zq2 = genZQ(batch_size, number_of_points, Q2, ideality_factor2,
                angular_frequency)

    Zeq = Zr1 + 1 / (1 / Zr2 + 1 / Zq1) + 1 / (1 / Zr3 + 1 / Zq2)

    Zparam = []
    for idx in range(batch_size):
        Zparam.append([R1[idx], R2[idx], R3[idx],
                       ideality_factor1[idx], Q1[idx],
                       ideality_factor2[idx], Q2[idx]])

    return Zeq, np.array(Zparam)


def sim_cir3(batch_size, number_of_points, resistance_range, alpha_range,
             q_range, sigma_range, angular_frequency):

    R1 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr1 = genZR(batch_size, number_of_points, R1)

    R2 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr2 = genZR(batch_size, number_of_points, R2)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1],
                                         batch_size), 3)
    Q1 = log_rand(q_range[0], q_range[1], batch_size)
    Zq1 = genZQ(batch_size, number_of_points, Q1, ideality_factor1,
                angular_frequency)

    sigma = log_rand(sigma_range[0], sigma_range[1], batch_size)
    Zw = genZW(batch_size, number_of_points, sigma, angular_frequency)

    Zeq = Zr1 + 1 / (1 / Zq1 + 1 / (Zr2 + Zw))

    Zparam = []
    for idx in range(batch_size):
        Zparam.append([R1[idx], R2[idx],
                       ideality_factor1[idx], Q1[idx],
                       sigma[idx]])

    return Zeq, np.array(Zparam)


def sim_cir4(batch_size, number_of_points, resistance_range, alpha_range,
             q_range, sigma_range, angular_frequency):

    R1 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr1 = genZR(batch_size, number_of_points, R1)

    R2 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr2 = genZR(batch_size, number_of_points, R2)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1],
                                         batch_size), 3)
    Q1 = log_rand(q_range[0], q_range[1], batch_size)
    Zq1 = genZQ(batch_size, number_of_points, Q1, ideality_factor1,
                angular_frequency)

    R3 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr3 = genZR(batch_size, number_of_points, R3)

    ideality_factor2 = np.round(lin_rand(alpha_range[0], alpha_range[1],
                                         batch_size), 3)
    Q2 = log_rand(q_range[0], q_range[1], batch_size)
    Zq2 = genZQ(batch_size, number_of_points, Q2, ideality_factor2,
                angular_frequency)

    sigma = log_rand(sigma_range[0], sigma_range[1], batch_size)
    Zw = genZW(batch_size, number_of_points, sigma, angular_frequency)

    Zeq = Zr1 + 1 / (1 / Zr2 + 1 / Zq1) + 1 / (1 / Zq2 + 1 / (Zr3 + Zw))

    Zparam = []
    for idx in range(batch_size):
        Zparam.append([R1[idx], R2[idx], R3[idx],
                       ideality_factor1[idx], Q1[idx],
                       ideality_factor2[idx], Q2[idx],
                       sigma[idx]])

    return Zeq, np.array(Zparam)


def sim_cir5(batch_size, number_of_points, resistance_range, alpha_range,
             q_range, sigma_range, angular_frequency):

    R1 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr1 = genZR(batch_size, number_of_points, R1)

    R2 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr2 = genZR(batch_size, number_of_points, R2)

    ideality_factor1 = np.round(lin_rand(alpha_range[0], alpha_range[1],
                                         batch_size), 3)
    Q1 = log_rand(q_range[0], q_range[1], batch_size)
    Zq1 = genZQ(batch_size, number_of_points, Q1, ideality_factor1,
                angular_frequency)

    R3 = log_rand(resistance_range[0], resistance_range[1], batch_size)
    Zr3 = genZR(batch_size, number_of_points, R3)

    ideality_factor2 = np.round(lin_rand(alpha_range[0], alpha_range[1],
                                         batch_size), 3)
    Q2 = log_rand(q_range[0], q_range[1], batch_size)
    Zq2 = genZQ(batch_size, number_of_points, Q2, ideality_factor2,
                angular_frequency)

    sigma = log_rand(sigma_range[0], sigma_range[1], batch_size)
    Zw = genZW(batch_size, number_of_points, sigma, angular_frequency)

    Zeq = Zr1 + 1 / (1 / (Zr2 + 1 / ((1 / (Zr3 + Zw)) + 1 / Zq2)) + 1 / Zq1)

    Zparam = []
    for idx in range(batch_size):
        Zparam.append([R1[idx], R2[idx], R3[idx],
                       ideality_factor1[idx], Q1[idx],
                       ideality_factor2[idx], Q2[idx],
                       sigma[idx]])

    return Zeq, np.array(Zparam)


def arrange_data(circuit, cir_class, batch_size, number_of_poits):
    real = circuit.real
    imag = circuit.imag

    x = np.zeros((batch_size, 2, number_of_poits))
    y = np.zeros(batch_size)

    for batch in range(batch_size):
        y[batch] = cir_class

        for point in range(number_of_points):
            x[batch][0][point] = real[batch][point]
            x[batch][1][point] = imag[batch][point]

    return x, y


def export_data(circuit, batch_size, number_of_points, numc):

    x = np.zeros((numc, batch_size, 2, number_of_points))
    y = np.zeros((numc, batch_size))

    for circ in range(numc):
        x[circ], y[circ] = arrange_data(circuit[circ], (circ),
                                        batch_size, number_of_points)

    x_data = x[0]
    y_data = y[0]

    for circ in range(numc - 1):
        x_data = np.append(x_data, x[circ + 1], axis=0)

    for circ in range(numc - 1):
        y_data = np.append(y_data, y[circ + 1], axis=0)

    return x_data, y_data


# EIS data simulation
if __name__ == "__main__":
    number_of_circuit = 5

    # Number of spectrum in each circuit
    batch_size = 32768

    # number of data points in each spectrum (plot)
    number_of_points = 100

    # Range of frequency
    angular_frequency, _ = F_range(0.01, 1000000, number_of_points)

    # Range of resistance
    resistance_range = [1e-1, 1e+4]

    # Range of ideality factor of CPE
    alpha_range = [0.8, 1.0]

    # Range of CPE capacitance
    q_range = [1e-5, 1e-3]

    # Range of sigma
    sigma_range = [1e0, 1e+3]

    Circuit_spec = np.zeros((number_of_circuit, batch_size, number_of_points),
                            dtype=complex)

    Circuit_spec[0], Circuit0_param = sim_cir1(batch_size, number_of_points,
                                               resistance_range, alpha_range,
                                               q_range, angular_frequency)
    Circuit_spec[1], Circuit1_param = sim_cir2(batch_size, number_of_points,
                                               resistance_range, alpha_range,
                                               q_range, angular_frequency)
    Circuit_spec[2], Circuit2_param = sim_cir3(batch_size, number_of_points,
                                               resistance_range, alpha_range,
                                               q_range, sigma_range,
                                               angular_frequency)
    Circuit_spec[3], Circuit3_param = sim_cir4(batch_size, number_of_points,
                                               resistance_range, alpha_range,
                                               q_range, sigma_range,
                                               angular_frequency)
    Circuit_spec[4], Circuit4_param = sim_cir5(batch_size, number_of_points,
                                               resistance_range, alpha_range,
                                               q_range, sigma_range,
                                               angular_frequency)

    all_param = []
    all_param.append(Circuit0_param.tolist())
    all_param.append(Circuit1_param.tolist())
    all_param.append(Circuit2_param.tolist())
    all_param.append(Circuit3_param.tolist())
    all_param.append(Circuit4_param.tolist())
    df = pd.DataFrame(all_param)
    print(df.head())
    print(df[0][0])

    x_data, y_data = export_data(Circuit_spec, batch_size,
                                 number_of_points, 5)
    print(x_data.shape, y_data.shape)
    print(y_data[4])
    print(y_data[8])
    print(y_data[12])
    print(y_data[16])

    import h5py

    with h5py.File('data2.h5', 'w') as f:
        f.create_dataset('x_data', data=x_data)
        f.create_dataset("y_data", data=y_data)
