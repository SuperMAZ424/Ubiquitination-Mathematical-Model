import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# important note:
# 1- all rate constants are in (nM.S)-1
# 2- time is in sec
# 3- input initial concentrations are in nM (you have to calculate them to fit this unit)
# 4- all output concentrations at different times are in nM concentration

def odes(x, t):
    # constants
    k1f = 0.00001                                                                  #nM-1.S-1
    k1b = 0.55                                                                     #S^-1
    k2f = 0.00001                                                                  #nM-1.S-1
    k2b = 0.00019                                                                  #nM-1.S-1
    k3f = 0.001                                                                    #nM-1.S-1
    k3b = 0.37                                                                     #S^-1
    k4 = [0, 0.00034, 0.0078, 0.002, 0.0011, 0.00062, 0.00082, 0.0008, 0.0005]     #nM-1.sec-1
    k5 = [0, 0.4, 0.29, 0.27, 0.29, 0.89, 0.8, 0.5, 0.2]                           #sec-1
    k6 = [0, 0, 0, 0, 0.01, 0.02, 0.04, 0.06, 0.08]                                #nM-1.sec-1
    k7 = [0, 0, 0, 0, 0.1, 0.2, 0.4, 0.6, 0.8]                                     #sec-1

    # assign each ode to a vector
    ub = x[0]
    E1 = x[1]
    E2 = x[2]
    E3 = x[3]
    S26 = x[4]
    S = x[5]
    ubE1 = x[6]
    ubE2 = x[7]
    SE3 = x[8]
    ub1SE3 = x[9]
    ub2SE3 = x[10]
    ub3SE3 = x[11]
    ub4SE3 = x[12]
    ub5SE3 = x[13]
    ub6SE3 = x[14]
    ub7SE3 = x[15]
    ub8SE3 = x[16]
    ub1S = x[17]
    ub2S = x[18]
    ub3S = x[19]
    ub4S = x[20]
    ub5S = x[21]
    ub6S = x[22]
    ub7S = x[23]
    ub8S = x[24]
    ub4S26S = x[25]
    ub5S26S = x[26]
    ub6S26S = x[27]
    ub7S26S = x[28]
    ub8S26S = x[29]
    #pep = x[30]


    V1f = k1f * ub * E1
    V1b = k1b * ubE1
    V2f = k2f * ubE1 * E2
    V2b = k2b * ubE2 * E1
    V3f = k3f * S * E3
    V3b = k3b * SE3
    V4_1 = k4[1] * SE3 * ubE2
    V4_2 = k4[2] * ub1SE3 * ubE2
    V4_3 = k4[3] * ub2SE3 * ubE2
    V4_4 = k4[4] * ub3SE3 * ubE2
    V4_5 = k4[5] * ub4SE3 * ubE2
    V4_6 = k4[6] * ub5SE3 * ubE2
    V4_7 = k4[7] * ub6SE3 * ubE2
    V4_8 = k4[8] * ub7SE3 * ubE2
    V5_1 = k5[1] * ub1SE3
    V5_2 = k5[2] * ub2SE3
    V5_3 = k5[3] * ub3SE3
    V5_4 = k5[4] * ub4SE3
    V5_5 = k5[5] * ub5SE3
    V5_6 = k5[6] * ub6SE3
    V5_7 = k5[7] * ub7SE3
    V5_8 = k5[8] * ub8SE3
    V7_4 = k6[4] * ub4S * S26
    V7_5 = k6[5] * ub5S * S26
    V7_6 = k6[6] * ub6S * S26
    V7_7 = k6[7] * ub7S * S26
    V7_8 = k6[8] * ub8S * S26
    V8_4 = k7[4] * ub4S26S
    V8_5 = k7[5] * ub5S26S
    V8_6 = k7[6] * ub6S26S
    V8_7 = k7[7] * ub7S26S
    V8_8 = k7[8] * ub8S26S



    # define each ODE
    dubdt = V1b - (V1f) + ((V8_4)*4 + (V8_5)*5 + (V8_6)*6 + (V8_7)*7 + (V8_4)*8)
    dE1dt = V1b - V1f + V2f - V2b
    dE2dt = -V2f + V2b + V4_1 + V4_2 + V4_3 + V4_4 + V4_5 + V4_6 + V4_7 + V4_8
    dE3dt = -V3f + V3b + V5_1 + V5_2 + V5_3 + V5_4 + V5_5 + V5_6 + V5_7 + V5_8
    dS26dt = -(V7_4 + V7_5 + V7_6 + V7_7 + V7_8) + (V8_4 + V8_5 + V8_6 + V8_7 + V8_8)
    dSdt = -V3f + V3b   # need revision
    dubE1dt = V1f - V1b - V2f + V2b
    dubE2dt = V2f - V2b - V4_1 - V4_2 - V4_3 - V4_4 - V4_5 - V4_6 - V4_7 - V4_8
    dSE3dt = V3f - V3b - V4_1
    dub1SE3dt = V4_1 - V4_2 - V5_1
    dub2SE3dt = V4_2 - V4_3 - V5_2
    dub3SE3dt = V4_3 - V4_4 - V5_3
    dub4SE3dt = V4_4 - V4_5 - V5_4
    dub5SE3dt = V4_5 - V4_6 - V5_5
    dub6SE3dt = V4_6 - V4_7 - V5_6
    dub7SE3dt = V4_7 - V4_8 - V5_7
    dub8SE3dt = V4_8 - V5_8
    dub1Sdt = V5_1
    dub2Sdt = V5_2
    dub3Sdt = V5_3
    dub4Sdt = V5_4 - V7_4
    dub5Sdt = V5_5 - V7_5
    dub6Sdt = V5_6 - V7_6
    dub7Sdt = V5_7 - V7_7
    dub8Sdt = V5_8 - V7_8
    dub4S26Sdt = V7_4 - V8_4
    dub5S26Sdt = V7_5 - V8_5
    dub6S26Sdt = V7_6 - V8_6
    dub7S26Sdt = V7_7 - V8_7
    dub8S26Sdt = V7_8 - V8_8

    return [dubdt, dE1dt, dE2dt, dE3dt, dS26dt, dSdt, dubE1dt, dubE2dt, dSE3dt, dub1SE3dt, dub2SE3dt, dub3SE3dt,
            dub4SE3dt, dub5SE3dt, dub6SE3dt, dub7SE3dt, dub8SE3dt, dub1Sdt, dub2Sdt, dub3Sdt, dub4Sdt, dub5Sdt,
            dub6Sdt, dub7Sdt, dub8Sdt, dub4S26Sdt, dub5S26Sdt, dub6S26Sdt, dub7S26Sdt, dub8S26Sdt]


# initial conditions

x0 = [150, 1000, 10000, 150, 500, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



# declare a time vector (time window)
t = np.linspace(0, 36000, 36000)
#solve the ODE
x = odeint(odes, x0, t)
#define each compound
ub = x[:, 0]
E1 = x[:, 1]
E2 = x[:, 2]
E3 = x[:, 3]
S26 = x[:, 4]
S = x[:, 5]
ubE1 = x[:, 6]
ubE2 = x[:, 7]
SE3 = x[:, 8]
ub1SE3 = x[:, 9]
ub2SE3 = x[:, 10]
ub3SE3 = x[:, 11]
ub4SE3 = x[:, 12]
ub5SE3 = x[:, 13]
ub6SE3 = x[:, 14]
ub7SE3 = x[:, 15]
ub8SE3 = x[:, 16]
ub1S = x[:, 17]
ub2S = x[:, 18]
ub3S = x[:, 19]
ub4S = x[:, 20]
ub5S = x[:, 21]
ub6S = x[:, 22]
ub7S = x[:, 23]
ub8S = x[:, 24]
ub4S26S = x[:, 25]
ub5S26S = x[:, 26]
ub6S26S = x[:, 27]
ub7S26S = x[:, 28]
ub8S26S = x[:, 29]


# loop iterations counter
r = 0
c = 0
i = 0
#lists to use the compounds and their labels in the loop
prmtrs= [ub, E1, E2, E3, S26, S, SE3, ubE1, ubE2, ub1SE3, ub2SE3, ub3SE3, ub4SE3, ub5SE3, ub6SE3, ub7SE3, ub8SE3, ub1S, ub2S, ub3S, ub4S, ub5S,
         ub6S, ub7S, ub8S, ub4S26S, ub5S26S, ub6S26S, ub7S26S, ub8S26S]
prmtrs_name= ['ub', 'E1', 'E2', 'E3', 'S26', 'S', 'SE3', 'ubE1', 'ubE2', 'ub1SE3', 'ub2SE3', 'ub3SE3', 'ub4SE3', 'ub5SE3', 'ub6SE3', 'ub7SE3',
              'ub8SE3', 'ub1S', 'ub2S', 'ub3S', 'ub4S', 'ub5S', 'ub6S', 'ub7S', 'ub8S', 'ub4S26S', 'ub5S26S', 'ub6S26S', 'ub7S26S', 'ub8S26S']

#adjust the plot fonts sizes
params = {
            'axes.labelsize':10,
            'font.size':15,
            'legend.fontsize':10,
            'xtick.labelsize':8,
            'ytick.labelsize':8,
            'figure.figsize': [8,8],
        }
plt.rcParams.update(params)

#define the number of plots in rows and columns
figure, axis = plt.subplots(5, 6)

#loops to draw each plot
while r <= 4:
    c = 0
    while c <= 5:
        axis[r, c].plot(t, prmtrs[i], label=prmtrs_name[i])
        axis[r, c].set_title(prmtrs_name[i])
        axis[r, c].grid()
        #axis[r, c].legend()
        c += 1
        i += 1
    r += 1

#adjust plot position
plt.subplots_adjust(left=0.03,
                    bottom=0.032,
                    right=0.99,
                    top=0.94,
                    wspace=0.17,
                    hspace=0.7)
#uncomment the line below if you want to save the figure
#plt.savefig('ubq')
plt.show()