
class PID:
    def __init__(self, Kp=1, Ki=0, Kd=0):
        # Coefs
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.target = 0
        self._prevI = 0 # For I part
        self._prevE = 0 # For D part
        self._prev_signal = 0


    def next_signal(self, current):
        signal = 0

        #! P
        signal += self._P(current)
        #! I
        self._prevI = self._I(current)
        signal += self._prevI
        #! D
        signal += self._D(current)
        _prevE = self.target-current
        self._prev_signal = signal

        return signal


    def _P(self, current):
        return self.Kp*(current-self.target) if self.Kp != 0 else 0

    def _I(self, current):
        return self._prevI + self.Ki*(current-self.target) if self.Ki != 0 else 0

    def _D(self, current):
        return self.Kd*((current-self.target)-self._prevE) if self.Kd != 0 else 0

    # def restriction(self, signal, min, max):
    #     if signal < min:
    #         return min
    #     if signal > max:
    #         return max
    #     return signal

if __name__ == "__main__":
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    def controled_object(F0, controller, T=100):
        Ns = [-12.6*(1-math.exp(t/-32))*(1-math.exp(-2.85714)) for t in range(T)]
        Ts = np.zeros(T)
        Ts[0] = 423 + F0*Ns[0]
        for i, n in enumerate(Ns[1:]):
            Ts[i+1] = 423 + controller.next_signal(Ts[i])*n
        return Ts

    pid = PID(1)
    pid.target = 1

    Y = controled_object(1, pid, 100)

    plt.plot(Y)
    plt.show()