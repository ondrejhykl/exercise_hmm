import json
import math
import itertools


def log2(p):
    if p > 0:
        return math.log2(p)
    else:
        return -math.inf


class HMM:
    def __init__(self, states, alphabet, pi, a, e):
        self.states = states
        self.alphabet = alphabet
        self.pi = pi
        self.a = a
        self.e = e
        self.log_pi = {
            state: log2(pi[state])
            for state in states
        }
        self.log_a = {
            state: {
                next_state: log2(a[state][next_state])
                for next_state in states
            }
            for state in states
        }
        self.log_e = {
            state: {
                symbol: log2(e[state][symbol])
                for symbol in alphabet
            }
            for state in states
        }

    def joint_probability(self, x, z):
        probability = self.pi[z[0]] * self.e[z[0]][x[0]]

        for i in range(1, len(x)):
            probability *= self.a[z[i - 1]][z[i]]
            probability *= self.e[z[i]][x[i]]

        return probability

    def log_joint(self, x, z):
        probability = self.log_pi[z[0]] + self.log_e[z[0]][x[0]]

        for i in range(1, len(x)):
            probability += self.log_a[z[i - 1]][z[i]]
            probability += self.log_e[z[i]][x[i]]

        return probability

    def brute_force(self, x):
        best_path = None
        best_probability = -1
        total_probability = 0

        for path in itertools.product(self.states, repeat=len(x)):
            z = "".join(path)

            probability = self.joint_probability(x, z)

            total_probability += probability

            if probability > best_probability:
                best_probability = probability
                best_path = z

        return best_path, best_probability, total_probability

    def viterbi(self, x):
        v = []
        back = []

        first_v = {}
        first_back = {}

        for k in self.states:
            first_v[k] = self.log_pi[k] + self.log_e[k][x[0]]

        v.append(first_v)
        back.append(first_back)

        for i in range(1, len(x)):
            current_v = {}
            current_back = {}

            for k in self.states:
                best_prev = self.states[0]
                best_score = (
                        v[i - 1][best_prev] + self.log_a[best_prev][k]
                )

                for l in self.states[1:]:
                    candidate = v[i - 1][l] + self.log_a[l][k]

                    if candidate > best_score:
                        best_score = candidate
                        best_prev = l

                current_v[k] = best_score + self.log_e[k][x[i]]
                current_back[k] = best_prev

            v.append(current_v)
            back.append(current_back)

        best_state = self.states[0]
        best_score = v[-1][best_state]

        for k in self.states[1:]:
            if v[-1][k] > best_score:
                best_state = k
                best_score = v[-1][k]

        path = [best_state]

        for i in range(len(x) - 1, 0, -1):
            best_state = back[i][best_state]
            path.append(best_state)

        path.reverse()

        return "".join(path), best_score, 2**best_score


def read_model(path):
    with open(path, "r") as file:
        data = json.load(file)

    return HMM(
        data["states"],
        data["alphabet"],
        data["pi"],
        data["a"],
        data["e"]
    )


if __name__ == "__main__":
    toy_model = read_model("toy_model.json")
    weather_model = read_model("weather_model.json")
    print(toy_model.brute_force("phchc"))
    print(toy_model.viterbi("phchc"))

