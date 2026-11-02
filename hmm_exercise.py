import json
import math
import itertools
from Bio import SeqIO


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

    def forward(self, x):
        if not x:
            return [], 0.0

        f = []

        # Initialization: position 0
        first_f = {}

        for k in self.states:
            first_f[k] = self.log_pi[k] + self.log_e[k][x[0]]

        f.append(first_f)

        # Fill Forward table
        for i in range(1, len(x)):
            current_f = {}

            for k in self.states:
                total = -math.inf

                for l in self.states:
                    candidate = f[i - 1][l] + self.log_a[l][k]
                    total = log_sum(total, candidate)

                current_f[k] = total + self.log_e[k][x[i]]

            f.append(current_f)

        # Calculate log2 P(x)
        log_probability = -math.inf

        for k in self.states:
            log_probability = log_sum(
                log_probability,
                f[-1][k]
            )

        return f, log_probability

    def backward(self, x):
        if not x:
            return [], 0.0

        b = [{} for _ in range(len(x))]

        # Initialization: last position
        for k in self.states:
            b[-1][k] = 0.0

        # Fill Backward table from right to left
        for i in range(len(x) - 2, -1, -1):
            for k in self.states:
                total = -math.inf

                for l in self.states:
                    candidate = (
                            self.log_a[k][l]
                            + self.log_e[l][x[i + 1]]
                            + b[i + 1][l]
                    )

                    total = log_sum(total, candidate)

                b[i][k] = total

        # Calculate log2 P(x)
        log_probability = -math.inf

        for k in self.states:
            candidate = (
                    self.log_pi[k]
                    + self.log_e[k][x[0]]
                    + b[0][k]
            )

            log_probability = log_sum(log_probability, candidate)

        return b, log_probability

    def posterior(self, x):
        f, log_probability = self.forward(x)
        b, _ = self.backward(x)

        posterior = []

        for i in range(len(x)):
            current = {}

            for k in self.states:
                current[k] = 2 ** (
                        f[i][k] + b[i][k] - log_probability
                )

            posterior.append(current)

        return posterior

    def posterior_path(self, x):
        probabilities = self.posterior(x)

        path = []

        for position in probabilities:
            best_state = self.states[0]

            for k in self.states[1:]:
                if position[k] > position[best_state]:
                    best_state = k

            path.append(best_state)

        return "".join(path)


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


def log_sum(a, b):
    if a == -math.inf:
        return b

    if b == -math.inf:
        return a

    maximum = max(a, b)
    minimum = min(a, b)

    return maximum + math.log2(1 + 2 ** (minimum - maximum))


def agreement(z1, z2):
    if len(z1) != len(z2):
        raise ValueError("Paths must have the same length.")

    if len(z1) == 0:
        return 1.0

    matches = sum(
        state1 == state2
        for state1, state2 in zip(z1, z2)
    )

    return matches / len(z1)


if __name__ == "__main__":
    toy_model = read_model("toy_model.json")
    weather_model = read_model("weather_model.json")
    # TASK 1
    print(toy_model.brute_force("phchc"))
    # TASK 2
    print(toy_model.viterbi("phchc"))
    # TASK 3
    sequences = {
        record.id: str(record.seq)
        for record in SeqIO.parse("sequences.fasta", "fasta")
    }

    true_paths = {
        record.id: str(record.seq)
        for record in SeqIO.parse("true_paths.fasta", "fasta")
    }
    # Decode simulated sequences

    print("\nID                   Length      Viterbi    Posterior")
    print("-" * 60)

    for record_id, x in sequences.items():
        true_path = true_paths[record_id]

        # Viterbi decoding
        viterbi_path, viterbi_score, _ = toy_model.viterbi(x)

        # Posterior decoding
        posterior_path = toy_model.posterior_path(x)

        # Calculate agreement
        viterbi_agreement = agreement(viterbi_path, true_path)
        posterior_agreement = agreement(posterior_path, true_path)

        # Print results
        print(
            f"{record_id:<20} "
            f"{len(x):>6} "
            f"{viterbi_agreement:>11.2%} "
            f"{posterior_agreement:>11.2%}"
        )
