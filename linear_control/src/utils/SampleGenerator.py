import numpy as np
from src.utils.policies import Policy

class SampleGenerator:
    def __init__(self, problem):
        self.problem = problem
        self._generated = np.array([])

    def generate(self, num=1e6):
        experiences = []

        env = self.problem.getEnvironment()
        rep = self.problem.getRepresentation()
        gamma = self.problem.getGamma()

        behavior = self.problem.behavior

        s = env.start()
        for step in range(int(num)):
            a = behavior.selectAction(s)
            r, sp, d = env.step(a)

            g = 0 if d else gamma

            # get the observable values from the representation
            # if this is terminal, make sure the observation is an array of 0s
            obs = rep.encode(s)
            obsp = np.zeros(obs.shape) if d else rep.encode(sp)


            ex = obs, a, obsp, r, g
            experiences.append(ex)

            s = sp
            if d:
                s = env.start()

        self._generated = np.array(experiences, dtype='object')
        return self._generated

    def sample(self, samples=100, generate=1e6):
        if self._generated.shape[0] == 0:
            self.generate(generate)

        sampled_exp = np.random.randint(0, self._generated.shape[0], size=samples)
        return self._generated[sampled_exp]

class SequentialSampleGenerator(SampleGenerator):
    def __init__(self, problem):
        self.problem = problem
        self._generated = np.array([])
        self.idx = 0

    def sample(self, samples=100, generate=1e6):
        if self._generated.shape[0] == 0:
            self.generate(generate)
        elif self.idx + samples >= self._generated.shape[0]:
            self.generate(generate)
            self.idx = 0

        idx = self.idx
        self.idx += samples
        return self._generated[idx:idx+samples]
