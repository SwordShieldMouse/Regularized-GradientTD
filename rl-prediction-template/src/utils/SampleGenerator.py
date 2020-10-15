import numpy as np

class SampleGenerator:
    def __init__(self, problem):
        self.problem = problem
        self._generated = np.array([])

    def generate(self, num=1e6):
        experiences = []

        env = self.problem.getEnvironment()
        behavior = self.problem.behavior
        target = self.problem.target
        rep = self.problem.getRepresentation()
        gamma = self.problem.getGamma()

        s = env.start()
        for step in range(int(num)):
            a = behavior.selectAction(s)
            r, sp, d = env.step(a)

            g = 0 if d else gamma
            rho = target.ratio(behavior, s, a)

            # get the observable values from the representation
            # if this is terminal, make sure the observation is an array of 0s
            obs = rep.encode(s)
            obsp = np.zeros(obs.shape) if d else rep.encode(sp)

            ex = obs, a, obsp, r, g, rho
            experiences.append(ex)

            s = sp
            if d:
                s = env.start()

        self._generated = np.array(experiences)
        return self._generated

    def sample(self, samples=100, generate=1e6):
        if self._generated.shape[0] == 0:
            self.generate(generate)

        sampled_exp = np.random.randint(0, self._generated.shape[0], size=samples)
        return self._generated[sampled_exp]
