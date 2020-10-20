from RlGlue import BaseAgent

# keeps a one step memory for TD based agents
class OneStepWrapper(BaseAgent):
    def __init__(self, agent, gamma, rep):
        self.agent = agent
        self.gamma = gamma
        self.rep = rep

        self.s = None
        self.a = None
        self.x = None

    def start(self, s):
        self.s = s
        self.x = self.rep.encode(s)
        self.a = self.agent.selectAction(self.x)

        return self.a

    def step(self, r, sp):
        xp = self.rep.encode(sp)

        self.agent.update(self.x, self.a, xp, r, self.gamma)

        self.s = sp
        self.a = self.agent.selectAction(xp)
        self.x = xp

        return self.a

    def end(self, r):
        gamma = 0

        self.agent.update(self.x, self.a, self.x, r, gamma)

        # reset agent here if necessary (e.g. to clear traces)

class OfflineOneStepWrapper(OneStepWrapper):
    def __init__(self, agent, behavior, gamma, rep):
        super().__init__(agent, gamma, rep)
        self.behavior = behavior

    def start(self, s):
        self.s = s
        self.x = self.rep.encode(s)
        self.a = self.behavior.selectAction(self.s)

        return self.a

    def step(self, r, sp):
        xp = self.rep.encode(sp)

        self.agent.update(self.x, self.a, xp, r, self.gamma)

        self.s = sp
        self.a = self.behavior.selectAction(sp)
        self.x = xp

        return self.a

class PolicyWrapper(BaseAgent):
    def __init__(self, agent, rep):
        self.agent = agent
        self.rep = rep

        self.s = None
        self.a = None
        self.x = None

    def start(self, s):
        self.s = s
        self.x = self.rep.encode(s)
        self.a = self.agent.selectAction(self.x)

        return self.a

    def step(self, r, sp):
        xp = self.rep.encode(sp)

        self.s = sp
        self.a = self.agent.selectAction(xp)
        self.x = xp

        return self.a

    def end(self, r):
        pass
