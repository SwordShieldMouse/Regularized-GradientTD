import torch
import numpy as np
import torch.nn.functional as f
from TDRC.utils import getBatchColumns

class TDRC:
    def __init__(self, features, policy_net, target_net, optimizer, params, device=None):
        self.features = features
        self.params = params
        self.device = device
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer

        # regularization parameter
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.beta = params['beta']

        # secondary weights optimization parameters
        self.beta_1 = params.get('beta_1', 0.99)
        self.beta_2 = params.get('beta_2', 0.999)
        self.eps = params.get('eps', 1e-8)

        # learnable parameters for secondary weights
        self.h = torch.zeros(features, requires_grad=False).to(device)
        # ADAM optimizer parameters for secondary weights
        self.v = torch.zeros(features, requires_grad=False).to(device)
        self.m = torch.zeros(features, requires_grad=False).to(device)

    def updateNetwork(self, samples):
        # organize the mini-batch so that we can request "columns" from the data
        # e.g. we can get all of the actions, or all of the states with a single call
        batch = getBatchColumns(samples)

        # compute V(s) for each sample in mini-batch
        Vs, x = self.policy_net(batch.states)

        # by default V(s') = 0 unless the next states are non-terminal
        Vsp = torch.zeros(batch.size, device=self.device)

        # if we don't have any non-terminal next states, then no need to bootstrap
        if batch.nterm_sp.shape[0] > 0:
            Vsp, _ = self.target_net(batch.nterm_sp)

        # compute the empirical MSBE for this mini-batch and let torch auto-diff to optimize
        # don't worry about detaching the bootstrapping term for semi-gradient TD
        # the target network handles that
        target = batch.rewards + batch.gamma * Vsp.detach()
        td_loss = 0.5 * f.mse_loss(target, Vs)

        # compute E[\delta | x] ~= <h, x>
        with torch.no_grad():
            delta_hat = torch.matmul(x, self.h.t())

        # the gradient correction term is gamma * <h, x> * \nabla_w V(s')
        # to compute this gradient, we use pytorch auto-diff
        correction_loss = torch.mean(batch.gamma * delta_hat * Vsp)

        # make sure we have no gradients left over from previous update
        self.optimizer.zero_grad()
        self.target_net.zero_grad()

        # compute the entire gradient of the network using only the td error
        td_loss.backward()

        # if we have non-terminal states in the mini-batch
        # the compute the correction term using the gradient of the *target network*
        if batch.nterm_sp.shape[0] > 0:
            correction_loss.backward()

        # add the gradients of the target network for the correction term to the gradients for the td error
        for (policy_param, target_param) in zip(self.policy_net.parameters(), self.target_net.parameters()):
            policy_param.grad.add_(target_param.grad)

        # update the *policy network* using the combined gradients
        self.optimizer.step()

        # update the secondary weights using a *fixed* feature representation generated by the policy network
        with torch.no_grad():
            delta = target - Vs
            dh = (delta - delta_hat) * x - self.beta * self.h

            # ADAM optimizer
            # keep a separate set of weights for each action here as well
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * (dh**2)
            self.m = self.beta_1 * self.m + (1 - self.beta_1) * dh

            self.h = self.h + self.alpha * self.m / (torch.sqrt(self.v) + self.eps)