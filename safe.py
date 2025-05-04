import utils
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchdiffeq import odeint_adjoint as odeint

from model_nets import HDNet

env_name = "mountain_car"
env = utils.get_environment(env_name)
batch_size = 32
n_epochs = 1000
log_interval = 50
epsilon = 0.01
time_steps = list(np.arange(0, 1.0, 0.05))

q_test = torch.tensor(env.sample_q(1, mode="test"), dtype=torch.float)


def safe(epsilon):
    _, adj_net, hnet, hnet_decoder, _, _ = utils.get_architectures(
        arch_file="models/architectures.csv", env_name=env_name
    )

    adj_net.load_state_dict(torch.load("models/" + env_name + "/adjoint.pth"))
    hnet.load_state_dict(torch.load("models/" + env_name + "/hamiltonian_dynamics.pth"))
    HDnet = HDNet(hnet=hnet)

    optim = torch.optim.Adam(
        list(hnet.parameters()) + list(adj_net.parameters()), lr=1e-2
    )
    optim.zero_grad()
    loss_history = []
    loss_barrier_history = []
    total_loss = 0
    total_loss_barrier = 0

    for ep in range(n_epochs):
        q = torch.tensor(env.sample_q(batch_size), dtype=torch.float)
        p = adj_net(q)
        qp = torch.cat((q, p), axis=1)
        traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=False))
        loss = 0
        for i in range(batch_size):
            cost, cost_barrier = env.w(traj[:, i, :2], epsilon)
            loss += cost_barrier
            total_loss += cost / batch_size
        loss /= batch_size
        loss.backward()
        optim.step()
        optim.zero_grad()
        total_loss_barrier += loss.item()

        if ep % log_interval == log_interval - 1:
            mean_loss = total_loss / log_interval
            mean_loss_barrier = total_loss_barrier / log_interval
            print(
                f"Epoch {ep + 1}: loss = {mean_loss}, barrier loss = {mean_loss_barrier}"
            )
            loss_history.append(mean_loss.detach().numpy())
            loss_barrier_history.append(mean_loss_barrier)
            total_loss = 0
            total_loss_barrier = 0

    # Test
    p = adj_net(q_test)
    qp = torch.cat((q_test, p), axis=1)
    traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=False))
    x = traj[:, 0, 0].detach().numpy()
    return x, loss_history


x1, loss_history1 = safe(0.5)
x2, loss_history2 = safe(0.1)
x3, loss_history3 = safe(0.01)


plt.figure()
plt.plot(time_steps, x1, label="epsilon=0.5")
plt.plot(time_steps, x2, label="epsilon=0.1")
plt.plot(time_steps, x3, label="epsilon=0.01")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()

plt.figure()
plt.plot(np.arange(len(loss_history1)) * 50, loss_history1, label="epsilon=0.5")
plt.plot(np.arange(len(loss_history1)) * 50, loss_history2, label="epsilon=0.1")
plt.plot(np.arange(len(loss_history1)) * 50, loss_history3, label="epsilon=0.01")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
