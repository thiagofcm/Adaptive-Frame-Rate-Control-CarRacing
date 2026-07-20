# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
os.environ["OMP_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]   = "1"
os.environ["OPENBLAS_NTHREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO as SB3PPO
from gymnasium.wrappers import TimeLimit
import envs.lunar_lander_var_fps_simple_padd
from datetime import datetime

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "LunarLander_VarFramerate_SimplePadded"
    """the id of the environment"""
    total_timesteps: int = 20000000
    """total timesteps of the experiments"""
    learning_rate: float = 3.0e-4
    """the learning rate of the optimizer"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.98
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    frame_cost: float = 2.0
    budget: float = 100.0
    max_episode_steps: int = 500
    resume_path: str = None
    """path to a checkpoint .pt file to resume training from"""

def make_env(env_id, nav_model_path, frame_cost, budget, max_episode_steps):
    def thunk():
        nav_model = NavModel(nav_model_path, device=torch.device("cpu"))
        print("Navigation Model Loaded")
        env = gym.make(env_id, frame_cost=frame_cost, budget=budget)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.navigation_model = nav_model  # ← injected here
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, lstm_hidden_size = 64):
        super().__init__()
        
        obs_dim   = np.array(envs.single_observation_space.shape).prod()

        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(64, lstm_hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_states(self, x, lstm_state, done):
        """
        Run input network + LSTM.
        Resets hidden state automatically when done=True (episode boundary).

        x:          (n_envs, obs_dim)
        lstm_state: ((1, batch(n_envs), hidden), (1, batch(n_envs), hidden))
        done:       (batch,)
        """
        hidden = self.network(x) # pass obs to the network: (n_envs,10) -> (n_envs, 64) 

        # reshape for LSTM: (seq_len, batch, input_size)
        batch_size = lstm_state[0].shape[1] #(n_envs)
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size)) #-1 means infer this dimension automatically
        done   = done.reshape((-1, batch_size))                         #-1 means infer this dimension automatically

        # Example:
        # During rollout:
        # hidden shape: (16, 64)
        # reshape(-1, 16, 64) → (1, 16, 64)   ← seq_len=1, one step at a time

        # process step by step
        # (1 - done) zeros out h and c when episode ends
        new_hidden = []
        for h, d in zip(hidden, done):
            # h, lstm_state = self.lstm(h.unsqueeze(0), # first argument  — input features, (reset_h, reset_c),  # second argument — initial hidden state (h, c)
            # the h output is the latent vector, and lastm_State is the final h,c memory updated.
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],  # reset h if done
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],  # reset c if done
                ),
            )
            new_hidden.append(h)

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs  = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

class NavAgent(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
    
class NavModel:
    """Frozen nav model wrapper with predict() interface."""
    def __init__(self, model_path, device):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        self.agent = NavAgent().to(device)
        # handle both formats
        if "model_state_dict" in checkpoint:
            self.agent.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.agent.load_state_dict(checkpoint)
        self.agent.eval()

    def predict(self, obs, deterministic=True):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.agent.actor(obs_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action.cpu().numpy()[0], None

if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.set_start_method("forkserver", force=True)
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    date_str = datetime.now().strftime("%d-%m-%H-%M-%S")
    run_name = f"{args.env_id}_fc_{args.frame_cost}_bud_{args.budget}_{date_str}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    info_file = os.path.join(f"runs/{run_name}", "info_settings.txt")
    with open(info_file, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        if args.resume_path is not None:
            f.write(f"Resumed from checkpoint: {args.resume_path}\n")
    print(f"Experiment info saved → {info_file}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    nav_model_path = "runs/LunarLander-v3__ppo__1__1779191150/model.pt"

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, nav_model_path, args.frame_cost, args.budget, args.max_episode_steps) 
        for _ in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    start_iteration = 1
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location=device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_iteration = checkpoint["iteration"] + 1
        next_lstm_state = (
            checkpoint["next_lstm_state_h"].to(device),
            checkpoint["next_lstm_state_c"].to(device),
        )
        print(f"Resumed from checkpoint: {args.resume_path} (iteration {checkpoint['iteration']}, step {global_step})")
    else:
        global_step = 0


    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    #global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Chosen FPS plots 
    episode_fps_sum   = np.zeros(args.num_envs)
    episode_fps_count = np.zeros(args.num_envs)

    for iteration in range(start_iteration, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # -----------------------------------------------------------------    
        # ROLLOUT COLLECTION.
        # -----------------------------------------------------------------
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # log chosen fps every step
            if "chosen_fps" in infos:
                for i in range(args.num_envs):
                    episode_fps_sum[i]   += infos["chosen_fps"][i]
                    episode_fps_count[i] += 1

            if "episode" in infos:
                finished = infos["episode"]["_r"]  # boolean mask — which envs finished
                for i, done in enumerate(finished):
                    if done:
                        ep_return = infos["episode"]["r"][i]
                        ep_length = infos["episode"]["l"][i]

                        # compute mean chosen fps for this episode
                        mean_fps = episode_fps_sum[i] / episode_fps_count[i] if episode_fps_count[i] > 0 else 0

                        print(f"global_step={global_step} | return={ep_return:.2f} | steps={ep_length}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
                        writer.add_scalar("charts/mean_chosen_fps", mean_fps, global_step)

                        # reset accumulators for this env
                        episode_fps_sum[i]   = 0
                        episode_fps_count[i] = 0
            
            if "frame_cost" in infos:
                writer.add_scalar("charts/frame_cost", infos["frame_cost"][0], global_step)
            if "budget" in infos:
                writer.add_scalar("charts/budget", infos["budget"][0], global_step)

        #Buffer os experience completed.

        # -----------------------------------------------------------------    
        # COMPUTING ADVANTAGES AND bootstrap value if EP in env not done
        # -----------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            #returns is a tensor of shape (1024, 16) — same shape as rewards and values.

        # Buffer with experience + advantages and returns completed.

        # -----------------------------------------------------------------    
        # PREPARING FOR UPDATE
        # -----------------------------------------------------------------
        # Flat every dimenstion of the buffer.
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones      = dones.reshape(-1)

        # Shuffle Env not timesteps
        assert args.num_envs % args.num_minibatches == 0
        envs_per_minibatch = args.num_envs // args.num_minibatches
        envs_indices       = np.arange(args.num_envs)
        timestep_env_grid  = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        # shape of timestep_env_grid: (1024, 16)
        #
        #         env0   env1   env2  ...  env15
        # step0  [   0      1      2        15]
        # step1  [  16     17     18        31]
        # step2  [  32     33     34        47]
        # ...
        # step1023 [16368 16369  ...     16383]

        # Optimizing the policy and value network
        #b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envs_indices)
            for start in range(0, args.num_envs, envs_per_minibatch):
                end = start + envs_per_minibatch
                minibatch_env_indices    = envs_indices[start:end]
                minibatch_flat_indices   = timestep_env_grid[:, minibatch_env_indices].ravel()
                minibatch_dones          = dones[:, minibatch_env_indices].reshape(-1)

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[minibatch_flat_indices],
                    # replay from initial lstm state at start of rollout
                    (
                        initial_lstm_state[0][:, minibatch_env_indices],
                        initial_lstm_state[1][:, minibatch_env_indices],
                    ),
                    minibatch_dones,
                    b_actions.long()[minibatch_flat_indices],
                )
                logratio = newlogprob - b_logprobs[minibatch_flat_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[minibatch_flat_indices]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[minibatch_flat_indices]) ** 2
                    v_clipped = b_values[minibatch_flat_indices] + torch.clamp(
                        newvalue - b_values[minibatch_flat_indices],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[minibatch_flat_indices]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[minibatch_flat_indices]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
                
        # ── Checkpoint saving ──────────────────
        if iteration % 50 == 0:  # save every 50 iterations
            checkpoint_path = f"runs/{run_name}/ckpts/timestep_{global_step}_iterations_{iteration}"
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_model_path = f"{checkpoint_path}/ckpt_{global_step}_iterations_{iteration}.pt"
            torch.save({
                "model_state_dict":     agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args":                 vars(args),
                "global_step":          global_step,
                "iteration":            iteration,
                "next_lstm_state_h":    next_lstm_state[0].cpu(),
                "next_lstm_state_c":    next_lstm_state[1].cpu(),
            }, checkpoint_model_path)
            print(f"  Checkpoint saved → {checkpoint_model_path}")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    torch.save({
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "global_step": global_step,
    }, f"runs/{run_name}/model.pt")
    print(f"Model saved → runs/{run_name}/model.pt")
    writer.close()