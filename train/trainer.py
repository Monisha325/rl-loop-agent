
import os
import json
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.loop_env import LoopDistributionEnv


def train_agent(sdg_file, output_dir):

    env = LoopDistributionEnv(sdg_file)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=0.0003
    )

    rewards = []

    for episode in range(200):

        model.learn(total_timesteps=50)

        obs,_ = env.reset()

        total_reward = 0

        while True:

            action,_ = model.predict(obs)

            obs,reward,done,_,_ = env.step(action)

            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    name = os.path.basename(sdg_file).replace("_sdg.json","")

    model.save(f"{output_dir}/models/{name}_ppo")

    plot_reward_curve(rewards,name,output_dir)

    env = LoopDistributionEnv(sdg_file)

    rl_results = evaluate_model(model,env)

    env = LoopDistributionEnv(sdg_file)

    no_opt_reward,_ = no_opt_agent(env)

    env = LoopDistributionEnv(sdg_file)

    split_reward,_ = always_split_agent(env)

    save_results(name,rl_results,no_opt_reward,split_reward,output_dir)

    plot_comparison(name,no_opt_reward,split_reward,rl_results["total_reward"],output_dir)

    return model


def evaluate_model(model,env):

    obs,_ = env.reset()

    total_reward = 0
    plan = []

    node_index = 0

    while True:

        action,_ = model.predict(obs)

        obs,reward,done,_,_ = env.step(action)

        label = env.node_labels[node_index]

        decision = "SPLIT" if action==1 else "KEEP"

        plan.append({"node":label,"action":decision})

        total_reward += reward

        node_index += 1

        if done:
            break

    return {"total_reward":float(total_reward),"distribution_plan":plan}


def no_opt_agent(env):

    obs,_ = env.reset()

    total_reward = 0
    plan = []

    node_index = 0

    while True:

        action = 0

        obs,reward,done,_,_ = env.step(action)

        label = env.node_labels[node_index]

        plan.append({"node":label,"action":"KEEP"})

        total_reward += reward

        node_index += 1

        if done:
            break

    return total_reward,plan


def always_split_agent(env):

    obs,_ = env.reset()

    total_reward = 0
    plan = []

    node_index = 0

    while True:

        action = 1

        obs,reward,done,_,_ = env.step(action)

        label = env.node_labels[node_index]

        plan.append({"node":label,"action":"SPLIT"})

        total_reward += reward

        node_index += 1

        if done:
            break

    return total_reward,plan


def plot_reward_curve(rewards,name,output_dir):

    plt.figure()

    plt.plot(rewards)

    plt.title(f"Training Reward — {name}")

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.savefig(f"{output_dir}/plots/{name}_reward.png")

    plt.close()


def plot_comparison(name,no_opt,split,rl,output_dir):

    labels = ["No Optimization","Always Split","RL Agent"]

    values = [no_opt,split,rl]

    plt.figure()

    plt.bar(labels,values,color=["red","orange","green"])

    plt.title(f"Strategy Comparison — {name}")

    plt.ylabel("Total Reward")

    plt.savefig(f"{output_dir}/plots/{name}_comparison.png")

    plt.close()


def save_results(loop_name,rl_results,no_opt,split,output_dir):

    results = {
        "loop":loop_name,
        "no_opt_reward":no_opt,
        "split_reward":split,
        "rl_reward":rl_results["total_reward"],
        "distribution_plan":rl_results["distribution_plan"]
    }

    path = f"{output_dir}/results/{loop_name}_results.json"

    with open(path,"w") as f:
        json.dump(results,f,indent=4)
