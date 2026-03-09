
def compute_reward(node, action):

    size = node["size"]
    parallelizable = node["parallelizable"]
    loop_carried = node["loop_carried_count"]

    instruction_cost = size * 2.0
    cache_miss_penalty = loop_carried * 3.0

    if action == 1:

        if parallelizable:
            reward = 5.0 - (instruction_cost * 0.1)
        else:
            reward = 2.0 - (cache_miss_penalty * 0.2)

    else:

        if parallelizable:
            reward = -2.0
        else:
            reward = 1.0 - (instruction_cost * 0.05)

    return reward
