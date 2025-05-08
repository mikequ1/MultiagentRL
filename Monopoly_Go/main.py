from monopoly_go import monopoly_go_v0
from monopoly_go.utils import get_action
import time
winners = [0, 0, 0]
start = time.time()

for i in range(1000):
    env = monopoly_go_v0.env(render_mode="human")
    env.reset(seed=42)

    for agent in env.agent_iter():
        # print(agent)
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            # print("GAME OVER")
            if env.winner < 4:
                winners[env.winner] += 1
            env.render()
            break
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample(info["action_mask"])
            action_name, target, _ = get_action(action)
            # print(f"Action {action}: {action_name} [target: {target}]")

        env.step(action)
        env.render()
    print(f"Game {i+1} finished.")

    env.render()
    env.close()

end = time.time()
print(f"Took {end-start:.2f} seconds")
print(winners)


