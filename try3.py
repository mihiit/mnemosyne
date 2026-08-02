from mnemosyne import MemoryStore
from mnemosyne.agent.coding_agent import MemoryAugmentedAgent

memory = MemoryStore()

# repo-a: establish a fact
agent_a = MemoryAugmentedAgent(memory, repo="repo-a")
agent_a.run_task("Team rejected Redis for rate limiting, using in-memory token bucket instead.")
agent_a.memory.force_consolidate("repo-a")

# repo-b: brand new, cold-start — should pull cross-repo priors from repo-a
agent_b = MemoryAugmentedAgent(memory, repo="repo-b")
response = agent_b.run_task("Should we use Redis for rate limiting in this new service?")
print(response)
