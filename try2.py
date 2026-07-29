from mnemosyne import MemoryStore

memory = MemoryStore()

memory.remember("Team decided to use dependency injection instead of singletons for the DB client", repo="my-project")
memory.remember("DI pattern confirmed again after a code review comment", repo="my-project")

facts = memory.force_consolidate("my-project")
print("New semantic facts:", facts)
