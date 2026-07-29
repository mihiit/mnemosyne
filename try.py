from mnemosyne import MemoryStore

memory = MemoryStore()

memory.remember(
    "Fixed race condition in auth.py by adding a lock around token refresh",
    repo="my-project",
)

result = memory.recall("why does auth keep breaking", repo="my-project")
print(result["episodic_entries"])
print(result["semantic_facts"])
