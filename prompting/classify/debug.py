import json

with open('output_arc_challenge/claude3_knowledge.json', 'r') as f:
    knowledge = json.load(f)

print(knowledge)