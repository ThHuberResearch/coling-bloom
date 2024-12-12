from sklearn.metrics import cohen_kappa_score

foo = """Apply
Analyze
Apply understand
Understand analyze
Understand apply
Analyze
Understand analyze
Understand
Understand analyze
Understand analyze
Understand analyze
Analyze understand
Apply
Understand
Remember
Remember understand
Understand apply
Understand analyze
Analyze apply
Analyze
Analyze understand
Apply analyze
Apply analyze
Apply analyze
Apply analyze
Analyze
Remember apply
Apply
Understand
Understand analyze
Analyze
Understand analyze
Understand analyze
Apply
Understand
Apply
Understand analyze
Understand remember
Understand
Analyze
Apply
Apply understand
Analyze apply"""

a1 = []
a2 = []
for line in foo.split('\n'):
    if len(line.split(' ')) == 1:
        a1.append(line)
        a2.append(line)
    else:
        t = line.split(' ')[0]
        c = line.split(' ')[1]

        # capitalize first character of c
        c = c[0].upper() + c[1:]

        a1.append(t)
        a2.append(c)

cognitive_mapping = {
    'Create': 0,
    'Evaluate': 1,
    'Analyze': 2,
    'Apply': 3,
    'Understand': 4,
    'Remember': 5
}

a1_mapped = [cognitive_mapping[t] for t in a1]
a2_mapped = [cognitive_mapping[t] for t in a2]

score = cohen_kappa_score(a1_mapped, a2_mapped)
print(score)