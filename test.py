import itertools

dicti = {"a": ["a1", "a2", "a3"],
        "b": ["b1", "b2"],
        "c": ["c1"]}


keys, values = zip(*dicti.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(dicti.values())
print(keys)
print(values)
print(experiments)