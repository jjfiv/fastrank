N = 1000
with open('docs.trecrun', 'w') as out:
    for i in range(N):
        rank = i + 1
        print("001 Q0 doc{0} {0} {1} fake".format(rank, N + 1 - rank), file=out)
