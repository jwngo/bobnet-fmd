classes = ['fabric', 'foliage', 'glass', 'leather', 'metal', 'paper',
           'plastic', 'stone', 'water', 'wood']
map = {'fabric': 0, 'foliage': 1, 'glass': 2, 'leather': 3, 
                    'metal': 4, 'paper': 5, 'plastic': 6, 'stone': 7,
                                       'water': 8, 'wood': 9}
with open('train.txt', 'w') as f:
    for c in classes:
        for i in range(0, 2000):
            print('images/{}/{}_00{:04d}.jpg {}'.format(c, c, i, map[c]), file=f)
with open('test.txt', 'w') as f:
    for c in classes:
        for i in range(2000, 2500):
            print('images/{}/{}_00{:04d}.jpg {}'.format(c, c, i, map[c]), file=f)
