import pickle

with open('../examples/vision/data/cifar-10-batches-py/test_batch', 'rb') as f1:
    dict = pickle.load(f1, encoding='bytes')
    print(dict[b'data'][:5])

    with open('./cifar_test_first200.csv', 'w') as f2:
        for index in range(600, 700):
            image = dict[b'data'][index]
            label = dict[b'labels'][index]

            f2.write(str(label))
            for j in range(1024):
                f2.write(',' + ','.join(str(image[1024 * k + j]) for k in range(3)))
            f2.write('\n')
        f2.close()
    f1.close()
