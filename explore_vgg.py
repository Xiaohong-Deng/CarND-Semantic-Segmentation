import tensorflow as tf

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ['vgg16'], './data/vgg')
    graph = tf.get_default_graph()
    ops = graph.get_operations()
    for op in ops:
        print(op.name)
        print(op.values())
        print()
