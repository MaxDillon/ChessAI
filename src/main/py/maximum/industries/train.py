import concurrent.futures
import gc, getopt, os, pickle, sys, time
import numpy as np

sys.path.append('.')
sys.path.append('src/main/py')
import maximum.industries.loader as loader

# We'll use a pool of worker processes to load and transform the input data in parallel.
# This is time consuming and would otherwise stall training and cause low GPU utilization.
#
# The results from ProcessPoolExecutor work are passed between processes via pickle. If
# we pass numpy arrays by pickling them directly then we wind up with large memory leaks.
# Pickle does not seem to manage the reference counts correctly for numpy objects. We
# work around this by using numpy's own method to serialize its arrays, and then picke
# tuple containing the serialized bytes and shapes.
def pack(data):
    x, y, z = data
    return pickle.dumps((x.tobytes(), x.shape, y.tobytes(), y.shape, z.tobytes(), z.shape))

def unpack(data):
    x, xs, y, ys, z, zs = pickle.loads(data)
    return (np.frombuffer(x, dtype=loader.DTYPE).reshape(xs),
            np.frombuffer(y, dtype=loader.DTYPE).reshape(ys),
            np.frombuffer(z, dtype=loader.DTYPE).reshape(zs))

# This function will be invoked in worker subprocesses to load data in the background
# while training occurs in the main process. Each subprocess is forked and initially
# shares memory including random number generator state. We first reinitialize RNG state
# for each process (by default from /dev/urandom).
def worker_load_data(data_pattern, choose_n, from_last_n):
    np.random.set_state(np.random.RandomState().get_state())
    return pack(loader.load_balance_transform('%s.*.done' % data_pattern, choose_n, from_last_n))

def get_opt(opts, opt, opttype, default):
    if opt in opts:
        if opttype == bool:
            return True
        else:
            return opttype(opts[opt])
    else:
        return default

def main(argv):
    opts, args = getopt.getopt(argv, 'hb:c:d:f:l:o:r:s:tv:', ['ldecay=', 'rdecay=', 'data='])
    opts = dict(opts)
    if '-h' in opts:
        print('train.py [-h] // help')
        print('         [-b <batchsize>]')
        print('         [-c <config>] // filters:blocks')
        print('         [-d <device>]')
        print('         [-f <from_model>]')
        print('         [-l <lastn>]')
        print('         [-o <outdir>]')
        print('         [-r <rate>]')
        print('         [-s <saveevery>]')
        print('         [-t] // use tensorboard')
        print('         [-v <num_validation_files>]')
        print('         [--data <data>] // e.g., data/shuffled')
        print('         [--ldecay <lastn_decay>]')
        print('         [--rdecay <rate_decay>]')
        exit()
        
    batch = get_opt(opts, '-b', int, 1000)
    config = [int(x) for x in get_opt(opts, '-c', str, '160:8').split(':')]
    device = get_opt(opts, '-d', str, '0')
    from_model = get_opt(opts, '-f', str, None)
    from_last_n = get_opt(opts, '-l', int, 600)
    outdir = get_opt(opts, '-o', str, 'tfmodels')
    rate = get_opt(opts, '-r', float, 0.001)
    save_every = get_opt(opts, 's', int, 50)
    use_tensorboard = get_opt(opts, '-t', bool, False)
    num_validation = get_opt(opts, '-v', int, 0)
    data_pattern = get_opt(opts, '--data', str, 'shuffled')
    rate_decay = get_opt(opts, '--rdecay', float, 1.0)
    last_decay = get_opt(opts, '--ldecay', float, 1.0)

    # Set CUDA_DEVICE_ORDER so cuda libs number devices in the same way as nvidia-smi
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=device

    num_workers = 2 # two seem to be enough
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # pre-load the first batch of training data. 
        nextdata = [executor.submit(worker_load_data, data_pattern, 6, from_last_n)
                    for _ in range(num_workers)]
        
        # construct model after workers are forked to keep forked processes small
        import maximum.industries.modeldef as modeldef
        if from_model:
            from tensorflow.keras.models import load_model
            import tensorflow.keras.backend as K
            model = load_model(from_model)
            K.set_value(model.optimizer.lr, rate)  # override previous learning rate
        else:
            model = modeldef.make_model(filters=config[0], blocks=config[1], rate=rate)

        # load validation data if requested
        validation_data = None
        if num_validation > 0:
            xt_input, yt_value, yt_policy = loader.load_balance_transform('%s.*.test' % data_pattern, num_validation)
            validation_data = (xt_input, { 'value': yt_value, 'policy': yt_policy })

        # create tensorboard callback if requested
        callbacks=[]
        if use_tensorboard:
            from tensorflow.keras.callbacks import TensorBoard
            callbacks.append(TensorBoard(log_dir='%s/logs/%d' % (outdir, int(time.time())),
                                         histogram_freq=1,
                                         batch_size=batch,
                                         write_graph=False))
        epoch = 1
        if rate_decay < 1.0:
            from tensorflow.keras.callbacks import LearningRateScheduler
            callbacks.append(LearningRateScheduler(lambda _: rate * rate_decay ** int(epoch/20)))
            
        # training loop
        while True:
            # let 'loaded' be a list of futures with pre-loaded data
            loaded = nextdata
            # submit another round of pre-load requests
            nextdata = [executor.submit(worker_load_data, data_pattern, 6, from_last_n)
                        for _ in range(num_workers)]
            if epoch % 20 == 0:
                from_last_n = int(from_last_n * last_decay)

            for future in loaded:
                # unpack loaded data. result() will block if the data is not ready yet.
                x_input, y_value, y_policy = unpack(future.result())
                h = model.fit(x_input, { 'value': y_value, 'policy': y_policy },
                              validation_data=validation_data,
                              batch_size=batch,
                              epochs=1,
                              verbose=1,
                              callbacks=callbacks)
                if epoch % save_every == 0:
                    print('saving model after %d epochs' % epoch)
                    modeldef.save_model(model, outdir)
                epoch += 1

if __name__ == '__main__':
    main(sys.argv[1:])
