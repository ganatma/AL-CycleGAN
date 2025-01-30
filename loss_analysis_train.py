import time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)
    print(opt)   
    epoch_load = np.arange(0, 201, 5)
    opt.eval = True
    opt.num_test = 80

    for count in epoch_load:
        opt.epoch = str(count)
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.eval()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.compute_alpha_beta_gamma(i,opt.epoch)           # run inference


    # #opt.epoch = str(count)
    # model = create_model(opt)      # create a model given opt.model and other options
    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    # model.eval()
    # print(opt.epoch)
    # print(opt.num_test)
    # for i, data in enumerate(dataset):
    #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #         break
    #     model.set_input(data)  # unpack data from data loader
    #     model.compute_alpha_beta_gamma(i) #,opt.epoch)           # run inference

